import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, status
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileUploadResponse(BaseModel):
    upload_url: str
    file_path: str
    expires_in: int


class FileDownloadResponse(BaseModel):
    download_url: str
    expires_in: int
    note: str = "Use this URL directly without authentication headers"


class FileMetadataResponse(BaseModel):
    size: int
    content_type: str
    last_modified: datetime
    metadata: dict


class FileUploadItem(BaseModel):
    file_name: str
    mime_type: str
    file_size: Optional[int] = None


class MultiFileUploadResponse(BaseModel):
    files: List[FileUploadResponse]
    total_files: int
    successful: int
    failed: int
    errors: List[dict] = []


class FilePathResponse(BaseModel):
    original_name: str
    stored_path: str
    file_size: Optional[int] = None
    mime_type: str
    uploaded_at: datetime


class FileStorageService:

    _DEFAULT_UPLOAD_EXPIRATION = 3600  
    _DEFAULT_DOWNLOAD_EXPIRATION = 86400  
    _DEFAULT_MAX_FILE_SIZE = 10485760  

    # The documents the app can read, by file extension. The browser's own content type is not
    # reliable (Markdown arrives as text/markdown, an empty string or application/octet-stream,
    # depending on the browser and operating system), so for uploads the extension decides.
    _EXTENSION_TYPES = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    # Used when ALLOWED_FILE_TYPES is not set: exactly the types above, which the website offers
    # in its upload dialog and the document processor can read.
    _DEFAULT_ALLOWED_TYPES = ",".join(_EXTENSION_TYPES.values())
    # The old binary Word format (.doc) cannot be read, so it is refused even if listed
    _UNREADABLE_TYPES = frozenset({"application/msword"})
    # Spreadsheets are for batch questions only, never indexed as documents.
    _SPREADSHEET_TYPES = frozenset({
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    })

    def __init__(self):
        self._init_s3_client()
        self._validate_configuration()
    
    def _init_s3_client(self) -> None:
        from botocore.client import Config
        
        s3_config = {
            "region_name": settings.AWS_REGION,
            "config": Config(
                signature_version='s3v4',  # Force AWS Signature Version 4
                s3={'addressing_style': 'virtual'}  # Use virtual-hosted-style URLs
            )
        }
        if aws_access_key := settings.AWS_ACCESS_KEY_ID:
            s3_config["aws_access_key_id"] = aws_access_key
        if aws_secret_key := settings.AWS_SECRET_ACCESS_KEY:
            s3_config["aws_secret_access_key"] = aws_secret_key
        if s3_endpoint := settings.S3_ENDPOINT:
            s3_config["endpoint_url"] = s3_endpoint
        
        self.s3_client = boto3.client("s3", **s3_config)
        self.s3_resource = boto3.resource("s3", **s3_config)
        
        self.bucket_name = settings.S3_BUCKET_NAME
        self.cdn_endpoint = settings.CDN_ENDPOINT if settings.CDN_ENDPOINT else None
        self.upload_expiration = int(settings.UPLOAD_EXPIRATION) if settings.UPLOAD_EXPIRATION else self._DEFAULT_UPLOAD_EXPIRATION
        self.download_expiration = int(settings.DOWNLOAD_EXPIRATION) if settings.DOWNLOAD_EXPIRATION else self._DEFAULT_DOWNLOAD_EXPIRATION
        self.max_file_size = int(settings.MAX_FILE_SIZE) if settings.MAX_FILE_SIZE else self._DEFAULT_MAX_FILE_SIZE
        self.allowed_mime_types = self._parse_allowed_types(settings.ALLOWED_FILE_TYPES)

    @classmethod
    def _parse_allowed_types(cls, raw: Optional[str]) -> List[str]:
        """The accepted content types from ALLOWED_FILE_TYPES (comma-separated), or the defaults."""
        listed = (mime.strip() for mime in (raw or cls._DEFAULT_ALLOWED_TYPES).split(","))
        return [mime for mime in listed if mime and mime not in cls._UNREADABLE_TYPES]

    def resolve_mime_type(self, file_name: Optional[str], declared: Optional[str]) -> str:
        """The content type to use for an uploaded file: the one its extension stands for, or else
        the one the browser declared."""
        by_extension = self._EXTENSION_TYPES.get(Path(file_name or "").suffix.lower())
        return by_extension or declared or "application/octet-stream"
    
    def _validate_configuration(self) -> None:
        """Require the S3 settings only when files are stored in S3."""
        if not settings.use_s3:
            return
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is required")
        if not self.s3_client:
            raise ValueError("Failed to initialize S3 client")
    
    # S3 error codes that can mean "this upload was refused because of the ACL we asked for": the
    # bucket has ACLs turned off (Object Ownership "Bucket owner enforced", the default for new
    # buckets), or the user lacks s3:PutObjectAcl (which AWS reports as a plain AccessDenied).
    # AccessDenied is also what a bad key or a missing s3:PutObject looks like, so it only counts
    # as "the ACL's fault" once the same upload works without the ACL.
    _ACL_REJECTED = {"AccessControlListNotSupported", "AccessDenied"}
    # Once an ACL is proven to be the problem, uploads skip it for this long and then try it again,
    # so a bucket that is fixed (permission added, ownership changed) is picked up without a restart.
    _ACL_SKIP_SECONDS = 600.0
    _acl_skip_until = 0.0  # time.monotonic() value, shared by every instance
    _acl_lock = threading.Lock()  # guards _acl_skip_until (requests run in several threads)

    @staticmethod
    def _rewind(body) -> bool:
        """Make ``body`` readable from the start again. False if that is not possible."""
        if body is None or isinstance(body, (bytes, bytearray, str)):
            return True
        try:
            body.seek(0)
            return True
        except (AttributeError, OSError, ValueError):
            return False

    @classmethod
    def _remember_acl_rejected(cls, code: str) -> None:
        """Skip the ACL for a while, and say so once per period (not once per upload)."""
        with cls._acl_lock:
            already_skipping = time.monotonic() < cls._acl_skip_until
            cls._acl_skip_until = time.monotonic() + cls._ACL_SKIP_SECONDS
        if not already_skipping:
            logger.warning(
                "S3 refused the private ACL (%s), so files are uploaded without it. Who can read "
                "them is decided by the bucket's own settings (Block Public Access and policy). To "
                "use ACLs again, allow s3:PutObjectAcl for the app's user and set the bucket's "
                "Object Ownership to 'Bucket owner preferred'.", code,
            )

    def _put_object(self, **params) -> None:
        """
        Upload with ``ACL="private"``; if S3 refuses the ACL, upload again without it.

        An object without an ACL is just as private as one with ``private``: with ACLs turned off
        the bucket's own rules apply, and with ACLs on the default ACL is private. Access is
        decided by the bucket's Block Public Access setting and policy, not by this parameter.

        The ACL is only blamed, and skipped for later uploads, when the retry without it succeeds.
        If the retry fails too (for example s3:PutObject is missing), that error is raised and
        nothing about the ACL changes. Any other error is raised straight away.
        """
        refused_with = None
        if time.monotonic() >= FileStorageService._acl_skip_until:
            try:
                self.s3_client.put_object(**params, ACL="private")
                return
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                # A streamed body was used up by the first attempt: it must be rewound, and if it
                # cannot be, retrying would upload an empty file, so give up with the real error.
                if code not in self._ACL_REJECTED or not self._rewind(params.get("Body")):
                    raise
                refused_with = code
        self.s3_client.put_object(**params)
        if refused_with:
            self._remember_acl_rejected(refused_with)

    def _get_expiration_settings(self) -> dict:
        """Get current expiration settings."""
        return {
            "upload_expiration": self.upload_expiration,
            "download_expiration": self.download_expiration,
        }
    def direct_file_upload(
        self,
        original_file_name: str,  # Changed from file_path to original_file_name
        file_data: bytes,
        mime_type: str,
        use_secure_path: bool = True,
    ) -> str:  # Return the file path used
        self.validate_mime_type(mime_type)
    
        # Generate secure path if requested
        if use_secure_path:
            file_path = self._generate_secure_file_path(original_file_name)
        else:
            file_path = original_file_name
        
        metadata = self._build_metadata(original_file_name)
    
        try:
            self._put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=file_data,
                ContentType=mime_type,
                Metadata=metadata,
                ServerSideEncryption="AES256",
            )
            return file_path  # Return the path used
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not upload file: {e.response.get('Error', {}).get('Message', str(e))}",
            )
            
    def save_file_locally(
        self,
        original_file_name: str,
        file_data: bytes,
        mime_type: str,
    ) -> str:
        """Save file to local upload directory and return the file path."""
        self.validate_mime_type(mime_type)
        
        upload_dir = Path(settings.upload_directory)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        secure_file_name = self._generate_secure_file_path(original_file_name).replace("uploads/", "")
        file_path = upload_dir / secure_file_name
        
        try:
            with open(file_path, "wb") as f:
                f.write(file_data)
            return str(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not save file locally: {e}",
            )
    def load_file_paths(
        self,
        folder_path: str,
    ) -> List[dict]:
        """Load documents from a local folder, filtering by allowed mime types."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provided path '{folder_path}' is not a valid directory.",
            )
        
        accepted_mime_types = [m for m in self.allowed_mime_types if m not in self._SPREADSHEET_TYPES]
        file_list = []
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                mime_type = self._guess_mime_type(file_path)
                if mime_type in accepted_mime_types:
                    file_list.append({
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "mime_type": mime_type,
                    })
        
        return file_list
    def _guess_mime_type(self, file_path: Path) -> str:
        import mimetypes
        by_extension = self._EXTENSION_TYPES.get(file_path.suffix.lower())
        if by_extension:
            return by_extension
        mime_type, _ = mimetypes.guess_type(file_path.as_posix())
        return mime_type or "application/octet-stream"
    def _generate_secure_file_path(
        self,
        original_file_name: str,
    ) -> str:

        file_ext = Path(original_file_name).suffix
        base_name = Path(original_file_name).stem

        timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        unique_id = str(uuid4())[:8]
        return f"uploads/{base_name}_{unique_id}_{timestamp}{file_ext}"
    
    def validate_mime_type(self, mime_type: str) -> None:
        if mime_type not in self.allowed_mime_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{mime_type}' is not allowed. "
                       f"Allowed types: {', '.join(self.allowed_mime_types)}",
            )
    
    def _build_metadata(
        self,
        original_file_name: str,

    ) -> dict:
        metadata = {
            "original_filename": original_file_name,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        return metadata
    
    
    async def generate_upload_url(
        self,
        original_file_name: str,
        mime_type: str,
    ) -> FileUploadResponse:

        self.validate_mime_type(mime_type)
        file_path = self._generate_secure_file_path(original_file_name)
        metadata = self._build_metadata(original_file_name)
        
        try:
            upload_url = self.s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": file_path,
                    "ContentType": mime_type,
                    "Metadata": metadata,
                    "ServerSideEncryption": "AES256",
                    # No ACL here, unlike _put_object: this link is signed ahead of time, so it
                    # cannot fall back if the bucket refuses the ACL. The file is private anyway.
                },
                ExpiresIn=self.upload_expiration,
            )
            
            return FileUploadResponse(
                upload_url=upload_url,
                file_path=file_path,
                expires_in=self.upload_expiration,
            )
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not generate upload URL: {e.response.get('Error', {}).get('Message', str(e))}",
            )
    
    
    def generate_download_url(
        self,
        file_path: str
    ) -> FileDownloadResponse:
        """Generate a presigned download URL that can be used directly without authentication."""
        try:
            download_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": file_path,
                },
                ExpiresIn=self.download_expiration,
            )
            if self.cdn_endpoint:
                s3_endpoint = self.s3_client.meta.endpoint_url or f"https://{self.bucket_name}.s3.amazonaws.com"
                download_url = download_url.replace(s3_endpoint, self.cdn_endpoint, 1)
            
            return FileDownloadResponse(
                download_url=download_url,
                expires_in=self.download_expiration,
            )
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not generate download URL: {str(e)}",
            )
    
    async def delete_file(
        self,
        file_path: str,
    ) -> None:
        self.delete_file_sync(file_path)

    def delete_file_sync(self, file_path: str) -> None:
        """
        Delete an object from the bucket. The S3 call itself is blocking either way, so this form
        is for code that runs outside an event loop, such as background ingestion, which runs in a
        worker thread where there is no loop to run delete_file on.
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_path)
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not delete file: {str(e)}",
            )
    
    async def get_file_metadata(
        self,
        file_path: str,
    ) -> FileMetadataResponse:
        response = await self._verify_file_access(file_path)
        
        return FileMetadataResponse(
            size=response.get("ContentLength", 0),
            content_type=response.get("ContentType", "application/octet-stream"),
            last_modified=response.get("LastModified", datetime.now(timezone.utc)),
            metadata=response.get("Metadata", {}),
        )
    
    def generate_public_file_url(self, file_path: str) -> str:
        if self.cdn_endpoint:
            return f"{self.cdn_endpoint}/{file_path}"
        bucket = self.bucket_name
        region = settings.AWS_REGION or "us-east-1"
        return f"https://{bucket}.s3.{region}.amazonaws.com/{file_path}"
    
    def generate_public_url_with_cache_busting(
        self,
        file_path: str,
        last_modified: Optional[datetime] = None,
    ) -> str:
        base_url = self.generate_public_file_url(file_path)
        
        if last_modified:
            timestamp = int(last_modified.timestamp() * 1000)
        else:
            timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        return f"{base_url}?v={timestamp}"
    
    def validate_file(self, file_size: int, mime_type: str) -> None:
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum limit of {self.max_file_size} bytes",
            )
        self.validate_mime_type(mime_type)
    
    async def generate_multi_upload_urls(
        self,
        files: List[FileUploadItem]
    ) -> MultiFileUploadResponse:
    
        successful_uploads: List[FileUploadResponse] = []
        errors: List[dict] = []
        
        for idx, file_item in enumerate(files):
            try:
                # Validate file if size is provided
                if file_item.file_size:
                    self.validate_file(file_item.file_size, file_item.mime_type)
                
                # Generate upload URL
                upload_response = await self.generate_upload_url(
                    original_file_name=file_item.file_name,
                    mime_type=file_item.mime_type,
                )
                successful_uploads.append(upload_response)
                
            except HTTPException as e:
                errors.append({
                    "index": idx,
                    "file_name": file_item.file_name,
                    "error": e.detail,
                    "status_code": e.status_code,
                })
            except Exception as e:
                errors.append({
                    "index": idx,
                    "file_name": file_item.file_name,
                    "error": str(e),
                    "status_code": 500,
                })
        
        return MultiFileUploadResponse(
            files=successful_uploads,
            total_files=len(files),
            successful=len(successful_uploads),
            failed=len(errors),
            errors=errors,
        )
    
    async def track_uploaded_files(
        self,
        file_paths: List[str],
    ) -> List[FilePathResponse]:
        tracked_files: List[FilePathResponse] = []
        for file_path in file_paths:
            try:
                metadata = await self.get_file_metadata(file_path)
                
                original_name = metadata.metadata.get(
                    "original_filename",
                    Path(file_path).name
                )
                
                tracked_files.append(FilePathResponse(
                    original_name=original_name,
                    stored_path=file_path,
                    file_size=metadata.size,
                    mime_type=metadata.content_type,
                    uploaded_at=metadata.last_modified,
                ))
            except HTTPException:
                # Skip files that can't be accessed
                continue
        
        return tracked_files
    
    async def delete_multiple_files(
        self,
        file_paths: List[str],
    ) -> dict:

        deleted = []
        failed = []
        
        for file_path in file_paths:
            try:
                await self.delete_file(file_path)
                deleted.append(file_path)
            except HTTPException as e:
                failed.append({
                    "file_path": file_path,
                    "error": e.detail,
                    "status_code": e.status_code,
                })
            except Exception as e:
                failed.append({
                    "file_path": file_path,
                    "error": str(e),
                    "status_code": 500,
                })
        
        return {
            "total": len(file_paths),
            "deleted": len(deleted),
            "failed": len(failed),
            "deleted_paths": deleted,
            "errors": failed,
        }
    
    async def generate_multi_download_urls(
        self,
        file_paths: List[str]
    ) -> List[dict]:

        download_urls = []

        for file_path in file_paths:
            try:
                result = await self.generate_download_url(file_path)
                download_urls.append({
                    "file_path": file_path,
                    "download_url": result.download_url,
                    "expires_in": result.expires_in,
                    "success": True,
                })
            except HTTPException as e:
                download_urls.append({
                    "file_path": file_path,
                    "error": e.detail,
                    "status_code": e.status_code,
                    "success": False,
                })

        return download_urls

    def upload_file_bytes(
        self,
        file_bytes: bytes,
        file_path: str,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload file bytes directly to S3.

        Args:
            file_bytes: The file content as bytes
            file_path: The S3 key/path for the file
            content_type: MIME type of the file

        Returns:
            The file path used in S3
        """
        # Extract original filename from path for metadata
        original_file_name = Path(file_path).name

        try:
            self._put_object(
                Bucket=self.bucket_name,
                Key=file_path,
                Body=file_bytes,
                ContentType=content_type,
                Metadata=self._build_metadata(original_file_name),
                ServerSideEncryption="AES256",
            )
            return file_path
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not upload file: {e.response.get('Error', {}).get('Message', str(e))}",
            )

    def generate_presigned_url(
        self,
        file_path: str,
        expiration: int = None
    ) -> str:
        """
        Generate a presigned download URL.

        Args:
            file_path: The S3 key/path for the file
            expiration: URL expiration time in seconds (defaults to download_expiration)

        Returns:
            Presigned URL string
        """
        if expiration is None:
            expiration = self.download_expiration

        try:
            download_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": file_path,
                },
                ExpiresIn=expiration,
            )

            # Apply CDN endpoint if configured
            if self.cdn_endpoint:
                s3_endpoint = self.s3_client.meta.endpoint_url or f"https://{self.bucket_name}.s3.amazonaws.com"
                download_url = download_url.replace(s3_endpoint, self.cdn_endpoint, 1)

            return download_url
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not generate presigned URL: {str(e)}",
            )


_file_storage_service: Optional[FileStorageService] = None


def get_file_storage_service() -> FileStorageService:
    """Get or create file storage service singleton."""
    global _file_storage_service
    if _file_storage_service is None:
        _file_storage_service = FileStorageService()
    return _file_storage_service
