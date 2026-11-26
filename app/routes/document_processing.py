
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, List, Dict, Any, Optional
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from app.db.base_db import mongo_client
from app.services.document_processing import DocumentProcessingV2
from app.services.file_management import FileStorageService, get_file_storage_service
from app.services.llm_factory import LLMFactory
from app.services.vector_store import get_vector_stores_v2
from app.core.config import settings
from app.core.safe_paths import resolve_within
from app.middleware.rate_limit import rate_limit
from fastapi import WebSocket, WebSocketDisconnect
from app.services.websocket_manager import get_websocket_manager
import asyncio



class UploadDocumentRequest(BaseModel):
    link: str = Field(..., description="Link to the document")
    document_type: Literal["pdf", "txt", "docx", "web", "md"] = Field(
        ..., description="Type of the document to be processed")


class UploadDocRequest(BaseModel):
    documents: List[UploadDocumentRequest] = Field(
        ..., description="List of documents to be processed")
    tab: Literal["data-protection", "digital-id"] = Field(..., description="Tab to process the documents")

class VisibilityType(str,Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    
class ProcessingResponse(BaseModel):
    message: str
    file_ids: List[str]
    processing_id: str
    visibility: str



router = APIRouter()

# Every uploaded file costs model calls (metadata, images, embeddings) and any website visitor can
# upload, so both the number of uploads and the number of files in one are limited.
MAX_FILES_PER_UPLOAD = 10

upload_rate_limit = rate_limit(
    "document upload", lambda: settings.upload_rate_limit_per_hour, window=3600
)


def get_mongo_client():
    return mongo_client

async def process_documents_background_v2(
    files_data: List[Dict[str, Any]],
    processing_id: str,
    file_storage_service: FileStorageService,
    collection: Any,
    vectorStore: Any,
    llm: Any,
    tab: str = "general",
    local_files: bool = False,
    allowed_root: Optional[str] = None
):
    """
    Process documents in background.

    Args:
        files_data: List of file metadata
        processing_id: Unique processing task ID
        file_storage_service: File storage service instance
        collection: MongoDB collection for task tracking
        vectorStore: Vector store instance
        llm: LLM instance
        tab: Tab/category for documents
        local_files: If True, files are on local filesystem. If False, files are in S3/remote storage.
        allowed_root: For folder processing, the folder the files must really be in. Each path is
            resolved again here, just before reading, and files that now lie outside it (for
            example replaced by a symbolic link since the request was checked) are skipped.
    """
    collection.update_one(
        {"processing_id": processing_id},
        {"$set": {"status": "processing", "updated_at": datetime.now(timezone.utc)}},
    )
    document_processor = DocumentProcessingV2(
        llm=llm,
        vector_store=vectorStore,
        file_storage=file_storage_service,  # needed to delete private uploads from S3
    )
    # Uploads are copies made for this request, so a private one is deleted once indexed. Folder
    # processing (allowed_root set) reads the operator's own files, which are never deleted.
    is_upload = allowed_root is None
    processed_documents = []
    for file_data in files_data:
        common = {
            "document_type": file_data["content_type"],
            "tab": tab,
            "client_id": file_data["client_id"],
            "visibility": file_data.get("visibility", "public"),
            "file_name": file_data.get("filename"),
        }
        if is_upload:
            common["stored_file"] = file_data["file_path"]
        if not local_files:
            # Files in S3 - generate presigned download URL
            download_path = file_storage_service.generate_download_url(
                file_path=file_data["file_path"]
            )
            processed_documents.append({"link": download_path.download_url, **common})
        else:
            # Files on local filesystem - use path directly
            file_path = file_data["file_path"]
            if allowed_root is not None:
                real_path = resolve_within(allowed_root, file_path)
                if real_path is None:
                    print(f"Skipping {file_path}: no longer inside {allowed_root}")
                    continue
                file_path = str(real_path)
            processed_documents.append({"link": file_path, **common})
    try:
        summary = document_processor.process_and_store(
            documents=processed_documents,
            local_files=local_files
        ) or {}
        failed = summary.get("failed", [])
        processed = summary.get("processed", len(processed_documents) - len(failed))
        update = {
            # Nothing indexed is an error; some files failing is reported with the names
            "status": "error" if failed and processed == 0 else "completed",
            "processed_count": processed,
            "failed_count": len(failed),
            "failed_files": failed,
            "error_message": (
                f"{len(failed)} of {len(processed_documents)} files could not be processed: "
                + ", ".join(failed)
            ) if failed else None,
            "updated_at": datetime.now(timezone.utc),
        }
        collection.update_one({"processing_id": processing_id}, {"$set": update})
    except Exception as e:
        print(f"Error processing documents for task {processing_id}: {e}")
        collection.update_one(
            {"processing_id": processing_id},
            {
                "$set": {
                    "status": "error",
                    # The exception text stays in the log: it can name storage paths and URLs
                    "error_message": "The files could not be processed. Please try again.",
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

@router.post("/upload-documents", response_model=ProcessingResponse)
async def upload_documents(
    request:Request,
    background_tasks:BackgroundTasks,
    files:List[UploadFile]=File(...),
    visibility:VisibilityType=VisibilityType.PUBLIC,
    tab:str=None,
    client_id:str=None,
    file_storage_service:FileStorageService=Depends(get_file_storage_service),
):
    if tab is None:
        tab = "general"
    if tab != "general" and tab not in settings.supported_tabs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tab. Must be one of: {', '.join(settings.supported_tabs)}"
        )
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided for upload"
        )
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time"
        )
    # Check every file before any is stored, so a refused upload leaves nothing behind
    checked_files = []
    for uploaded_file in files:
        # The file's extension decides its type: browsers disagree on what to send for .md files
        mime_type = file_storage_service.resolve_mime_type(uploaded_file.filename, uploaded_file.content_type)
        file_storage_service.validate_mime_type(mime_type)
        content = await uploaded_file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"{uploaded_file.filename} is empty")
        if len(content) > file_storage_service.max_file_size:
            limit_mb = file_storage_service.max_file_size / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"{uploaded_file.filename} is larger than the {limit_mb:g} MB limit",
            )
        checked_files.append((uploaded_file, mime_type, content))
    # Counted only now, so that a refused upload does not use up the hourly quota
    upload_rate_limit(request)
    saved_files = []
    for uploaded_file, mime_type, content in checked_files:
        if settings.use_s3:
            s3_file_path = file_storage_service.direct_file_upload(
                original_file_name=uploaded_file.filename,
                file_data=content,
                mime_type=mime_type,
            )
        else:
            s3_file_path = file_storage_service.save_file_locally(
                original_file_name=uploaded_file.filename,
                file_data=content,
                mime_type=mime_type,
            )
        saved_files.append({
            "file_path": s3_file_path,
            "filename": uploaded_file.filename,
            "content_type": mime_type,
            "client_id": client_id if visibility == VisibilityType.PRIVATE else None,
            "visibility": visibility.value,
        })
    processing_id = str(uuid.uuid4())
    document_task_collection = get_mongo_client().get_collection("document_tasks")
    document_task_record = {
        "processing_id": processing_id,
        "status": "pending",
        "file_count": len(saved_files),
        "processed_count": 0,
        "visibility": visibility.value,
        "client_id": client_id if visibility==VisibilityType.PRIVATE else None,
        "files_data": saved_files,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    document_task_collection.insert_one(document_task_record)
    background_tasks.add_task(
        process_documents_background_v2,
        saved_files,
        processing_id,
        file_storage_service,
        document_task_collection,
        get_vector_stores_v2()[visibility.value],
        # todo will have to handle this LLM to be more dynamic based on config
        LLMFactory.create_llm(),
        tab,
        not settings.use_s3  # local_files: True if not using S3, False if using S3
    )
    return ProcessingResponse(
        message="Files uploaded successfully and processing started",
        file_ids=[file["file_path"] for file in saved_files],
        processing_id=processing_id,
        visibility=visibility.value
    )  
    
class ProcessingStatusResponse(BaseModel):
    processing_id: str
    status: str
    file_count: int = 0
    processed_count: int = 0
    failed_count: int = 0
    failed_files: List[str] = []
    error_message: Optional[str] = None
    updated_at: Optional[datetime] = None


@router.get("/status/{processing_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(processing_id: str):
    """
    How an upload's indexing went. The processing_id, returned only to the uploader, is what
    identifies the task. Storage paths and the client ID stay out of the answer: only the counts,
    the names of files that failed, and a message written for users are returned.
    """
    task = get_mongo_client().get_collection("document_tasks").find_one({"processing_id": processing_id})
    if not task:
        raise HTTPException(status_code=404, detail="No upload with this ID")
    return ProcessingStatusResponse(
        processing_id=processing_id,
        status=task.get("status", "pending"),
        file_count=task.get("file_count", 0),
        processed_count=task.get("processed_count", 0),
        failed_count=task.get("failed_count", 0),
        failed_files=task.get("failed_files") or [],
        error_message=task.get("error_message"),
        updated_at=task.get("updated_at"),
    )


class ProcessLocalFolder(BaseModel):
    folder_path: str = Field(..., description="Path to the local folder containing documents")
    visibility: VisibilityType = Field(VisibilityType.PUBLIC, description="Visibility type for the documents (public or private)")
    tab: str = Field("general", description="Tab to process the documents")
    client_id: Optional[str] = Field(None, description="Client ID for private documents. Required when visibility is private.")
    
@router.post("/folder-process", response_model=ProcessingResponse)

async def process_local_folder(
    background_tasks:BackgroundTasks,
    request: ProcessLocalFolder,
    file_storage_service:FileStorageService=Depends(get_file_storage_service),
):
    visibility = request.visibility
    tab = request.tab
    client_id = request.client_id

    # This route reads files from the server's disk, so it only works inside one directory chosen
    # by the operator (INGEST_FOLDER_ROOT). Without that setting it is switched off.
    if not settings.ingest_folder_root:
        raise HTTPException(
            status_code=403,
            detail="Folder processing is disabled. Set INGEST_FOLDER_ROOT to enable it."
        )
    resolved_folder = resolve_within(settings.ingest_folder_root, request.folder_path)
    if resolved_folder is None or not resolved_folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail="folder_path must be an existing folder inside the allowed ingest directory"
        )
    folder_path = str(resolved_folder)

    # Validate tab
    if tab != "general" and tab not in settings.supported_tabs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tab. Must be one of: {', '.join(settings.supported_tabs)}"
        )

    # Validate client_id for private documents
    if visibility == VisibilityType.PRIVATE and not client_id:
        raise HTTPException(
            status_code=400,
            detail="client_id is required when visibility is 'private'"
        )

    document_processor = DocumentProcessingV2(
        llm=LLMFactory.create_llm(),
        vector_store=get_vector_stores_v2()[visibility.value],
    )
    # documents = document_processor.load_documents_from_local_fol(folder_path)
    # read all the documents from the local folder with the correct mime types
    documents = document_processor.load_file_paths(folder_path)
    # A symbolic link inside the folder could point somewhere else, so keep only files whose real
    # location is inside the folder that was asked for (which is itself inside the allowed root),
    # and from here on use that real location rather than the path as listed.
    checked = []
    for doc in documents:
        real_path = resolve_within(resolved_folder, doc["file_path"])
        if real_path is not None:
            checked.append({**doc, "file_path": str(real_path)})
    documents = checked
    if not documents or len(documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents found in the specified folder"
        )
    saved_files = []
    for doc in documents:
        saved_files.append({
            "file_path": doc["file_path"],
            "filename": doc.get("file_name", "unknown"),
            "content_type": doc.get("mime_type", "unknown"),
            "client_id": client_id if visibility == VisibilityType.PRIVATE else None,
            "visibility": visibility.value,
        })
    processing_id = str(uuid.uuid4())
    document_task_collection = get_mongo_client().get_collection("document_tasks")
    document_task_record = {
        "processing_id": processing_id,
        "status": "pending",
        "file_count": len(saved_files),
        "processed_count": 0,
        "visibility": visibility.value,
        "client_id": client_id if visibility == VisibilityType.PRIVATE else None,
        "files_data": saved_files,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    document_task_collection.insert_one(document_task_record)
    background_tasks.add_task(
        process_documents_background_v2,
        saved_files,
        processing_id,
        file_storage_service,
        document_task_collection,
        get_vector_stores_v2()[visibility.value],
        LLMFactory.create_llm(),
        tab,
        True,  # local_files: Always True for folder processing (files already on filesystem)
        str(resolved_folder),  # checked again just before the files are read
    )
    return ProcessingResponse(
        message="Folder processing started",
        file_ids=[file["file_path"] for file in saved_files],
        processing_id=processing_id,
        visibility=visibility.value
    )


  
@router.websocket("/ws/document-processing/{processing_id}")
async def document_processing_websocket(
    websocket: WebSocket,
    processing_id: str,
):
    print(f"Client connected to WebSocket for processing_id: {processing_id}")
    connection_manager = get_websocket_manager()
    await connection_manager.connect(websocket, processing_id)
    try:
        document_task_collection = get_mongo_client().get_collection("document_tasks")
        task = document_task_collection.find_one({"processing_id": processing_id})
        if not task:
            await websocket.send_json({
                "error": "Invalid processing_id",
                "message": f"No task found with processing_id: {processing_id}"
            })
            await websocket.close()
            return
        
        # Send initial status
        await websocket.send_json({
            "processing_id": processing_id,
            "status": task["status"],
            "visibility": task["visibility"],
            "file_count": task.get("file_count", 0),
            "processed_count": task.get("processed_count", 0),
            "created_at": task["created_at"].isoformat(),
            "updated_at": task["updated_at"].isoformat(),
            "client_id": task.get("client_id"),
        })
        
        # Poll for updates
        while True:
            await asyncio.sleep(2) 
            task = document_task_collection.find_one({"processing_id": processing_id})
            if not task:
                await websocket.send_json({
                    "error": "Processing task not found",
                    "message": "Task may have been deleted"
                })
                await websocket.close()
                return
            
            await websocket.send_json({
                "processing_id": processing_id,
                "status": task["status"],
                "processed_count": task.get("processed_count", 0),
                "failed_count": task.get("failed_count", 0),
                "error_message": task.get("error_message"),
                "file_count": task.get("file_count", 0),
                "updated_at": task["updated_at"].isoformat(),
                "client_id": task.get("client_id"),
            })
            
            # Close connection when processing is complete
            if task["status"] in ["completed", "error"]:
                await websocket.send_json({
                    "final": True,
                    "message": f"Processing {task['status']}"
                })
                await websocket.close()
                break
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, processing_id)
        print(f"Client disconnected from processing_id: {processing_id}")
    except Exception as e:
        connection_manager.disconnect(websocket, processing_id)
        print(f"Error in WebSocket for processing_id {processing_id}: {e}")
        import traceback
        traceback.print_exc()

            
        