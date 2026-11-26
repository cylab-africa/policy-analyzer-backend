import typing
from typing import List, Dict, Any, Optional
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import TextLoader
import json
from app.services.vector_store import get_vector_stores, VectorStoreV2
from app.services.file_management import FileStorageService
from app.services.pdf_images import PdfImageParser
from datetime import datetime
import traceback
import os
from time import time
import zipfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# A .docx is a zip file, so a small upload can unpack into far more than the size limit allows.
# This caps the unpacked text parts (the ones the loader reads into memory).
MAX_DOCX_TEXT_BYTES = 30 * 1024 * 1024


def _loggable(link: str) -> str:
    """
    A link as it may be logged: without its query string. In a presigned S3 link that part lets
    anyone who reads it download the file until it expires, and it names the access key.
    """
    return str(link).split("?", 1)[0]


class DocumentProcessingV2:
    def __init__(self, llm:Any, vector_store:VectorStoreV2, file_storage: Optional[FileStorageService] = None):
        self.llm = llm
        self.vector_store = vector_store
        self.file_storage = file_storage

    def load_file_paths(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Load all processable files from a local folder.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            List of dictionaries containing file_path, file_name, and mime_type

        Raises:
            ValueError: If folder doesn't exist or is not accessible
        """
        import mimetypes

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Supported file extensions and their MIME types
        supported_extensions = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': DOCX_MIME_TYPE,
            '.md': 'text/markdown',
        }

        files_info = []

        try:
            # Walk through directory recursively
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()

                    # Check if file extension is supported
                    if file_ext in supported_extensions:
                        mime_type = supported_extensions[file_ext]

                        # Verify file is readable
                        if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                            files_info.append({
                                "file_path": file_path,
                                "file_name": filename,
                                "mime_type": mime_type
                            })
                        else:
                            print(f"⚠ Warning: File not readable: {file_path}")

            print(f"✓ Found {len(files_info)} processable files in {folder_path}")
            return files_info

        except Exception as e:
            print(f"✗ Error scanning folder {folder_path}: {e}")
            raise ValueError(f"Error scanning folder: {str(e)}")

    def _delete_stored_file(self, stored_file: str, local_files: bool) -> bool:
        """
        Remove the uploaded copy of a private document once it has been indexed. Returns True
        when it is gone. Runs synchronously: ingestion runs in a worker thread with no event loop.
        """
        try:
            if local_files:
                if os.path.exists(stored_file):
                    os.remove(stored_file)
                return True
            if not self.file_storage:
                print("⚠ Warning: File storage service not available; private upload not deleted")
                return False
            self.file_storage.delete_file_sync(stored_file)
            return True
        except Exception as e:
            print(f"⚠ Warning: Could not delete private upload {stored_file}: {e}")
            return False

    def _detect_metadata(self, document_text):
        """Extract metadata from document text. Limits text to first 3000 chars to avoid timeouts."""
        # Truncate text to avoid overwhelming the LLM
        text_sample = document_text[:3000] if len(document_text) > 3000 else document_text
        
        messages = [
            ("system", """You are a metadata extraction assistant. Extract metadata from the document link provided.
                        Return ONLY valid JSON in the following format:
                        {
                            "country": "string",
                            "isoAlpha2": "string",
                            "title": "string",
                            "description": "string"
                        }"""),
            ("user", f"Extract metadata from this document: {text_sample}"),
        ]
        
        try:
            response = self.llm.invoke(messages)
            metadata = response.content.strip()
            
            # Remove markdown code blocks if present
            if metadata.startswith("```json"):
                metadata = metadata[7:]  
            elif metadata.startswith("```"):
                metadata = metadata[3:] 
            
            if metadata.endswith("```"):
                metadata = metadata[:-3]  
            
            metadata = metadata.strip()
            
            metadata_dict = json.loads(metadata)
            print(f"✓ Metadata extracted successfully")
            return metadata_dict
            
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to parse metadata JSON: {metadata[:200]}")
            print(f"JSON Error: {e}")
            return {
                "country": "Unknown",
                "isoAlpha2": "XX",
                "title": "Unknown Document",
                "description": "Metadata extraction failed"
            }
        except Exception as e:
            print(f"⚠ Error during metadata extraction: {e}")
            return {
                "country": "Unknown",
                "isoAlpha2": "XX",
                "title": "Unknown Document",
                "description": f"Error: {str(e)}"
            }
    def process_and_store(
        self,
        documents: List[Dict[str, Any]],
        local_files: bool = False,
        local_path: str = None,
    ) -> Dict[str, Any]:
        """
        Process and store documents in the vector store.

        Parameters:
        documents (List[Dict[str, Any]]): List of documents to be processed.
            Required fields: 'link', 'document_type'
            Optional fields: 'client_id', 'tab', 'visibility', 'file_name', 'stored_file'
            - file_name: the name shown when reporting which files failed.
            - stored_file: the uploaded copy of the file (S3 key, or local path when local_files
              is True). For a private document it is deleted once the document is indexed.
              Documents without it (such as files from an operator's ingest folder) are never
              deleted.
        local_files (bool): Whether files are local (True) or remote/S3 (False)
        local_path (str, optional): Unused, kept for existing callers.

        Returns:
        A summary: {"processed": int, "failed": [file names], "total": int,
        "private_not_deleted": [stored files that could not be removed]}. One file failing does
        not stop the others, so callers must check "failed" rather than rely on no exception.
        """
        for idx, doc in enumerate(documents):
            if "link" not in doc.keys() or "document_type" not in doc.keys():
                raise ValueError(
                    f"Document at index {idx} is missing required fields: 'link' or 'document_type'")
            if "client_id" not in doc.keys():
                doc["client_id"] = "default_client"
            if "tab" not in doc.keys():
                doc["tab"] = "general"
            if "visibility" not in doc.keys():
                doc["visibility"] = "public"  # Default to public if not specified
        accepted_types = ["pdf", "txt", "docx", "web", "md", "application/pdf", DOCX_MIME_TYPE, "text/plain", "text/markdown"]
        for doc in documents:
            if doc["document_type"] not in accepted_types:
                raise ValueError(
                    f"Document type '{doc['document_type']}' is not supported. Supported types are: {accepted_types}")

        processed_count = 0
        failed: List[str] = []
        private_files_to_delete: List[str] = []
        for doc in documents:
            name = doc.get("file_name") or os.path.basename(str(doc.get("stored_file") or "")) or "document"
            try:
                if doc["document_type"] == "application/pdf" or doc["document_type"] == "pdf":
                    documents_data = self._process_pdf(doc, file_local=local_files)
                elif doc["document_type"] == "text/plain" or doc["document_type"] == "txt":
                    documents_data = self._process_txt(doc, file_local=local_files)
                elif doc["document_type"] == DOCX_MIME_TYPE or doc["document_type"] == "docx":
                    documents_data = self._process_docx(doc, file_local=local_files)
                elif doc["document_type"] == "text/markdown" or doc["document_type"] == "md":
                    documents_data = self._process_md(doc, file_local=local_files)
                else:  # "web": accepted by the request model but not processed
                    raise ValueError("Web links are not processed")

                if not documents_data:
                    raise ValueError("No text could be read from the file")

                from datetime import timezone
                processing_time = datetime.now(timezone.utc).isoformat()
                metadata_to_add = {
                    "tab": doc["tab"],
                    "client_id": doc["client_id"],
                    "date_processed": processing_time,
                    "visibility": doc["visibility"]
                }
                for document in documents_data:
                    document.metadata.update(metadata_to_add)
                documents_chunks = self.chunk_documents(documents_data)
                self.vector_store.get_vector_store().add_documents(documents_chunks)
                print(f"✓ Added {len(documents_chunks)} chunks from {name} to the vector store")

                # Only the uploaded copy is removed, and only once it is indexed
                if doc["visibility"] == "private" and doc.get("stored_file"):
                    private_files_to_delete.append(doc["stored_file"])

                processed_count += 1
            except Exception as doc_error:
                # Details (which can include a presigned URL) go to the log only
                print(f"✗ Error processing {name}: {doc_error}")
                import traceback
                traceback.print_exc()
                failed.append(name)

        not_deleted = [
            stored for stored in private_files_to_delete
            if not self._delete_stored_file(stored, local_files)
        ]
        if private_files_to_delete:
            print(f"✓ Deleted {len(private_files_to_delete) - len(not_deleted)} of "
                  f"{len(private_files_to_delete)} private uploads after indexing")

        return {
            "total": len(documents),
            "processed": processed_count,
            "failed": failed,
            "private_not_deleted": not_deleted,
        }

    def _process_pdf(self, doc: Dict[str, Any], file_local=False) -> Dict[str, Any]:
        print("Processing PDF document...")
        print(f"Document link: {_loggable(doc['link'])}")

        if not file_local:
            file_path, should_cleanup = self._download_file_remote(doc["link"], "temp.pdf", is_local=False)
        else:
            file_path, should_cleanup = self._download_file_remote(doc["link"], "temp.pdf", is_local=True)

        try:
            print("Loading PDF with PyMuPDF4LLM...")
            loader = PyMuPDF4LLMLoader(
                file_path,
                mode="page",
                extract_images=True,
                images_parser=PdfImageParser(model=self.llm))
            documents = loader.load()
            print(f"✓ Processed {len(documents)} pages from PDF document.")

            print("Extracting metadata from first page...")
            document_metadata = self._detect_metadata(documents[0].page_content)

            print("Adding metadata to all pages...")
            for document in documents:
                document.metadata.update(document_metadata)

            print(f"✓ PDF processing complete. Ready to chunk.")
            return documents
        except Exception as e:
            print(f"✗ Error loading PDF document: {e}")
            raise e
        finally:
            if should_cleanup and os.path.exists(file_path):
                os.remove(file_path)
                print("✓ Temporary PDF file cleaned up")
    def _download_file_remote(self, link, document_name: str, is_local: bool = False) -> tuple[str, bool]:
        """
        Download a file from a remote URL or handle local file path.
        For local files, returns the path directly without copying.
        For remote URLs, downloads the file.

        Returns:
            tuple[str, bool]: (file_path, should_cleanup)
                - file_path: Path to the file to process
                - should_cleanup: Whether to delete the file after processing
        """
        try:
            # Handle local file paths - read directly without copying
            if is_local or not link.startswith(('http://', 'https://')):
                if os.path.exists(link):
                    print(f"✓ Using local file: {link}")
                    return (link, False)  # Don't cleanup local files
                else:
                    raise FileNotFoundError(f"Local file not found: {link}")

            # Handle remote URLs - download to temp location
            import requests
            session = requests.Session()
            session.headers.clear()

            try:
                response = session.get(link, allow_redirects=True)
                response.raise_for_status()
            except requests.RequestException as error:
                # The original message repeats the whole link, and so would the traceback
                status = getattr(error.response, "status_code", None)
                raise RuntimeError(
                    f"Could not download the file: {type(error).__name__}"
                    + (f" (HTTP {status})" if status else "")
                ) from None

            # Write to a fresh file in the system temp directory, not the working directory: the
            # app's own directory is read-only in the container, and a fixed name such as
            # temp.pdf would let two documents processed at once overwrite each other.
            import tempfile
            suffix = os.path.splitext(document_name)[1]
            fd, download_path = tempfile.mkstemp(prefix="pdp_download_", suffix=suffix)
            with os.fdopen(fd, 'wb') as file:
                file.write(response.content)

            print(f"✓ Downloaded file from URL")
            return (download_path, True)  # Cleanup downloaded files
        except Exception as e:
            print(f"✗ Error accessing file from {_loggable(link)}: {e}")
            raise e
    def _process_txt(self, doc: Dict[str, Any], file_local=False) -> Dict[str, Any]:
        print("Processing TXT document...")
        print(f"Document link: {_loggable(doc['link'])}")

        if not file_local:
            file_path, should_cleanup = self._download_file_remote(
                doc["link"], f"temp_{time()}.txt", is_local=False)
        else:
            file_path, should_cleanup = self._download_file_remote(
                doc["link"], f"temp_{time()}.txt", is_local=True)

        try:
            print("Loading TXT with TextLoader...")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            print(f"✓ Processed TXT document.")

            print("Extracting metadata from document...")
            document_metadata = self._detect_metadata(documents[0].page_content)

            print("Adding metadata...")
            for document in documents:
                document.metadata.update(document_metadata)

            print(f"✓ TXT processing complete. Ready to chunk.")
            return documents
        except Exception as e:
            print(f"✗ Error loading TXT document: {e}")
            raise e
        finally:
            if should_cleanup and os.path.exists(file_path):
                os.remove(file_path)
                print("✓ Temporary TXT file cleaned up")

    def _check_docx_size(self, file_path: str) -> None:
        """Refuse a .docx whose text would unpack to more than MAX_DOCX_TEXT_BYTES."""
        with zipfile.ZipFile(file_path) as archive:
            # By name, once per listing, which is how the loader reads them (names can repeat)
            unpacked = sum(
                archive.getinfo(name).file_size for name in archive.namelist()
                if name == "word/document.xml" or name.startswith(("word/header", "word/footer"))
            )
        if unpacked > MAX_DOCX_TEXT_BYTES:
            raise ValueError(f"The document's text unpacks to {unpacked} bytes, over the limit")

    def _process_docx(self, doc: Dict[str, Any], file_local=False) -> Dict[str, Any]:
        print("Processing DOCX document...")
        print(f"Document link: {_loggable(doc['link'])}")

        from langchain_community.document_loaders import Docx2txtLoader

        if not file_local:
            file_path, should_cleanup = self._download_file_remote(
                doc["link"], f"temp_{time()}.docx", is_local=False)
        else:
            file_path, should_cleanup = self._download_file_remote(
                doc["link"], f"temp_{time()}.docx", is_local=True)

        try:
            self._check_docx_size(file_path)
            print("Loading DOCX with Docx2txtLoader...")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            print(f"✓ Processed DOCX document.")

            print("Extracting metadata from document...")
            document_metadata = self._detect_metadata(documents[0].page_content)

            print("Adding metadata...")
            for document in documents:
                document.metadata.update(document_metadata)

            print(f"✓ DOCX processing complete. Ready to chunk.")
            return documents
        except Exception as e:
            print(f"✗ Error loading DOCX document: {e}")
            raise e
        finally:
            if should_cleanup and os.path.exists(file_path):
                os.remove(file_path)
                print("✓ Temporary DOCX file cleaned up")

    def _process_web(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        try:    
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(doc["link"])
            documents = loader.load()
            document_metadata = self._detect_metadata(documents[0].page_content)
            for document in documents:
                document.metadata.update(document_metadata)
            return documents
        except Exception as e:
            print(f"Error processing WEB document: {e}")
            raise e
        finally:
            pass  # No downloaded file to remove for web documents

    def _process_md(self, doc: Dict[str, Any], file_local=False) -> Dict[str, Any]:
        print("Processing MD document...")
        print(f"Document link: {_loggable(doc['link'])}")

        if not file_local:
            file_path, should_cleanup = self._download_file_remote(
                doc["link"], f"temp_{time()}.md", is_local=False)
        else:
            file_path, should_cleanup = self._download_file_remote(
                doc["link"], f"temp_{time()}.md", is_local=True)

        try:
            # Markdown is plain text, and keeping its headings and lists helps the search
            print("Loading MD with TextLoader...")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            print(f"✓ Processed MD document.")

            print("Extracting metadata from document...")
            document_metadata = self._detect_metadata(documents[0].page_content)

            print("Adding metadata...")
            for document in documents:
                document.metadata.update(document_metadata)

            print(f"✓ MD processing complete. Ready to chunk.")
            return documents
        except Exception as e:
            print(f"✗ Error loading MD document: {e}")
            raise e
        finally:
            if should_cleanup and os.path.exists(file_path):
                os.remove(file_path)
                print("✓ Temporary MD file cleaned up")

    def chunk_documents(self, documents: List[Document], chunk_size: int = 1500, overlay: int = 100) -> List[Document]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=overlay)
            chunked_documents = text_splitter.split_documents(documents)
            return chunked_documents
        except Exception as e:
            print(f"✗ Error chunking documents: {e}")
            import traceback
            traceback.print_exc()
            raise e

