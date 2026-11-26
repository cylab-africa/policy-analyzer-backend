import io
import uuid
import pandas as pd
from typing import Optional, List
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Header, Query, Request
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.admin_auth import is_admin
from app.core.job_tokens import hash_token, new_token, token_matches
from app.middleware.rate_limit import rate_limit
from app.db.base_db import mongo_client
from app.models.batch_query import (
    BatchQueryJobResponse,
    BatchQueryJobDetail,
    BatchQueryJobList,
    BatchQueryJobStatus
)
from app.services.batch_processor import run_batch_processor_sync
from app.services.file_management import FileStorageService
from email_validator import EmailNotValidError, validate_email

router = APIRouter()

def get_batch_collection():
    """Get MongoDB collection for batch query jobs."""
    return mongo_client.get_collection(settings.BATCH_JOBS_COLLECTION)


upload_rate_limit = rate_limit(
    "batch upload", lambda: settings.batch_upload_rate_limit_per_hour, window=3600
)


@router.post("/upload", response_model=BatchQueryJobResponse)
async def upload_batch_queries(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Excel file with questions (single column)"),
    user_email: Optional[str] = Form(None, description="Optional user email for notifications"),
    client_id: Optional[str] = Form(None, description="Optional client ID for multi-tenant"),
    visibility: str = Form("public", description="Visibility: public or private"),
    tab: Optional[str] = Form(None, description="Optional tab for all queries (otherwise auto-detect)")
):
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")

    if visibility not in ["public", "private"]:
        raise HTTPException(status_code=400, detail="Visibility must be 'public' or 'private'")

    # Private documents belong to one client. Without a client_id the search would cover every
    # client's private documents (as the chat already requires, a private search needs one).
    client_id = (client_id or "").strip() or None
    if visibility == "private" and not client_id:
        raise HTTPException(status_code=400, detail="client_id is required when visibility is 'private'")

    user_email = (user_email or "").strip() or None
    if user_email:
        try:
            user_email = validate_email(user_email, check_deliverability=False).normalized
        except EmailNotValidError:
            raise HTTPException(status_code=400, detail="user_email is not a valid email address")

    try:
        contents = await file.read()
        # pandas no longer accepts raw bytes here, so wrap them in a file-like object
        df = pd.read_excel(io.BytesIO(contents), sheet_name=0)
        if df.empty:
            raise HTTPException(status_code=400, detail="Excel file is empty")
        queries = df.iloc[:, 0].dropna().astype(str).tolist()

        if not queries:
            raise HTTPException(status_code=400, detail="No valid questions found in Excel file")

        if len(queries) > 100:
            raise HTTPException(
                status_code=400,
                detail=f"Too many queries ({len(queries)}). Maximum allowed is 100."
            )

        # Count the upload only now that it is valid, so a wrong file type or an empty sheet does
        # not use up the hourly quota. (The body has already been received by this point either
        # way: cap upload sizes at the reverse proxy, e.g. Nginx client_max_body_size.)
        upload_rate_limit(request)

        print(f"[BATCH UPLOAD] Extracted {len(queries)} queries")
        job_id = str(uuid.uuid4())
        # Returned once to the creator; only its hash is stored (see app/core/job_tokens.py)
        delete_token = new_token()

        collection = get_batch_collection()
        job_record = {
            "job_id": job_id,
            "delete_token_hash": hash_token(delete_token),
            "user_email": user_email,
            "client_id": client_id,
            "status": BatchQueryJobStatus.PENDING.value,
            "total_queries": len(queries),
            "completed_queries": 0,
            "failed_queries": 0,
            "queries": queries,
            "results": [],
            "pdf_file_path": None,
            "visibility": visibility,
            "tab": tab,
            "created_at": datetime.now(timezone.utc),
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "processing_time": None
        }

        collection.insert_one(job_record)
        print(f"[BATCH UPLOAD] Created job record: {job_id}")

        background_tasks.add_task(
            run_batch_processor_sync,
            job_id=job_id,
            queries=queries,
            user_email=user_email,
            collection=collection,
            visibility=visibility,
            tab=tab,
            client_id=client_id,
            # No WebSocket: nothing listens for batch updates (the website polls the status
            # endpoint), and the job runs in a worker thread with its own event loop, from which
            # sending on a connection owned by the server's loop does not work.
            websocket_manager=None
        )

        return BatchQueryJobResponse(
            job_id=job_id,
            status=BatchQueryJobStatus.PENDING,
            total_queries=len(queries),
            created_at=job_record["created_at"],
            message=f"Batch query job created successfully. Processing {len(queries)} questions.",
            delete_token=delete_token
        )

    except HTTPException:
        # Validation errors raised above (empty file, too many questions) keep their status
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Excel file is empty or invalid")
    except Exception as e:
        print(f"[BATCH UPLOAD] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/status/{job_id}", response_model=BatchQueryJobDetail)
async def get_batch_job_status(job_id: str):
    """
    Get the status and details of a batch query job.

    Returns complete job information including progress, results, and PDF download URL.
    """
    collection = get_batch_collection()
    job = collection.find_one({"job_id": job_id})

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    pdf_download_url = None
    if job.get("pdf_file_path") and job.get("status") == BatchQueryJobStatus.COMPLETED.value:
        if settings.use_s3:
            try:
                file_storage = FileStorageService()
                pdf_download_url = file_storage.generate_presigned_url(
                    job["pdf_file_path"],
                    expiration=86400  # 24 hours
                )
            except Exception as e:
                print(f"Error generating presigned URL: {e}")
        else:
            pdf_download_url = f"/api/batch/download/{job_id}"

    job.pop('_id', None)
    job.pop('delete_token_hash', None)

    return BatchQueryJobDetail(
        **job,
        pdf_download_url=pdf_download_url
    )


@router.get("/download/{job_id}")
async def download_batch_report(job_id: str):

    collection = get_batch_collection()
    job = collection.find_one({"job_id": job_id})

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != BatchQueryJobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet. Current status: {job['status']}"
        )

    pdf_path = job.get("pdf_file_path")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="PDF report not found for this job")

    if settings.use_s3:
        # Redirect to S3 presigned URL
        try:
            file_storage = FileStorageService()
            presigned_url = file_storage.generate_presigned_url(pdf_path, expiration=3600)
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=presigned_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error accessing PDF: {str(e)}")
    else:
        # Serve local file
        local_file = Path(settings.upload_directory) / pdf_path
        if not local_file.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on server")

        return FileResponse(
            path=str(local_file),
            media_type="application/pdf",
            filename=f"batch_report_{job_id}.pdf"
        )


@router.get("/jobs", response_model=BatchQueryJobList)
async def list_batch_jobs(
    user_email: Optional[str] = None,
    client_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    x_admin_key: Optional[str] = Header(None, description="The operator key from ADMIN_API_KEY")
):
    # Listing returns other people's questions, answers and email addresses. The shared API key
    # is also held by the website, and a user_email or client_id filter is only what the caller
    # claims, so neither proves ownership. Listing is therefore for the operator only. A user
    # follows their own job by the job_id they got at upload (GET /status/{job_id}).
    if not is_admin(x_admin_key):
        raise HTTPException(
            status_code=403,
            detail="Listing batch jobs requires a valid X-Admin-Key header"
        )

    collection = get_batch_collection()

    # Build filter query
    filter_query = {}
    if user_email:
        filter_query["user_email"] = user_email
    if client_id:
        filter_query["client_id"] = client_id
    if status:
        if status not in [s.value for s in BatchQueryJobStatus]:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        filter_query["status"] = status

    # Get jobs from MongoDB
    cursor = collection.find(filter_query).sort("created_at", -1).skip(skip).limit(limit)
    jobs = list(cursor)

    # Get total count
    total = collection.count_documents(filter_query)

    # Remove MongoDB _id field and add download URLs
    for job in jobs:
        job.pop('_id', None)
        job.pop('delete_token_hash', None)

        # Add PDF download URL if available
        if job.get("pdf_file_path") and job.get("status") == BatchQueryJobStatus.COMPLETED.value:
            if settings.use_s3:
                try:
                    file_storage = FileStorageService()
                    job["pdf_download_url"] = file_storage.generate_presigned_url(
                        job["pdf_file_path"],
                        expiration=86400
                    )
                except Exception:
                    job["pdf_download_url"] = None
            else:
                job["pdf_download_url"] = f"/api/batch/download/{job['job_id']}"

    return BatchQueryJobList(
        jobs=[BatchQueryJobDetail(**job) for job in jobs],
        total=total
    )


@router.delete("/{job_id}")
async def delete_batch_job(
    job_id: str,
    x_job_token: Optional[str] = Header(None, description="The delete_token returned when the job was created"),
    x_admin_key: Optional[str] = Header(None, description="The operator key from ADMIN_API_KEY")
):
    collection = get_batch_collection()
    job = collection.find_one({"job_id": job_id})

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # The creator, who received the token at upload, may delete a job. The operator may delete
    # any job with the admin key, which covers jobs created before tokens existed (they have no
    # stored hash) and jobs whose creator lost the token.
    if not (token_matches(x_job_token, job.get("delete_token_hash")) or is_admin(x_admin_key)):
        raise HTTPException(
            status_code=403,
            detail="A valid X-Job-Token or X-Admin-Key header is required to delete this job"
        )

    # Delete PDF if exists
    pdf_path = job.get("pdf_file_path")
    if pdf_path:
        if settings.use_s3:
            try:
                file_storage = FileStorageService()
                file_storage.delete_file(pdf_path)
                print(f"Deleted PDF from S3: {pdf_path}")
            except Exception as e:
                print(f"Error deleting PDF from S3: {e}")
        else:
            try:
                local_file = Path(settings.upload_directory) / pdf_path
                if local_file.exists():
                    local_file.unlink()
                    print(f"Deleted local PDF: {local_file}")
            except Exception as e:
                print(f"Error deleting local PDF: {e}")

    # Delete job record
    collection.delete_one({"job_id": job_id})

    return {"message": f"Job {job_id} deleted successfully"}
