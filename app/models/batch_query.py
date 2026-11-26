from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class BatchQueryJobStatus(str, Enum):
    """Status of a batch query job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class QueryResult(BaseModel):
    """Result of processing a single query."""
    query: str = Field(..., description="The original query text")
    response: str = Field(default="", description="The generated response")
    status: str = Field(default="pending", description="Status: success, failed, pending")
    error: Optional[str] = Field(default=None, description="Error message if query failed")
    processing_time: Optional[float] = Field(default=None, description="Time taken to process (seconds)")
    tab: Optional[str] = Field(default=None, description="Detected tab for this query")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are Kenya's data protection laws?",
                "response": "Kenya has comprehensive data protection legislation...",
                "status": "success",
                "error": None,
                "processing_time": 3.45,
                "tab": "data-protection"
            }
        }


class BatchQueryUploadRequest(BaseModel):
    """Request for uploading a batch of queries."""
    user_email: str = Field(..., description="User's email for notification")
    client_id: Optional[str] = Field(default=None, description="Optional client ID for multi-tenant isolation")
    visibility: str = Field(default="public", description="Visibility: public or private")
    tab: Optional[str] = Field(default=None, description="Optional tab to use for all queries (otherwise auto-detect)")

    @field_validator('visibility')
    @classmethod
    def validate_visibility(cls, v: str) -> str:
        if v not in ["public", "private"]:
            raise ValueError("Visibility must be 'public' or 'private'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_email": "user@example.com",
                "client_id": None,
                "visibility": "public",
                "tab": None
            }
        }


class BatchQueryJobResponse(BaseModel):
    """Response after creating a batch query job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: BatchQueryJobStatus = Field(..., description="Current job status")
    total_queries: int = Field(..., description="Total number of queries to process")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(default="Batch query job created successfully", description="Status message")
    delete_token: Optional[str] = Field(
        default=None,
        description="Secret needed to delete this job (send it in the X-Job-Token header). Shown only once; keep it."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "total_queries": 10,
                "created_at": "2025-11-04T12:00:00Z",
                "message": "Batch query job created successfully",
                "delete_token": "k3J...(shown once)"
            }
        }


class BatchQueryJobDetail(BaseModel):
    """Detailed information about a batch query job."""
    job_id: str = Field(..., description="Unique job identifier")
    # Optional: the email is optional at upload (the website sends none when the box is blank),
    # so a stored job can have no email. Requiring it made /status and /jobs fail with a 500.
    user_email: Optional[str] = Field(default=None, description="User's email, if one was given")
    client_id: Optional[str] = Field(default=None, description="Client ID if multi-tenant")
    status: BatchQueryJobStatus = Field(..., description="Current job status")
    total_queries: int = Field(..., description="Total number of queries")
    completed_queries: int = Field(default=0, description="Number of successfully completed queries")
    failed_queries: int = Field(default=0, description="Number of failed queries")
    queries: List[str] = Field(default_factory=list, description="List of original queries")
    results: List[QueryResult] = Field(default_factory=list, description="List of query results")
    pdf_file_path: Optional[str] = Field(default=None, description="Path to generated PDF (S3 key or local path)")
    pdf_download_url: Optional[str] = Field(default=None, description="Presigned URL for PDF download")
    visibility: str = Field(default="public", description="Visibility: public or private")
    tab: Optional[str] = Field(default=None, description="Tab if specified for all queries")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Processing completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if job failed")
    processing_time: Optional[float] = Field(default=None, description="Total processing time (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_email": "user@example.com",
                "client_id": None,
                "status": "completed",
                "total_queries": 10,
                "completed_queries": 9,
                "failed_queries": 1,
                "queries": ["What are Kenya's data protection laws?", "..."],
                "results": [],
                "pdf_file_path": "batch_reports/550e8400.pdf",
                "pdf_download_url": "https://s3.amazonaws.com/...",
                "visibility": "public",
                "tab": None,
                "created_at": "2025-11-04T12:00:00Z",
                "started_at": "2025-11-04T12:00:05Z",
                "completed_at": "2025-11-04T12:05:30Z",
                "error_message": None,
                "processing_time": 325.5
            }
        }


class BatchQueryJobList(BaseModel):
    """List of batch query jobs."""
    jobs: List[BatchQueryJobDetail] = Field(..., description="List of batch query jobs")
    total: int = Field(..., description="Total number of jobs")

    class Config:
        json_schema_extra = {
            "example": {
                "jobs": [],
                "total": 5
            }
        }


class BatchQueryJobSummary(BaseModel):
    """Summary statistics for a batch query job (used in emails)."""
    job_id: str
    total_queries: int
    completed_queries: int
    failed_queries: int
    success_rate: float  # Percentage
    processing_time: float  # Seconds
    completed_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "total_queries": 10,
                "completed_queries": 9,
                "failed_queries": 1,
                "success_rate": 90.0,
                "processing_time": 325.5,
                "completed_at": "2025-11-04T12:05:30Z"
            }
        }
