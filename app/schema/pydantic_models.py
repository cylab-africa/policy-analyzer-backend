from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class QueryInput(BaseModel):
    query: str
    session_id: str = Field(..., description="Unique identifier for the session")
    model_provider: str = Field(..., description="Model provider to use for the query")
    model_id: str = Field(..., description="Model ID to use for the query")


class QueryOutput(BaseModel): 
    response: str = Field(..., description="Response from the model")
    session_id: str = Field(..., description="Unique identifier for the session")
    model_provider: str = Field(..., description="Model provider used for the query")
    model_id: str = Field(..., description="Model ID used for the query")


class DocumentType(str, Enum):
    """Types of policy documents"""
    CROSS_BORDER_POLICY = "cross_border_policy"
    DIGITAL_ID_POLICY = "digital_id_policy"
    DATA_PROTECTION_POLICY = "data_protection_policy"
    

class DocumentInfo(BaseModel):
    file_name: str = Field(..., description="Name of the document file")  
    file_path: Optional[str] = Field(None, description="Path to the document file") 
    document_type: Optional[DocumentType] = Field(None, description="Type of policy document")
    country: Optional[str] = Field(None, description="Country associated with the document")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the document was uploaded")  
    

class DeleteFileRequest(BaseModel):
    file_id: int = Field(..., description="Unique identifier for the file to be deleted")


class DocumentMetadata(BaseModel):
    """Extended metadata for documents stored in vector database"""
    file_name: str
    file_path: str
    document_type: str
    country: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None


class SearchRequest(BaseModel):
    """Request model for document search"""
    query: str = Field(..., description="Search query")
    country_filter: Optional[str] = Field(None, description="Filter by country")
    document_type_filter: Optional[DocumentType] = Field(None, description="Filter by document type")
    limit: int = Field(default=10, description="Maximum number of results")


class SearchResult(BaseModel):
    """Search result model"""
    content: str = Field(..., description="Document content snippet")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    score: float = Field(..., description="Relevance score")