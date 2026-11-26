from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import asyncio
from starlette.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
import uuid
from app.services.vector_store import  get_vector_stores_v2
from app.db.chat_history import database as chat_history_db
from app.middleware.auth import APIKeyMiddleware
from app.middleware.rate_limit import rate_limit
from pydantic import BaseModel, Field
from typing import Optional
from app.core.config import settings
from app.db.base_db import mongo_client
from app.services.info_api import router as info_router
from app.routes.document_processing import router as document_processing_router
from app.routes.admin import router as admin_router
from app.routes.batch_query import router as batch_query_router
from app.services.llm_factory import get_circuit_breaker, get_llm_metrics
from contextlib import asynccontextmanager
import json



@asynccontextmanager
async def lifespan(app: FastAPI):
    # print("Starting up the application...")
    if not mongo_client.is_connected():
        raise ConnectionError(
            "Failed to connect to MongoDB. Check your connection string and database name.")
    try:
        vector_stores = get_vector_stores_v2()
        # print(f"✓ Vector stores initialized: {list(vector_stores.keys())}")
    except Exception as e:
        print(f"✗ Error initializing vector stores: {e}")
        raise

    # Initialize query handler (singleton - happens once)
    # print("Initializing query handler...")
    from app.services.query_handler import get_query_handler
    try:
        query_handler = get_query_handler(mongo_client)
        # print("✓ Query handler initialized")
    except Exception as e:
        print(f"✗ Error initializing query handler: {e}")
        raise

    # print("✓ Application startup complete!")
    yield
    # print("Shutting down the application...")




app = FastAPI(
    title=settings.app_name,
    description="API for processing legal documents using RAG(Retrieval-Augmented Generation)",
    version=settings.app_version,
    # The docs pages and the OpenAPI file list every route, and they skip the API key check,
    # so they only exist when ENABLE_API_DOCS is set (local development).
    docs_url="/api/docs" if settings.enable_api_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.enable_api_docs else None,
    lifespan=lifespan
)

# The website reaches this API from its own server, so browsers normally never call it directly.
# Only origins listed in CORS_ALLOWED_ORIGINS get CORS headers. Authentication is a header key,
# not a cookie, so credentials are not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key", "Authorization", "X-Job-Token", "X-Admin-Key"],
)

app.add_middleware(
    APIKeyMiddleware,
    excluded_paths=["/"] + (["/api/docs", "/openapi.json"] if settings.enable_api_docs else [])
)
app.include_router(info_router, prefix="/api/info", tags=["Info API"])
app.include_router(document_processing_router,
                   prefix="/api/documents", tags=["Document Processing API"])
app.include_router(admin_router)
app.include_router(batch_query_router, prefix="/api/batch", tags=["Batch Query Processing"])


class QueryRequest(BaseModel):
    # Capped so one request cannot send an arbitrarily large prompt to the model (the website's
    # input has the same limit)
    query: str = Field(..., max_length=4000, description="The query text to process")
    tab: Optional[str] = Field(
        None, description="The tab/category for the query")
    session_id: Optional[str] = Field(
        None, description="Unique identifier for the session")
    client_id: Optional[str] = Field(
        None, description="Client ID for private document access. When provided, uses private vector store with client-specific filtering for multi-tenant isolation")
    enable_web_search: Optional[bool] = Field(
        True, description="Whether to enable web search when vector search yields insufficient results")
    search_tool: Optional[str] = Field(
        "tavily", description="Type of search tool to use if web search is enabled (tavily, serpapi, or duckduckgo)")
    max_docs_per_query: Optional[int] = Field(
        2, description="Maximum number of documents to retrieve per sub-query (default: 2)")
    max_total_docs: Optional[int] = Field(
        4, description="Maximum total unique documents to return from all sub-queries (default: 4)")
    enable_country_filter: Optional[bool] = Field(
        False, description="Enable smart metadata filtering. Automatically detects countries, document types, and other metadata from queries. When enabled, falls back to semantic search if no exact matches found. Set to false for regional/continental documents. (default: False)")


@app.get("/")
async def root():
    """Landing message. The web app lives in the separate frontend repository."""
    if settings.enable_api_docs:
        return {"message": "Policy Analyzer API. Interactive docs are at /api/docs."}
    return {"message": "Policy Analyzer API."}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session_history = chat_history_db.get_session_history(session_id)
    if not session_history:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(content=session_history)



query_rate_limit = rate_limit(
    "query", lambda: settings.query_rate_limit_per_minute, window=60
)


@app.post("/api/query_v2", dependencies=[Depends(query_rate_limit)])
async def query_rag_v2(request: QueryRequest):
    from app.services.query_handler import get_query_handler
    from fastapi.encoders import jsonable_encoder

    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    # Get query handler
    query_handler = get_query_handler(mongo_client)

    try:
        visibility = "private" if request.client_id else "public"
        metadata_filters = {"client_id": request.client_id} if request.client_id else None
        conversation_metadata = {"client_id": request.client_id} if request.client_id else {}

        # Save to SQLite for backwards compatibility (query_handler will save to MongoDB)
        chat_history_db.add_message(
            session_id=session_id,
            role="human",
            content=request.query,
            metadata=conversation_metadata
        )

        full_response = []

        async def stream_response():
            nonlocal full_response
            metadata = {
                "session_id": session_id,
                "type": "metadata",
                "system": "dynamic_rag_v2",
                "visibility": visibility,
                "client_id": request.client_id
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            effective_enable_web_search = False if request.client_id else request.enable_web_search

            try:
                async for chunk in query_handler.process_query(
                    query=request.query,
                    session_id=session_id,
                    visibility=visibility,
                    tab=request.tab,
                    metadata_filters=metadata_filters,
                    streaming=True,
                    save_conversation=False,  # We handle SQLite saving in main.py
                    enable_web_search=effective_enable_web_search,
                    search_tool_type=request.search_tool or "tavily"
                ):
                    full_response.append(chunk)

                    response_data = {
                        "content": chunk,
                        "type": "content"
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"
            except asyncio.CancelledError:
                raise  # the visitor went away: nothing to tell them
            except Exception:
                # The response has already started, so an HTTP error status is no longer
                # possible. Say so in the stream (the website shows it) instead of just stopping;
                # the details go to the log only.
                import traceback
                traceback.print_exc()
                error_event = {
                    "type": "error",
                    "message": "The answer could not be completed. Please try again.",
                }
                yield f"data: {json.dumps(error_event)}\n\n"

            # Also save to SQLite for backwards compatibility
            if full_response:
                complete_response = "".join(full_response)
                chat_history_db.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=complete_response,
                    metadata=conversation_metadata
                )

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    except asyncio.CancelledError:
        raise HTTPException(status_code=499, detail="Request cancelled by client")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        # The exception text can name hosts and internal paths: log it, do not return it
        raise HTTPException(status_code=500, detail="The question could not be processed. Please try again.")


# ============================================================================
# LLM Monitoring Endpoints
# ============================================================================

@app.get("/api/monitoring/health", tags=["Monitoring"])
async def get_llm_health():
    """
    Get health status of LLM providers including circuit breaker states.

    Returns:
        Health status for each configured LLM provider
    """
    circuit_breaker = get_circuit_breaker()
    cb_stats = circuit_breaker.get_stats()

    # Build health status
    health_status = {
        "status": "healthy",
        "providers": {}
    }

    for provider, stats in cb_stats.items():
        state = stats["state"]
        is_healthy = state == "CLOSED"

        health_status["providers"][provider] = {
            "healthy": is_healthy,
            "circuit_state": state,
            "failure_count": stats["failure_count"],
            "last_failure": stats["last_failure"].isoformat() if stats["last_failure"] else None
        }

        if not is_healthy:
            health_status["status"] = "degraded"

    return JSONResponse(content=health_status)


@app.get("/api/monitoring/metrics", tags=["Monitoring"])
async def get_llm_metrics_endpoint():
    """
    Get LLM call metrics including success rates, latency, and retry counts.

    Returns:
        Aggregated metrics for all LLM use cases
    """
    metrics = get_llm_metrics()
    metrics_data = metrics.get_stats()

    # Add summary statistics
    total_calls = sum(m["total_calls"] for m in metrics_data.values())
    total_errors = sum(m["error_count"] for m in metrics_data.values())
    total_retries = sum(m["total_retries"] for m in metrics_data.values())

    return JSONResponse(content={
        "summary": {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "total_retries": total_retries
        },
        "by_use_case": metrics_data
    })


@app.get("/api/monitoring/circuit-breaker", tags=["Monitoring"])
async def get_circuit_breaker_status():
    """
    Get detailed circuit breaker status for all providers.

    Returns:
        Circuit breaker state and configuration for each provider
    """
    circuit_breaker = get_circuit_breaker()

    return JSONResponse(content={
        "configuration": {
            "failure_threshold": circuit_breaker.failure_threshold,
            "recovery_timeout_seconds": circuit_breaker.recovery_timeout,
            "half_open_max_calls": circuit_breaker.half_open_max_calls
        },
        "providers": circuit_breaker.get_stats()
    })


