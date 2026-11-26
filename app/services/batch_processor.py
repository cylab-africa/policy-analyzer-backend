import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.services.query_handler import get_query_handler
from app.services.pdf_generator import get_pdf_generator
from app.services.email_service import get_email_service
from app.services.file_management import FileStorageService
from app.models.batch_query import BatchQueryJobStatus

# Shown to the user (the job's error_message is returned by the status endpoint). The real cause,
# which can name the storage account, bucket and user, is only written to the server log.
REPORT_FAILED_MESSAGE = (
    "The questions were answered, but the PDF report could not be saved. Please try again later."
)
# Shown for any other failure. The exception text can hold internal details (hosts, paths,
# provider messages), so it is only logged.
BATCH_FAILED_MESSAGE = "The batch could not be processed. Please try again later."
# Shown in the report for a question that could not be answered, for the same reason
QUERY_FAILED_MESSAGE = "This question could not be answered."


class ReportNotSaved(Exception):
    """The answers are ready but the PDF report could not be created or stored."""


async def process_batch_queries_background(
    job_id: str,
    queries: List[str],
    user_email: str,
    collection: Any,
    visibility: str = "public",
    tab: Optional[str] = None,
    client_id: Optional[str] = None,
    websocket_manager: Optional[Any] = None
):
    start_time = time.time()

    try:
        collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": BatchQueryJobStatus.PROCESSING.value,
                    "started_at": datetime.now(timezone.utc)
                }
            }
        )
        if websocket_manager:
            await websocket_manager.send_update(job_id, {
                "type": "batch_started",
                "job_id": job_id,
                "status": "processing",
                "total_queries": len(queries)
            })
        query_handler = get_query_handler()
        pdf_generator = get_pdf_generator()
        email_service = get_email_service()
        file_storage = FileStorageService()

        results = []
        completed_count = 0
        failed_count = 0

        for i, query in enumerate(queries):
            try:
                print(f"[BATCH {job_id}] Processing query {i + 1}/{len(queries)}: {query[:50]}...")

                query_start = time.time()
                enable_web_search = False if visibility == "private" else True

                result_dict = await query_handler.process_query_non_streaming(
                    query=query,
                    session_id=None, 
                    visibility=visibility,
                    tab=tab,
                    metadata_filters={"client_id": client_id} if client_id else None,
                    enable_web_search=enable_web_search,
                    search_tool_type="tavily"
                )

                query_end = time.time()
                processing_time = query_end - query_start

                result = {
                    "query": query,
                    "response": result_dict.get("response", ""),
                    "status": "success",
                    "error": None,
                    "processing_time": processing_time,
                    "tab": result_dict.get("tab", tab)
                }
                results.append(result)
                completed_count += 1

            except Exception as e:
                print(f"[BATCH {job_id}] Query {i + 1} failed: {e}")
                result = {
                    "query": query,
                    "response": "",
                    "status": "failed",
                    "error": QUERY_FAILED_MESSAGE,
                    "processing_time": None,
                    "tab": tab
                }
                results.append(result)
                failed_count += 1
            # Append this answer rather than rewrite the whole list: the write stays the same size
            # however many questions came before it
            collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "completed_queries": completed_count,
                        "failed_queries": failed_count,
                    },
                    "$push": {"results": result},
                }
            )
            if websocket_manager:
                await websocket_manager.send_update(job_id, {
                    "type": "batch_progress",
                    "job_id": job_id,
                    "completed": completed_count + failed_count,
                    "total": len(queries),
                    "success": completed_count,
                    "failed": failed_count
                })

        # Calculate processing time
        end_time = time.time()
        total_processing_time = end_time - start_time
        try:
            # Prepare metadata for PDF
            success_rate = (completed_count / len(queries) * 100) if len(queries) > 0 else 0
            pdf_metadata = {
                "job_id": job_id,
                "total_queries": len(queries),
                "completed_queries": completed_count,
                "failed_queries": failed_count,
                "success_rate": success_rate,
                "processing_time": total_processing_time,
                "completed_at": datetime.now(timezone.utc)
            }

            pdf_bytes = pdf_generator.generate_batch_report(queries, results, pdf_metadata)
            pdf_filename = f"{job_id}.pdf"
            pdf_folder = settings.BATCH_PDF_FOLDER
            pdf_path = f"{pdf_folder}/{pdf_filename}"

            if settings.use_s3:
                print(f"[BATCH {job_id}] Uploading PDF to S3...")
                file_storage.upload_file_bytes(
                    file_bytes=pdf_bytes,
                    file_path=pdf_path,
                    content_type="application/pdf"
                )
                pdf_download_url = file_storage.generate_presigned_url(
                    pdf_path,
                    expiration=86400  # 24 hours
                )
            else:
                print(f"[BATCH {job_id}] Saving PDF locally...")
                local_folder = Path(settings.upload_directory) / pdf_folder
                local_folder.mkdir(parents=True, exist_ok=True)
                local_pdf_path = local_folder / pdf_filename

                with open(local_pdf_path, 'wb') as f:
                    f.write(pdf_bytes)

                pdf_download_url = f"/api/batch/download/{job_id}"

            print(f"[BATCH {job_id}] PDF generated and uploaded successfully")

        except Exception as e:
            print(f"[BATCH {job_id}] Error generating PDF: {e}")
            # Without the report the job is not done: marking it "completed" would tell the user
            # a report is ready to download. Keep the answers that were produced, mark the job
            # failed, and let the handler below send the failure notifications.
            collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "completed_queries": completed_count,
                        "failed_queries": failed_count,
                        "results": results,
                        "pdf_file_path": None,
                        "processing_time": total_processing_time,
                    }
                }
            )
            raise ReportNotSaved(REPORT_FAILED_MESSAGE) from e

        # Update job as completed
        collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": BatchQueryJobStatus.COMPLETED.value,
                    "completed_queries": completed_count,
                    "failed_queries": failed_count,
                    "results": results,
                    "pdf_file_path": pdf_path,
                    "processing_time": total_processing_time,
                    "completed_at": datetime.now(timezone.utc)
                }
            }
        )

        # Send email notification
        if user_email:
            print(f"[BATCH {job_id}] Sending email notification to {user_email}...")
            try:
                job_summary = {
                    "job_id": job_id,
                    "total_queries": len(queries),
                    "completed_queries": completed_count,
                    "failed_queries": failed_count,
                    "success_rate": success_rate,
                    "processing_time": total_processing_time,
                    "completed_at": datetime.now(timezone.utc)
                }

                email_service.send_batch_complete_email(
                    recipient_email=user_email,
                    job_summary=job_summary,
                    pdf_download_url=pdf_download_url
                )
            except Exception as e:
                print(f"[BATCH {job_id}] Error sending email: {e}")

        # Send WebSocket completion notification
        if websocket_manager:
            await websocket_manager.send_update(job_id, {
                "type": "batch_completed",
                "job_id": job_id,
                "status": "completed",
                "total_queries": len(queries),
                "completed_queries": completed_count,
                "failed_queries": failed_count,
                "success_rate": success_rate,
                "processing_time": total_processing_time,
                "pdf_url": pdf_download_url
            })

        print(f"[BATCH {job_id}] Batch processing completed successfully in {total_processing_time:.2f}s")

    except Exception as e:
        # Handle catastrophic failure
        print(f"[BATCH {job_id}] Critical error: {e}")
        # Only the report failure has a message written for users; anything else is generic
        user_message = str(e) if isinstance(e, ReportNotSaved) else BATCH_FAILED_MESSAGE

        # Update job as failed
        collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": BatchQueryJobStatus.FAILED.value,
                    "error_message": user_message,
                    "completed_at": datetime.now(timezone.utc)
                }
            }
        )

        # Send failure email (only if the user gave an address)
        if user_email:
            try:
                email_service = get_email_service()
                email_service.send_batch_failed_email(
                    recipient_email=user_email,
                    job_id=job_id,
                    error_message=user_message
                )
            except Exception as email_error:
                print(f"[BATCH {job_id}] Error sending failure email: {email_error}")

        # Send WebSocket failure notification
        if websocket_manager:
            await websocket_manager.send_update(job_id, {
                "type": "batch_failed",
                "job_id": job_id,
                "status": "failed",
                "error": user_message
            })


def run_batch_processor_sync(
    job_id: str,
    queries: List[str],
    user_email: str,
    collection: Any,
    visibility: str = "public",
    tab: Optional[str] = None,
    client_id: Optional[str] = None,
    websocket_manager: Optional[Any] = None
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(
            process_batch_queries_background(
                job_id=job_id,
                queries=queries,
                user_email=user_email,
                collection=collection,
                visibility=visibility,
                tab=tab,
                client_id=client_id,
                websocket_manager=websocket_manager
            )
        )
    finally:
        loop.close()
