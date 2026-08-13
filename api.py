from __future__ import annotations

import hashlib
import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

import config
import ragtool
from app import _registry, chatbot, scheduler
from planner import create_search_plan, SearchPlan, SearchExecutionInstruction
from scheduler import ScheduleValidationError, _extract_content

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Auto-resume any persisted background schedules
    try:
        restored = scheduler.resume_persisted()
        print(f"Resumed {len(restored)} persisted schedule(s).")
    except Exception as exc:
        print(f"Failed to resume schedules: {exc}")
    yield
    # Shutdown: Signal background threads to stop
    for job in _registry.active():
        job.stop()

app = FastAPI(
    title="HAKATHON API",
    description="REST API wrapping the LangGraph Gemini Chatbot and Scheduler",
    version="2.0.0",
    lifespan=lifespan,
)

# ---- Schemas ----

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The user question or message")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant response content")
    plan: Dict[str, Any] = Field(..., description="The plan generated for this query")

class JobStartRequest(BaseModel):
    prompt: str = Field(..., description="Natural language scheduling instruction (e.g., 'check AAPL hourly')")

class JobStartResponse(BaseModel):
    job_id: str = Field(..., description="Unique stable ID of the started schedule")
    plan: Dict[str, Any] = Field(..., description="Parsed schedule details")

class JobStatusResponse(BaseModel):
    id: str
    prompt: str
    search_query: Optional[str]
    interval_minutes: Optional[float]
    run_count: Optional[int]
    absolute_start_iso: Optional[str]
    task_type: str
    completed_runs: int
    status: str
    created_at: str
    error_message: Optional[str]
    last_run_at: Optional[str]
    next_run_at: Optional[str]

class ActiveDocumentRequest(BaseModel):
    collection_name: Optional[str] = Field(..., description="Chroma collection name to activate, or null/empty to deactivate")

# ---- Endpoints ----

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Run a single one-off chat query through the LangGraph chatbot."""
    try:
        plan = create_search_plan(request.prompt)
    except Exception as exc:
        plan = SearchPlan(
            search_query=request.prompt,
            should_schedule=False,
            wait_minutes=0.0,
            run_count=1,
        )

    if plan.should_schedule:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This prompt asks for scheduling. Use /jobs/start to start a scheduled job."
        )

    system_messages = []
    source_label = ragtool.active_source_label()
    if source_label:
        system_messages.append(SystemMessage(content=source_label))
    system_messages.append(
        SystemMessage(content=SearchExecutionInstruction(plan.search_query).render())
    )

    try:
        out = chatbot.invoke(
            {"messages": system_messages + [HumanMessage(content=request.prompt)]}
        )
        reply = _extract_content(out)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat invocation failed: {exc}"
        )

    return ChatResponse(reply=reply, plan=plan.__dict__)

@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    """List all registered scheduled search/reminder jobs."""
    return [job.to_dict() for job in _registry.all()]

@app.post("/jobs/start", response_model=JobStartResponse)
async def start_job(request: JobStartRequest):
    """Analyze prompt and schedule a new background search or reminder job."""
    try:
        plan = create_search_plan(request.prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse scheduling instruction: {exc}"
        )

    if not plan.should_schedule:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt does not contain a scheduling command. Use /chat for immediate queries."
        )

    try:
        job = scheduler.start(
            prompt=request.prompt,
            interval_minutes=plan.wait_minutes or None,
            run_count=plan.run_count,
            search_query=plan.search_query,
            absolute_start_iso=plan.absolute_start_iso,
            task_type=plan.task_type,
            reminder_text=plan.reminder_text,
        )
    except ScheduleValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schedule validation failed: {exc}"
        )

    return JobStartResponse(job_id=job.id, plan=plan.__dict__)

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running scheduled background job."""
    job = _registry.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    job.stop()
    return {"message": f"Cancellation requested for job {job_id}"}

@app.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause an active scheduled background job."""
    job = _registry.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    job.pause()
    _registry.update(job)
    return {"message": f"Paused job {job_id}"}

@app.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused scheduled background job."""
    job = _registry.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    job.resume()
    _registry.update(job)
    return {"message": f"Resumed job {job_id}"}

@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """Get the list of completed runs and logs for a job."""
    job = _registry.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    
    history_dir = _registry.history_path(job_id)
    files = sorted(history_dir.glob("run-*.json"))
    
    logs = []
    for file in files:
        try:
            logs.append(json.loads(file.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return logs

@app.get("/documents")
async def list_documents():
    """List all indexed PDF documents in RAG."""
    return ragtool.list_indexed_documents()

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, index it using Jina Embeddings, and activate it for RAG queries."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")
    
    content = await file.read()
    sha = hashlib.sha256(content).hexdigest()
    
    uploads_dir = config.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename)
    target_path = uploads_dir / f"{sha[:16]}_{safe_name}"
    
    try:
        target_path.write_bytes(content)
        count = ragtool.index_pdf(target_path, name=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indexing failed: {exc}")
    
    collection_name = ragtool.collection_name_for_sha(sha)
    ragtool.set_active_collection(collection_name)
    
    return {
        "message": f"Successfully indexed {file.filename}",
        "chunks": count,
        "collection_name": collection_name
    }

@app.post("/documents/active")
async def set_active_document(request: ActiveDocumentRequest):
    """Select or clear the active RAG document collection."""
    ragtool.set_active_collection(request.collection_name)
    return {"message": f"Active collection set to {request.collection_name}"}
