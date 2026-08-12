"""Background scheduler for search jobs.

Fixes the following pain points from the original `app.py` implementation:

    * Cancel/list/pause/resume — jobs get stable IDs and a thread-safe registry.
    * Interleaved output — a single console lock and per-job JSONL log files
      route noisy scheduled runs out of the interactive prompt.
    * Persistence — the registry snapshots to `${DATA_DIR}/jobs.json`, so
      pending schedules can be resumed after a restart.
    * Retries — each chatbot invocation is wrapped in an exponential-backoff
      retry loop so an intermittent DuckDuckGo failure does not abort a run.
    * Absolute times — jobs can be armed to fire at a specific instant.

The scheduler does not import LangGraph directly. The caller passes in an
`invoker` callable so this module remains easy to test.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import config

log = logging.getLogger(__name__)

# ---- Console lock -------------------------------------------------------------

_console_lock = threading.Lock()


def console_print(*args: Any, **kwargs: Any) -> None:
    """Thread-safe print used by every scheduler line."""
    with _console_lock:
        print(*args, **kwargs, flush=True)


def console_lock() -> threading.Lock:
    return _console_lock


# ---- Errors -------------------------------------------------------------------


class ScheduleValidationError(ValueError):
    """Raised when a caller supplies invalid schedule inputs."""


# ---- Validation ---------------------------------------------------------------


def _validate_interval(interval_minutes: object) -> float:
    if isinstance(interval_minutes, bool):
        raise ScheduleValidationError("Interval minutes must be a positive number.")
    try:
        value = float(interval_minutes)
    except (TypeError, ValueError):
        raise ScheduleValidationError("Interval minutes must be a positive number.") from None
    import math
    if not math.isfinite(value) or value <= 0:
        raise ScheduleValidationError("Interval minutes must be a positive number.")
    return value


def _validate_run_count(run_count: object) -> int:
    if isinstance(run_count, bool) or not isinstance(run_count, int):
        raise ScheduleValidationError("Number of runs must be a positive integer.")
    if run_count <= 0:
        raise ScheduleValidationError("Number of runs must be a positive integer.")
    return run_count


def validate_schedule_inputs(interval_minutes: object, run_count: object) -> tuple[float, int]:
    """Public wrapper preserved for backward-compatible tests."""
    return _validate_interval(interval_minutes), _validate_run_count(run_count)


def validate_explicit_schedule_values(
    interval_minutes: object | None, run_count: object | None
) -> None:
    """Validate individual optional caller inputs before starting a worker thread."""
    if interval_minutes is not None:
        _validate_interval(interval_minutes)
    if run_count is not None:
        _validate_run_count(run_count)


# ---- Job model ----------------------------------------------------------------


JobInvoker = Callable[[dict], dict]
"""Function that runs one chatbot invocation. Receives a state dict, returns the result."""

MessageBuilder = Callable[[str, str], list]
"""Function that produces the LangChain message list for a single run."""


@dataclass
class ScheduledSearchJob:
    """Handle for a search schedule running in a background worker.

    Fields are grouped so that JSON persistence can round-trip cleanly.
    """

    prompt: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    search_query: Optional[str] = None
    interval_minutes: Optional[float] = None
    run_count: Optional[int] = None
    absolute_start_iso: Optional[str] = None
    task_type: str = "search"
    reminder_text: Optional[str] = None
    completed_runs: int = 0
    status: str = "pending"  # pending -> running -> completed/cancelled/failed/paused
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_message: Optional[str] = None
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None

    # Runtime-only (not persisted)
    results: list[Any] = field(default_factory=list, repr=False)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    pause_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    error: Optional[BaseException] = field(default=None, repr=False)

    # ---- Control -------------------------------------------------------------

    def stop(self) -> None:
        """Request cancellation before the next run or during the interval."""
        self.stop_event.set()

    cancel = stop

    def pause(self) -> None:
        self.pause_event.set()
        if self.status in ("pending", "running"):
            self.status = "paused"

    def resume(self) -> None:
        self.pause_event.clear()
        if self.status == "paused":
            self.status = "running"

    def join(self, timeout: Optional[float] = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def done(self) -> bool:
        return self.done_event.is_set()

    # ---- Persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "search_query": self.search_query,
            "interval_minutes": self.interval_minutes,
            "run_count": self.run_count,
            "absolute_start_iso": self.absolute_start_iso,
            "task_type": self.task_type,
            "reminder_text": self.reminder_text,
            "completed_runs": self.completed_runs,
            "status": self.status,
            "created_at": self.created_at,
            "error_message": self.error_message,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ScheduledSearchJob":
        return cls(
            id=payload.get("id") or uuid.uuid4().hex[:8],
            prompt=payload.get("prompt", ""),
            search_query=payload.get("search_query"),
            interval_minutes=payload.get("interval_minutes"),
            run_count=payload.get("run_count"),
            absolute_start_iso=payload.get("absolute_start_iso"),
            task_type=payload.get("task_type", "search"),
            reminder_text=payload.get("reminder_text"),
            completed_runs=payload.get("completed_runs", 0),
            status=payload.get("status", "pending"),
            created_at=payload.get("created_at") or datetime.now(timezone.utc).isoformat(),
            error_message=payload.get("error_message"),
            last_run_at=payload.get("last_run_at"),
            next_run_at=payload.get("next_run_at"),
        )


# ---- Registry (persistence + lookups) ----------------------------------------


class JobRegistry:
    """Thread-safe registry that snapshots active jobs to disk."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.jobs_file = data_dir / "jobs.json"
        self.history_dir = data_dir / "history"
        self._jobs: dict[str, ScheduledSearchJob] = {}
        self._lock = threading.Lock()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ---- Snapshot IO ---------------------------------------------------------

    def _snapshot_locked(self) -> None:
        payload = [job.to_dict() for job in self._jobs.values()]
        tmp_path = self.jobs_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.jobs_file)

    def snapshot(self) -> None:
        with self._lock:
            self._snapshot_locked()

    def load_persisted(self) -> list[ScheduledSearchJob]:
        """Return jobs saved on disk that are not yet in memory."""
        if not self.jobs_file.is_file():
            return []
        try:
            payload = json.loads(self.jobs_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("Could not read jobs file at %s", self.jobs_file)
            return []
        return [ScheduledSearchJob.from_dict(item) for item in payload if isinstance(item, dict)]

    # ---- Job lifecycle -------------------------------------------------------

    def register(self, job: ScheduledSearchJob) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._snapshot_locked()

    def update(self, job: ScheduledSearchJob) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._snapshot_locked()

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._snapshot_locked()

    def get(self, job_id: str) -> Optional[ScheduledSearchJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[ScheduledSearchJob]:
        with self._lock:
            return list(self._jobs.values())

    def active(self) -> list[ScheduledSearchJob]:
        with self._lock:
            return [
                job for job in self._jobs.values()
                if job.status in ("pending", "running", "paused")
            ]

    # ---- History -------------------------------------------------------------

    def history_path(self, job_id: str) -> Path:
        path = self.history_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def append_run(self, job_id: str, run_number: int, content: str) -> Path:
        path = self.history_path(job_id) / f"run-{run_number:03d}.json"
        payload = {
            "job_id": job_id,
            "run_number": run_number,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


# ---- Scheduler ---------------------------------------------------------------


class Scheduler:
    """Runs `ScheduledSearchJob` instances in daemon threads.

    The scheduler is intentionally decoupled from LangGraph: the caller
    supplies an `invoker` callable that turns a state dict into a result dict.
    That makes the scheduler easy to unit test without any LLM.
    """

    def __init__(
        self,
        registry: JobRegistry,
        invoker: JobInvoker,
        message_builder: MessageBuilder,
        planner: Optional[Callable[[str], Any]] = None,
        max_invoke_retries: int = 3,
        invoke_retry_backoff: float = 2.0,
    ):
        self.registry = registry
        self.invoker = invoker
        self.message_builder = message_builder
        self.planner = planner
        self.max_invoke_retries = max(1, max_invoke_retries)
        self.invoke_retry_backoff = max(0.1, invoke_retry_backoff)

    # ---- Start / resume ------------------------------------------------------

    def start(
        self,
        prompt: str,
        interval_minutes: Optional[float] = None,
        run_count: Optional[int] = None,
        search_query: Optional[str] = None,
        absolute_start_iso: Optional[str] = None,
        job_id: Optional[str] = None,
        task_type: str = "search",
        reminder_text: Optional[str] = None,
    ) -> ScheduledSearchJob:
        validate_explicit_schedule_values(interval_minutes, run_count)

        job = ScheduledSearchJob(
            prompt=prompt,
            search_query=search_query,
            interval_minutes=interval_minutes,
            run_count=run_count,
            absolute_start_iso=absolute_start_iso,
            task_type=task_type,
            reminder_text=reminder_text,
        )
        if job_id:
            job.id = job_id
        self.registry.register(job)
        self._launch(job)
        return job

    def resume_persisted(self) -> list[ScheduledSearchJob]:
        """Reload jobs from disk and resume any that were not finished."""
        restored: list[ScheduledSearchJob] = []
        for saved in self.registry.load_persisted():
            if saved.status in ("completed", "cancelled", "failed"):
                # Keep the record but do not re-launch.
                self.registry.register(saved)
                continue
            saved.status = "pending"
            saved.stop_event = threading.Event()
            saved.pause_event = threading.Event()
            saved.done_event = threading.Event()
            self.registry.register(saved)
            self._launch(saved)
            restored.append(saved)
        return restored

    # ---- Internal ------------------------------------------------------------

    def _launch(self, job: ScheduledSearchJob) -> None:
        job.thread = threading.Thread(
            target=self._worker,
            args=(job,),
            name=f"scheduled-search-{job.id}",
            daemon=True,
        )
        job.thread.start()

    def _wait_for_start(self, job: ScheduledSearchJob) -> bool:
        """Block until absolute_start_iso is reached or the job is cancelled."""
        if not job.absolute_start_iso:
            return True
        try:
            target = datetime.fromisoformat(job.absolute_start_iso)
        except ValueError:
            return True
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        wait_seconds = (target - now).total_seconds()
        if wait_seconds <= 0:
            return True
        job.next_run_at = target.isoformat()
        job.status = "pending"
        self.registry.update(job)
        console_print(
            f"[{job.id}] Waiting until {target.isoformat()} for the first run..."
        )
        return not job.stop_event.wait(wait_seconds)

    def _invoke_with_retries(self, job: ScheduledSearchJob, state: dict) -> dict:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_invoke_retries + 1):
            if job.stop_event.is_set():
                raise RuntimeError("stopped")
            try:
                return self.invoker(state)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.max_invoke_retries:
                    break
                sleep_for = self.invoke_retry_backoff * (2 ** (attempt - 1))
                console_print(
                    f"[{job.id}] Run failed (attempt {attempt}/{self.max_invoke_retries}): "
                    f"{exc}. Retrying in {sleep_for:.1f}s."
                )
                if job.stop_event.wait(sleep_for):
                    raise RuntimeError("stopped")
        assert last_error is not None
        raise last_error

    def _resolve_missing_fields(self, job: ScheduledSearchJob) -> None:
        if job.interval_minutes and job.run_count and job.search_query:
            return
        if not self.planner:
            # Fallback: single run of the raw prompt.
            if job.search_query is None:
                job.search_query = job.prompt
            if job.run_count is None:
                job.run_count = 1
            if job.interval_minutes is None:
                job.interval_minutes = 0.0
            return
        plan = self.planner(job.prompt)
        if job.interval_minutes is None:
            job.interval_minutes = float(plan.wait_minutes) or 0.0
        if job.run_count is None:
            job.run_count = int(plan.run_count)
        if job.search_query is None:
            job.search_query = plan.search_query
        if job.absolute_start_iso is None:
            job.absolute_start_iso = plan.absolute_start_iso

    def _worker(self, job: ScheduledSearchJob) -> None:
        try:
            self._resolve_missing_fields(job)
            interval = float(job.interval_minutes or 0.0)
            runs = int(job.run_count or 1)
            if runs > 1:
                validate_schedule_inputs(interval, runs)
            else:
                _validate_run_count(runs)

            job.interval_minutes = interval
            job.run_count = runs

            if not self._wait_for_start(job):
                job.status = "cancelled"
                console_print(f"[{job.id}] Cancelled before first run.")
                self.registry.update(job)
                return

            job.status = "running"
            self.registry.update(job)

            for run_number in range(job.completed_runs + 1, runs + 1):
                # Pause: block here until resumed or cancelled.
                if job.pause_event.is_set():
                    console_print(f"[{job.id}] Paused before run {run_number}.")
                    while job.pause_event.is_set() and not job.stop_event.is_set():
                        job.stop_event.wait(0.05)

                if job.stop_event.is_set():
                    job.status = "cancelled"
                    console_print(f"[{job.id}] Stopped before run {run_number}/{runs}.")
                    self.registry.update(job)
                    return

                console_print(f"[{job.id}] Starting run {run_number}/{runs}...")
                # Message builder may accept extra task-typing kwargs; call it
                # in a way that works with both the old 2-arg and new 4-arg signatures.
                try:
                    messages = self.message_builder(
                        job.prompt,
                        job.search_query or job.prompt,
                        task_type=job.task_type,
                        reminder_text=job.reminder_text,
                    )
                except TypeError:
                    messages = self.message_builder(
                        job.prompt, job.search_query or job.prompt
                    )
                state = {"messages": messages}

                try:
                    result = self._invoke_with_retries(job, state)
                except Exception as exc:
                    if str(exc) == "stopped":
                        job.status = "cancelled"
                        console_print(f"[{job.id}] Cancelled during retry.")
                    else:
                        job.status = "failed"
                        job.error_message = str(exc)
                        job.error = exc
                        console_print(f"[{job.id}] Run {run_number} failed permanently: {exc}")
                    self.registry.update(job)
                    return

                job.results.append(result)
                job.completed_runs = run_number
                job.last_run_at = datetime.now(timezone.utc).isoformat()

                # Persist run output.
                content = _extract_content(result)
                self.registry.append_run(job.id, run_number, content)
                console_print(
                    f"[{job.id}] Run {run_number}/{runs} [{job.last_run_at}] {content}"
                )

                if run_number < runs:
                    from datetime import timedelta
                    next_run = datetime.now(timezone.utc) + timedelta(minutes=interval)
                    job.next_run_at = next_run.isoformat()
                    self.registry.update(job)
                    console_print(
                        f"[{job.id}] Waiting {interval:g} minute(s) before run "
                        f"{run_number + 1}/{runs}..."
                    )
                    if job.stop_event.wait(interval * 60):
                        job.status = "cancelled"
                        console_print(f"[{job.id}] Cancelled during wait.")
                        self.registry.update(job)
                        return
                else:
                    job.next_run_at = None
                    self.registry.update(job)

            job.status = "completed"
            self.registry.update(job)
            console_print(f"[{job.id}] Completed all {runs} run(s).")
        except ScheduleValidationError as exc:
            job.status = "failed"
            job.error_message = str(exc)
            job.error = exc
            console_print(f"[{job.id}] Invalid schedule: {exc}")
            self.registry.update(job)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error_message = str(exc)
            job.error = exc
            console_print(f"[{job.id}] Schedule failed: {exc}")
            self.registry.update(job)
        finally:
            job.done_event.set()


# ---- Utilities ----------------------------------------------------------------


def _content_to_text(content: object) -> str:
    """Flatten LangChain/GenAI message content (string or block list) to text.

    Newer `langchain-google-genai` returns `content` as a list of blocks like
    `[{'type': 'text', 'text': '...'}]`; older versions returned a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_content(result: Any) -> str:
    """Best-effort extraction of a printable assistant message from a chatbot result."""
    try:
        messages = result["messages"]
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        return _content_to_text(content)
    except (KeyError, IndexError, TypeError):
        return str(result)


def format_job_row(job: ScheduledSearchJob) -> str:
    interval = job.interval_minutes or 0
    runs = job.run_count or 0
    completed = job.completed_runs
    progress = f"{completed}/{runs}" if runs else str(completed)
    next_run = job.next_run_at or "-"
    query = (job.search_query or job.prompt)[:60]
    return (
        f"{job.id}  {job.status:<10}  runs={progress:<8}  "
        f"every={interval:g}m  next={next_run}  q={query!r}"
    )


def format_jobs_table(jobs: Iterable[ScheduledSearchJob]) -> str:
    jobs = list(jobs)
    if not jobs:
        return "No scheduled jobs."
    header = "ID        status      progress        interval    next run              query"
    lines = [header, "-" * len(header)]
    lines.extend(format_job_row(j) for j in jobs)
    return "\n".join(lines)
