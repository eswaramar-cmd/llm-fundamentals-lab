"""Modular evidence sources for production incident investigation.

Provides abstract interfaces and concrete V1 implementations for extracting
evidence strictly from existing project logs and metrics without external databases
or hardcoded scenarios.
"""

from __future__ import annotations

import collections
import datetime
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.agent.config import PROJECT_ROOT, get_settings
from backend.agent.metrics import get_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-Memory Log Ring Buffer (captures logs emitted during runtime)
# ---------------------------------------------------------------------------

class LogRecordEntry:
    """Normalized log record representation."""

    def __init__(
        self,
        timestamp: str,
        level: str,
        logger_name: str,
        message: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        status: Optional[int] = None,
        duration: Optional[float] = None,
        error_type: Optional[str] = None,
        raw_text: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.level = level.upper()
        self.logger_name = logger_name
        self.message = message
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id
        self.endpoint = endpoint
        self.status = status
        self.duration = duration
        self.error_type = error_type
        self.raw_text = raw_text or message

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "logger": self.logger_name,
            "message": self.message,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "endpoint": self.endpoint,
            "status": self.status,
            "duration": self.duration,
            "error_type": self.error_type,
        }


class InMemoryLogBufferHandler(logging.Handler):
    """Logging handler that retains recent log entries in an in-memory deque."""

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self._capacity = capacity
        self.buffer: collections.deque[LogRecordEntry] = collections.deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat()
            msg = record.getMessage()
            entry = LogRecordEntry(
                timestamp=ts,
                level=record.levelname,
                logger_name=record.name,
                message=msg,
                request_id=getattr(record, "request_id", None),
                session_id=getattr(record, "session_id", None),
                user_id=getattr(record, "user_id", None),
                endpoint=getattr(record, "endpoint", None),
                status=getattr(record, "status", None),
                duration=getattr(record, "duration", None),
                error_type=getattr(record, "error_type", None) or (
                    type(record.exc_info[1]).__name__ if record.exc_info and record.exc_info[1] else None
                ),
                raw_text=msg,
            )
            self.buffer.append(entry)
        except Exception:
            self.handleError(record)


_LOG_BUFFER_HANDLER: Optional[InMemoryLogBufferHandler] = None


def get_log_buffer_handler() -> InMemoryLogBufferHandler:
    """Return singleton in-memory log buffer handler and ensure it is attached to root logger."""
    global _LOG_BUFFER_HANDLER
    if _LOG_BUFFER_HANDLER is None:
        _LOG_BUFFER_HANDLER = InMemoryLogBufferHandler(capacity=2000)
        root = logging.getLogger()
        if _LOG_BUFFER_HANDLER not in root.handlers:
            root.addHandler(_LOG_BUFFER_HANDLER)
    return _LOG_BUFFER_HANDLER


# Ensure buffer handler is initialized
get_log_buffer_handler()


# ---------------------------------------------------------------------------
# Incident Parameters & Source Result
# ---------------------------------------------------------------------------

@dataclass
class IncidentQueryParams:
    """Dynamic runtime parameters supplied by the user."""

    service_name: Optional[str] = None
    error_message: Optional[str] = None
    time_range: Optional[str] = None
    severity: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    details: Optional[str] = None
    user_id: Optional[str] = "default_user"


@dataclass
class SourceResult:
    """Evidence collected from an individual source."""

    source_name: str
    available: bool
    status_message: str
    observed_facts: list[str] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    timeline_events: list[dict[str, Any]] = field(default_factory=list)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base Interface: EvidenceSource
# ---------------------------------------------------------------------------

class EvidenceSource(ABC):
    """Abstract interface for modular incident evidence collectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the evidence source."""
        pass

    @abstractmethod
    def collect(self, params: IncidentQueryParams) -> SourceResult:
        """Collect and analyze evidence matching the dynamic runtime query."""
        pass


# ---------------------------------------------------------------------------
# Concrete Source 1: LogSource (Existing structured & file logs)
# ---------------------------------------------------------------------------

class LogSource(EvidenceSource):
    """Gathers evidence from existing application logs and in-memory log buffer."""

    @property
    def name(self) -> str:
        return "LogSource"

    def _parse_time_cutoff(self, time_range: Optional[str]) -> Optional[datetime.datetime]:
        """Parse human or ISO time range into a UTC cutoff datetime."""
        if not time_range:
            return None
        t_str = time_range.strip().lower()
        now = datetime.datetime.now(datetime.timezone.utc)

        # Match relative patterns e.g. "15m", "1h", "24h", "last 30m"
        match = re.search(r"(\d+)\s*(s|sec|m|min|minute|minutes|h|hr|hour|hours|d|day|days)", t_str)
        if match:
            val = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("s"):
                return now - datetime.timedelta(seconds=val)
            if unit.startswith("m"):
                return now - datetime.timedelta(minutes=val)
            if unit.startswith("h"):
                return now - datetime.timedelta(hours=val)
            if unit.startswith("d"):
                return now - datetime.timedelta(days=val)

        # Try ISO format
        try:
            cleaned = time_range.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except Exception:
            return None

    def _load_log_entries(self) -> list[LogRecordEntry]:
        """Combine in-memory log entries with existing log file entries."""
        entries: list[LogRecordEntry] = []

        # 1. From in-memory ring buffer
        buf = get_log_buffer_handler()
        entries.extend(list(buf.buffer))

        # 2. From existing log files if present
        log_dir = PROJECT_ROOT / "logs"
        candidate_files = [
            log_dir / "backend.err.log",
            log_dir / "backend.out.log",
            PROJECT_ROOT / "backend.err.log",
            PROJECT_ROOT / "backend.out.log",
        ]

        seen_messages: set[str] = {f"{e.timestamp}:{e.message}" for e in entries}

        for path in candidate_files:
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        # Try parsing structured JSON log line
                        if line_str.startswith("{") and line_str.endswith("}"):
                            try:
                                data = json.loads(line_str)
                                ts = data.get("timestamp", "")
                                msg = data.get("message", "")
                                key = f"{ts}:{msg}"
                                if key not in seen_messages:
                                    seen_messages.add(key)
                                    entries.append(
                                        LogRecordEntry(
                                            timestamp=ts,
                                            level=data.get("level", "INFO"),
                                            logger_name=data.get("logger", "backend"),
                                            message=msg,
                                            request_id=data.get("request_id"),
                                            session_id=data.get("session_id"),
                                            user_id=data.get("user_id"),
                                            endpoint=data.get("endpoint"),
                                            status=data.get("status"),
                                            duration=data.get("duration"),
                                            error_type=data.get("error_type"),
                                            raw_text=line_str,
                                        )
                                    )
                                continue
                            except Exception:
                                pass

                        # Fallback for plain log lines (e.g. Uvicorn INFO / ERROR)
                        level = "INFO"
                        if "ERROR" in line_str or "Traceback" in line_str or "Exception" in line_str:
                            level = "ERROR"
                        elif "WARNING" in line_str or "WARN" in line_str:
                            level = "WARNING"

                        # Extract basic timestamp if available
                        ts_match = re.match(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line_str)
                        ts = ts_match.group(1) if ts_match else ""

                        key = f"{ts}:{line_str[:80]}"
                        if key not in seen_messages:
                            seen_messages.add(key)
                            entries.append(
                                LogRecordEntry(
                                    timestamp=ts,
                                    level=level,
                                    logger_name="system",
                                    message=line_str,
                                    raw_text=line_str,
                                )
                            )
            except Exception as exc:
                logger.debug("Could not read log file %s: %s", path, exc)

        return entries

    def collect(self, params: IncidentQueryParams) -> SourceResult:
        all_entries = self._load_log_entries()
        cutoff_dt = self._parse_time_cutoff(params.time_range)

        matched_entries: list[LogRecordEntry] = []
        error_counts_by_type: collections.Counter = collections.Counter()
        matching_levels: collections.Counter = collections.Counter()

        # Build search patterns dynamically from user input
        error_pat = re.compile(re.escape(params.error_message), re.IGNORECASE) if params.error_message else None
        service_pat = re.compile(re.escape(params.service_name), re.IGNORECASE) if params.service_name else None
        endpoint_pat = re.compile(re.escape(params.endpoint), re.IGNORECASE) if params.endpoint else None

        for entry in all_entries:
            # Check time range
            if cutoff_dt and entry.timestamp:
                try:
                    entry_dt = datetime.datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=datetime.timezone.utc)
                    if entry_dt < cutoff_dt:
                        continue
                except Exception:
                    pass

            # Filter criteria match
            is_match = False

            # If specific search parameters were provided
            if params.request_id and entry.request_id == params.request_id:
                is_match = True
            elif error_pat and (error_pat.search(entry.message) or (entry.error_type and error_pat.search(entry.error_type))):
                is_match = True
            elif service_pat and (service_pat.search(entry.logger_name) or service_pat.search(entry.message)):
                is_match = True
            elif endpoint_pat and entry.endpoint and endpoint_pat.search(entry.endpoint):
                is_match = True
            elif not params.request_id and not error_pat and not service_pat and not endpoint_pat:
                # If no specific query filter was provided, look for error / warning logs
                if entry.level in ("ERROR", "CRITICAL", "WARNING"):
                    is_match = True
                elif params.severity and params.severity.upper() == entry.level:
                    is_match = True

            if is_match:
                matched_entries.append(entry)
                matching_levels[entry.level] += 1
                if entry.error_type:
                    error_counts_by_type[entry.error_type] += 1
                elif entry.level in ("ERROR", "CRITICAL"):
                    # Extract general exception name if present in message
                    exc_match = re.search(r"\b([A-Za-z]+Error|[A-Za-z]+Exception)\b", entry.message)
                    if exc_match:
                        error_counts_by_type[exc_match.group(1)] += 1

        observed_facts: list[str] = []
        evidence_items: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []

        if not all_entries:
            return SourceResult(
                source_name=self.name,
                available=False,
                status_message="Evidence unavailable: No application log records or log files found.",
                observed_facts=["No log records found in log buffer or log files."],
                limitations=["LogSource: Log files not present or empty."],
            )

        observed_facts.append(f"Scanned {len(all_entries)} total application log entries.")
        observed_facts.append(f"Identified {len(matched_entries)} log entries relevant to the incident query.")

        for level, count in matching_levels.items():
            observed_facts.append(f"Found {count} log record(s) at {level} level.")

        for err_type, count in error_counts_by_type.items():
            observed_facts.append(f"Encountered error type '{err_type}' {count} time(s).")

        # Extract timeline and sample evidence items
        for entry in matched_entries[-20:]:  # Keep recent relevant entries
            ts = entry.timestamp or "timestamp_unavailable"
            timeline_events.append({
                "timestamp": ts,
                "event": f"[{entry.level}] {entry.logger_name}: {entry.message[:200]}",
                "source": "application_log",
                "request_id": entry.request_id,
                "error_type": entry.error_type,
            })
            evidence_items.append({
                "source": "LogSource",
                "type": "log_record",
                "timestamp": ts,
                "level": entry.level,
                "logger": entry.logger_name,
                "message": entry.message[:300],
                "endpoint": entry.endpoint,
                "status": entry.status,
                "error_type": entry.error_type,
            })

        return SourceResult(
            source_name=self.name,
            available=True,
            status_message=f"Log analysis completed: {len(matched_entries)} matching entries found.",
            observed_facts=observed_facts,
            evidence_items=evidence_items,
            timeline_events=timeline_events,
        )


# ---------------------------------------------------------------------------
# Concrete Source 2: MetricSource (Existing in-memory & Prometheus metrics)
# ---------------------------------------------------------------------------

class MetricSource(EvidenceSource):
    """Gathers evidence from existing Metrics collector and live system probes."""

    @property
    def name(self) -> str:
        return "MetricSource"

    def collect(self, params: IncidentQueryParams) -> SourceResult:
        metrics = get_metrics()
        observed_facts: list[str] = []
        evidence_items: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []
        raw_metrics: dict[str, Any] = {}

        # 1. Inspect request counts and latencies
        request_counts = dict(metrics._request_count)
        request_latencies = dict(metrics._request_latency)
        total_requests = sum(request_counts.values())

        raw_metrics["total_requests"] = total_requests
        raw_metrics["request_counts_by_endpoint_status"] = request_counts

        observed_facts.append(f"Total API requests recorded in metrics collector: {total_requests}.")

        high_latency_endpoints = []
        endpoint_errors = []

        for key, count in request_counts.items():
            endpoint, status_str = key.rsplit(":", 1) if ":" in key else (key, "200")
            status = int(status_str) if status_str.isdigit() else 200

            if status >= 400:
                endpoint_errors.append((endpoint, status, count))

            durations = request_latencies.get(key, [])
            if durations:
                p50 = metrics._percentile(durations, 50)
                p95 = metrics._percentile(durations, 95)
                p99 = metrics._percentile(durations, 99)
                raw_metrics[f"latency_{key}"] = {"p50": p50, "p95": p95, "p99": p99, "samples": len(durations)}
                if p95 > 2.0:
                    high_latency_endpoints.append((endpoint, status, p95))

        for ep, st, cnt in endpoint_errors:
            observed_facts.append(f"HTTP {st} errors observed on endpoint '{ep}': {cnt} occurrence(s).")
            evidence_items.append({
                "source": "MetricSource",
                "type": "http_error_rate",
                "endpoint": ep,
                "status_code": st,
                "count": cnt,
            })

        for ep, st, p95 in high_latency_endpoints:
            observed_facts.append(f"Elevated p95 latency on '{ep}' ({st}): {p95:.2f}s.")
            evidence_items.append({
                "source": "MetricSource",
                "type": "elevated_latency",
                "endpoint": ep,
                "p95_seconds": p95,
            })

        # 2. Inspect LLM latencies
        llm_latencies = dict(metrics._llm_latency)
        for model_name, durations in llm_latencies.items():
            if durations:
                p50 = metrics._percentile(durations, 50)
                p95 = metrics._percentile(durations, 95)
                observed_facts.append(f"LLM model '{model_name}' latency: p50={p50:.2f}s, p95={p95:.2f}s across {len(durations)} calls.")
                evidence_items.append({
                    "source": "MetricSource",
                    "type": "llm_latency",
                    "model": model_name,
                    "p50_seconds": p50,
                    "p95_seconds": p95,
                    "sample_count": len(durations),
                })

        # 3. Inspect RAG latencies
        rag_latencies = dict(metrics._rag_latency)
        for key, durations in rag_latencies.items():
            if durations:
                p95 = metrics._percentile(durations, 95)
                observed_facts.append(f"RAG vector search p95 latency: {p95:.2f}s across {len(durations)} queries.")

        # 4. Inspect tool latencies and failures
        tool_failures = dict(metrics._tool_failures)
        tool_latencies = dict(metrics._tool_latency)
        raw_metrics["tool_failures"] = tool_failures

        for tool_name, failures in tool_failures.items():
            if failures > 0:
                observed_facts.append(f"Tool '{tool_name}' recorded {failures} execution failure(s).")
                evidence_items.append({
                    "source": "MetricSource",
                    "type": "tool_failure",
                    "tool": tool_name,
                    "failure_count": failures,
                })

        for key, durations in tool_latencies.items():
            tool_name, success_str = key.rsplit(":", 1) if ":" in key else (key, "true")
            if durations:
                p95 = metrics._percentile(durations, 95)
                if p95 > 5.0:
                    observed_facts.append(f"Tool '{tool_name}' (success={success_str}) p95 duration elevated at {p95:.2f}s.")

        # 5. Rate limit events
        rate_limits = metrics._rate_limit_events
        raw_metrics["rate_limit_events"] = rate_limits
        if rate_limits > 0:
            observed_facts.append(f"Recorded {rate_limits} rate-limit rejection event(s).")
            evidence_items.append({
                "source": "MetricSource",
                "type": "rate_limiting",
                "events_total": rate_limits,
            })

        # 6. Global error counts
        error_counts = dict(metrics._error_count)
        raw_metrics["error_counts_by_type"] = error_counts
        for err_type, count in error_counts.items():
            observed_facts.append(f"Metric error counter '{err_type}': {count} event(s).")

        # 7. Check Redis health if configured
        redis_url = get_settings().redis_url
        if redis_url:
            try:
                import redis
                client = redis.from_url(redis_url, socket_connect_timeout=1, socket_timeout=1)
                is_alive = bool(client.ping())
                observed_facts.append(f"Redis health probe: reachable ({redis_url}).")
            except Exception as exc:
                observed_facts.append(f"Redis health probe failed: {exc} ({redis_url}).")
                evidence_items.append({
                    "source": "MetricSource",
                    "type": "redis_probe_failure",
                    "detail": str(exc),
                })

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        timeline_events.append({
            "timestamp": now_iso,
            "event": f"Metrics snapshot evaluated across {total_requests} requests.",
            "source": "metrics_collector",
        })

        return SourceResult(
            source_name=self.name,
            available=True,
            status_message="Application metrics evaluated successfully.",
            observed_facts=observed_facts,
            evidence_items=evidence_items,
            timeline_events=timeline_events,
            raw_metrics=raw_metrics,
        )


# ---------------------------------------------------------------------------
# Interface Stubs for Future Integration (Return 'Evidence unavailable')
# ---------------------------------------------------------------------------

class TraceSource(EvidenceSource):
    """Distributed tracing source (e.g. OpenTelemetry / Jaeger). Ready for future integration."""

    @property
    def name(self) -> str:
        return "TraceSource"

    def collect(self, params: IncidentQueryParams) -> SourceResult:
        return SourceResult(
            source_name=self.name,
            available=False,
            status_message="Evidence unavailable.",
            observed_facts=[],
            evidence_items=[],
            timeline_events=[],
            limitations=["TraceSource: Distributed tracing telemetry not configured in V1."],
        )


class DeploymentSource(EvidenceSource):
    """CI/CD deployment source (e.g. GitHub Releases, ArgoCD). Ready for future integration."""

    @property
    def name(self) -> str:
        return "DeploymentSource"

    def collect(self, params: IncidentQueryParams) -> SourceResult:
        return SourceResult(
            source_name=self.name,
            available=False,
            status_message="Evidence unavailable.",
            observed_facts=[],
            evidence_items=[],
            timeline_events=[],
            limitations=["DeploymentSource: Deployment event history not configured in V1."],
        )


class GitSource(EvidenceSource):
    """Version control source (e.g. Git commit log, repository diffs). Ready for future integration."""

    @property
    def name(self) -> str:
        return "GitSource"

    def collect(self, params: IncidentQueryParams) -> SourceResult:
        return SourceResult(
            source_name=self.name,
            available=False,
            status_message="Evidence unavailable.",
            observed_facts=[],
            evidence_items=[],
            timeline_events=[],
            limitations=["GitSource: Git commit telemetry not configured in V1."],
        )


class DatabaseHealthSource(EvidenceSource):
    """External database health source (e.g. connection pool stats). Ready for future integration."""

    @property
    def name(self) -> str:
        return "DatabaseHealthSource"

    def collect(self, params: IncidentQueryParams) -> SourceResult:
        return SourceResult(
            source_name=self.name,
            available=False,
            status_message="Evidence unavailable.",
            observed_facts=[],
            evidence_items=[],
            timeline_events=[],
            limitations=["DatabaseHealthSource: External database health monitoring not configured in V1."],
        )
