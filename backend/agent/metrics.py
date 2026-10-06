"""Lightweight production metrics.

No external monitoring stack is required.  Exposes Prometheus-format
metrics on ``GET /metrics`` and maintains in-process counters that can
be scraped.

Tracks:
  - request count / latency (per endpoint, per status)
  - LLM latency
  - RAG latency
  - tool latency / failures
  - rate-limit events
  - error count
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class Metrics:
    """In-process metrics collector with Prometheus export."""

    def __init__(self) -> None:
        self._request_count: dict[str, int] = defaultdict(int)
        self._request_latency: dict[str, list[float]] = defaultdict(list)
        self._llm_latency: dict[str, list[float]] = defaultdict(list)
        self._rag_latency: dict[str, list[float]] = defaultdict(list)
        self._tool_latency: dict[str, list[float]] = defaultdict(list)
        self._tool_failures: dict[str, int] = defaultdict(int)
        self._rate_limit_events: int = 0
        self._error_count: dict[str, int] = defaultdict(int)
        self._max_samples = 500

    # -- Recording ----------------------------------------------------------

    def record_request(self, endpoint: str, status_code: int, duration: float) -> None:
        key = f"{endpoint}:{status_code}"
        self._request_count[key] += 1
        self._request_latency[key].append(duration)
        self._trim(self._request_latency[key])

    def record_llm(self, model: str, duration: float) -> None:
        self._llm_latency[model].append(duration)
        self._trim(self._llm_latency[model])

    def record_rag(self, duration: float) -> None:
        self._rag_latency["default"].append(duration)
        self._trim(self._rag_latency["default"])

    def record_tool(self, tool_name: str, duration: float, success: bool) -> None:
        key = f"{tool_name}:{success}"
        self._tool_latency[key].append(duration)
        self._trim(self._tool_latency[key])
        if not success:
            self._tool_failures[tool_name] += 1

    def record_rate_limit(self) -> None:
        self._rate_limit_events += 1

    def record_error(self, error_type: str) -> None:
        self._error_count[error_type] += 1

    # -- Helpers ------------------------------------------------------------

    def _trim(self, lst: list) -> None:
        if len(lst) > self._max_samples:
            del lst[: len(lst) - self._max_samples]

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
        return s[idx]

    # -- Export -------------------------------------------------------------

    def to_prometheus(self) -> str:
        lines: list[str] = []

        for key, count in sorted(self._request_count.items()):
            endpoint, status = key.rsplit(":", 1)
            lines.append(
                f'# TYPE ai_requests_total counter\n'
                f'ai_requests_total{{endpoint="{endpoint}", status="{status}"}} {count}\n'
            )

        for key, durations in sorted(self._request_latency.items()):
            endpoint, status = key.rsplit(":", 1)
            if durations:
                lines.append(
                    f'ai_request_duration_seconds{{endpoint="{endpoint}", status="{status}", '
                    f'p50="{self._percentile(durations, 50):.4f}", '
                    f'p95="{self._percentile(durations, 95):.4f}", '
                    f'p99="{self._percentile(durations, 99):.4f}"}} {durations[-1]:.4f}\n'
                )

        for model, durations in sorted(self._llm_latency.items()):
            if durations:
                lines.append(
                    f'# TYPE ai_llm_latency_seconds histogram\n'
                    f'ai_llm_latency_seconds{{model="{model}", '
                    f'p50="{self._percentile(durations, 50):.4f}", '
                    f'p95="{self._percentile(durations, 95):.4f}", '
                    f'p99="{self._percentile(durations, 99):.4f}"}} {durations[-1]:.4f}\n'
                )

        for key, durations in sorted(self._rag_latency.items()):
            if durations:
                lines.append(
                    f'# TYPE ai_rag_latency_seconds histogram\n'
                    f'ai_rag_latency_seconds{{p50="{self._percentile(durations, 50):.4f}", '
                    f'p95="{self._percentile(durations, 95):.4f}", '
                    f'p99="{self._percentile(durations, 99):.4f}"}} {durations[-1]:.4f}\n'
                )

        for key, durations in sorted(self._tool_latency.items()):
            tool_name, success = key.rsplit(":", 1)
            if durations:
                lines.append(
                    f'# TYPE ai_tool_duration_seconds histogram\n'
                    f'ai_tool_duration_seconds{{tool="{tool_name}", success="{success}", '
                    f'p50="{self._percentile(durations, 50):.4f}", '
                    f'p95="{self._percentile(durations, 95):.4f}", '
                    f'p99="{self._percentile(durations, 99):.4f}"}} {durations[-1]:.4f}\n'
                )

        for tool_name, count in sorted(self._tool_failures.items()):
            lines.append(
                f'# TYPE ai_tool_failures_total counter\n'
                f'ai_tool_failures_total{{tool="{tool_name}"}} {count}\n'
            )

        lines.append(
            f'# TYPE ai_rate_limit_events_total counter\n'
            f'ai_rate_limit_events_total {self._rate_limit_events}\n'
        )

        for error_type, count in sorted(self._error_count.items()):
            lines.append(
                f'# TYPE ai_errors_total counter\n'
                f'ai_errors_total{{error_type="{error_type}"}} {count}\n'
            )

        return "".join(lines)


# Singleton
_metrics: Optional[Metrics] = None


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics
