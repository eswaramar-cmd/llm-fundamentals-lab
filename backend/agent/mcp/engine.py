"""Production Incident Investigator Engine.

Correlates telemetry from modular evidence sources, separates observed facts from
inferences, ranks root-cause hypotheses strictly on evidence, and generates safe,
read-only remediation recommendations.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from backend.agent.mcp.sources import (
    DatabaseHealthSource,
    DeploymentSource,
    EvidenceSource,
    GitSource,
    IncidentQueryParams,
    LogSource,
    MetricSource,
    SourceResult,
    TraceSource,
)

logger = logging.getLogger(__name__)


class IncidentInvestigator:
    """Core analytical engine for investigating production incidents."""

    def __init__(self, sources: Optional[list[EvidenceSource]] = None) -> None:
        self.sources: list[EvidenceSource] = sources or [
            LogSource(),
            MetricSource(),
            TraceSource(),
            DeploymentSource(),
            GitSource(),
            DatabaseHealthSource(),
        ]

    def _matches_query_scope(self, text: str, params: IncidentQueryParams) -> bool:
        """Check whether a telemetry string matches the user-provided query scope."""
        if not text:
            return False
        t_low = text.lower()
        if params.service_name and params.service_name.lower() in t_low:
            return True
        if params.endpoint and params.endpoint.lower() in t_low:
            return True
        if params.error_message and params.error_message.lower() in t_low:
            return True
        if params.request_id and params.request_id.lower() in t_low:
            return True
        return False

    def investigate(self, params: IncidentQueryParams) -> dict[str, Any]:
        """Perform dynamic incident investigation based on user runtime inputs."""

        # 1. Collect evidence from all modular sources
        source_results: list[SourceResult] = []
        for src in self.sources:
            try:
                res = src.collect(params)
                source_results.append(res)
            except Exception as exc:
                logger.warning("Source %s failed collection: %s", src.name, exc)
                source_results.append(
                    SourceResult(
                        source_name=src.name,
                        available=False,
                        status_message=f"Collection error: {exc}",
                        limitations=[f"{src.name}: Failed to gather evidence due to internal error ({exc})."],
                    )
                )

        # 2. Aggregate observed facts, raw evidence, and limitations
        observed_facts: list[str] = []
        evidence_items: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        limitations: list[str] = []

        for sr in source_results:
            observed_facts.extend(sr.observed_facts)
            evidence_items.extend(sr.evidence_items)
            timeline.extend(sr.timeline_events)
            limitations.extend(sr.limitations)
            if not sr.available and sr.status_message:
                evidence_items.append({
                    "source": sr.source_name,
                    "status": "unavailable",
                    "finding": sr.status_message,
                })

        # Sort timeline by timestamp where possible
        def _sort_key(item: dict[str, Any]) -> str:
            return str(item.get("timestamp", ""))

        timeline = sorted(timeline, key=_sort_key)

        # 3. Formulate dynamic incident summary
        components = []
        if params.service_name:
            components.append(f"service '{params.service_name}'")
        if params.endpoint:
            components.append(f"endpoint '{params.endpoint}'")
        if params.error_message:
            components.append(f"error '{params.error_message}'")
        if params.severity:
            components.append(f"severity [{params.severity.upper()}]")
        if params.time_range:
            components.append(f"over time window '{params.time_range}'")
        if params.details:
            components.append(f"({params.details})")

        scope_desc = ", ".join(components) if components else "general production telemetry"
        incident_summary = f"Investigation for {scope_desc} across {len(self.sources)} modular evidence sources."

        # 4. Filter evidence based on query specificity
        has_specific_filter = bool(params.service_name or params.error_message or params.endpoint or params.request_id)

        # Separate evidence categories
        all_log_errors = [ev for ev in evidence_items if ev.get("source") == "LogSource" and ev.get("level") in ("ERROR", "CRITICAL")]
        all_elevated_latencies = [ev for ev in evidence_items if ev.get("type") == "elevated_latency" or ev.get("type") == "llm_latency"]
        all_tool_failures = [ev for ev in evidence_items if ev.get("type") == "tool_failure"]
        all_rate_limits = [ev for ev in evidence_items if ev.get("type") == "rate_limiting"]
        all_redis_failures = [ev for ev in evidence_items if ev.get("type") == "redis_probe_failure"]
        all_http_errors = [ev for ev in evidence_items if ev.get("type") == "http_error_rate"]

        # If specific query provided, scope the evidence
        if has_specific_filter:
            log_errors = [
                ev for ev in all_log_errors
                if self._matches_query_scope(f"{ev.get('logger')} {ev.get('message')} {ev.get('endpoint')} {ev.get('error_type')}", params)
            ]
            tool_failures = [
                ev for ev in all_tool_failures
                if self._matches_query_scope(str(ev.get("tool")), params)
            ]
            http_errors = [
                ev for ev in all_http_errors
                if self._matches_query_scope(str(ev.get("endpoint")), params)
            ]
            elevated_latencies = [
                ev for ev in all_elevated_latencies
                if self._matches_query_scope(f"{ev.get('endpoint')} {ev.get('model')}", params)
            ]
            redis_failures = all_redis_failures if (params.service_name and "redis" in params.service_name.lower()) else []
            rate_limits = all_rate_limits if (params.error_message and "rate" in params.error_message.lower()) else []
        else:
            log_errors = all_log_errors
            tool_failures = all_tool_failures
            http_errors = all_http_errors
            elevated_latencies = all_elevated_latencies
            redis_failures = all_redis_failures
            rate_limits = all_rate_limits

        candidates: list[dict[str, Any]] = []
        recommended_actions: list[str] = []
        evidence_score = 0.0

        if tool_failures:
            tools_affected = [tf.get("tool") for tf in tool_failures if tf.get("tool")]
            candidates.append({
                "candidate": f"Tool execution failure in component: {', '.join(tools_affected)}",
                "likelihood": "high",
                "score": 0.85,
                "reasoning": f"Metrics and logs explicitly recorded {len(tool_failures)} failed tool execution(s).",
                "supporting_evidence": [f"Tool '{tf.get('tool')}' failures count: {tf.get('failure_count')}" for tf in tool_failures],
            })
            for t in tools_affected:
                recommended_actions.append(f"Check configuration, network connectivity, and credentials for tool '{t}'.")
            evidence_score += 0.4

        if redis_failures:
            candidates.append({
                "candidate": "Redis state/cache connectivity failure",
                "likelihood": "high",
                "score": 0.80,
                "reasoning": "Live probe or log analysis indicated connection timeout / rejection reaching Redis.",
                "supporting_evidence": [str(rf.get("detail")) for rf in redis_failures],
            })
            recommended_actions.append("Inspect Redis server health, network partition, or REDIS_URL configuration.")
            evidence_score += 0.35

        if rate_limits:
            total_rl = sum(rl.get("events_total", 1) for rl in rate_limits)
            candidates.append({
                "candidate": "Client rate-limit threshold exceeded",
                "likelihood": "high",
                "score": 0.75,
                "reasoning": f"Metrics recorded {total_rl} rate-limiting event(s) rejecting incoming traffic.",
                "supporting_evidence": [f"Rate-limit rejections observed: {total_rl}"],
            })
            recommended_actions.append("Review client traffic patterns and adjust RATE_LIMIT_REQUESTS / window settings if legitimate.")
            evidence_score += 0.3

        if log_errors:
            top_err = log_errors[0].get("message", "Unknown error in log stream")
            candidates.append({
                "candidate": f"Application error: {log_errors[0].get('error_type') or 'Logged Exception'}",
                "likelihood": "high" if len(log_errors) > 2 else "medium",
                "score": 0.70,
                "reasoning": f"Identified {len(log_errors)} error-level log entries matching incident scope.",
                "supporting_evidence": [f"[{e.get('level')}] {e.get('message')}" for e in log_errors[:3]],
            })
            recommended_actions.append("Examine error tracebacks in logs and address unhandled exceptions.")
            evidence_score += 0.35

        if elevated_latencies:
            candidates.append({
                "candidate": "Upstream / LLM inference or vector search latency bottleneck",
                "likelihood": "medium",
                "score": 0.65,
                "reasoning": "Percentile metrics show elevated p95 latency on endpoints or LLM/RAG pipelines.",
                "supporting_evidence": [f"Latency alert: {l.get('endpoint') or l.get('model')} p95={l.get('p95_seconds', 0):.2f}s" for l in elevated_latencies[:3]],
            })
            recommended_actions.append("Review LLM provider response times, prompt token lengths, and concurrency settings.")
            evidence_score += 0.25

        if http_errors:
            candidates.append({
                "candidate": "HTTP 4xx/5xx API endpoint failure",
                "likelihood": "medium",
                "score": 0.60,
                "reasoning": "Recorded non-200 HTTP responses in endpoint metrics.",
                "supporting_evidence": [f"Endpoint {he.get('endpoint')} returned status {he.get('status_code')} ({he.get('count')} times)" for he in http_errors[:3]],
            })
            recommended_actions.append("Check API route parameter validation and server-side route handlers.")
            evidence_score += 0.2

        # If no concrete matching evidence was found
        if not candidates:
            candidates.append({
                "candidate": "Inconclusive / Telemetry shows nominal operations",
                "likelihood": "low",
                "score": 0.0,
                "reasoning": "No matching error logs, tool failures, latency spikes, or metric anomalies were found in available evidence sources for the specified query.",
                "supporting_evidence": ["No correlating error records in LogSource or MetricSource."],
            })
            root_cause = "Evidence unavailable: No failure signatures or matching log/metric anomalies were detected in the queried scope."
            confidence = 0.0
            recommended_actions.append("Verify if the incident occurred within the queried time range or check if external logging is enabled.")
        else:
            # Sort candidates by score descending
            candidates.sort(key=lambda c: c["score"], reverse=True)
            top = candidates[0]
            root_cause = f"{top['candidate']}: {top['reasoning']}"
            confidence = min(round(min(top["score"], 0.5 + (evidence_score * 0.5)), 2), 0.95)

        # Deduplicate recommended actions
        seen_actions = set()
        deduped_actions = []
        for a in recommended_actions:
            if a not in seen_actions:
                seen_actions.add(a)
                deduped_actions.append(a)

        # Build final structured response
        return {
            "incident_summary": incident_summary,
            "root_cause": root_cause,
            "confidence": confidence,
            "observed_facts": observed_facts,
            "evidence": evidence_items,
            "timeline": timeline,
            "root_cause_candidates": candidates,
            "recommended_actions": deduped_actions,
            "limitations": limitations,
        }
