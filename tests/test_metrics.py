"""Tests for metrics collection."""

from __future__ import annotations

from backend.agent.metrics import Metrics, get_metrics


class TestMetrics:
    def test_request_count_and_latency(self):
        m = Metrics()
        m.record_request("/agent/run", 200, 1.5)
        m.record_request("/agent/run", 200, 2.0)
        m.record_request("/agent/run", 429, 0.1)
        m.record_request("/agent/run", 500, 3.0)

        prometheus = m.to_prometheus()
        assert "ai_requests_total{endpoint=\"/agent/run\", status=\"200\"}" in prometheus
        assert "ai_requests_total{endpoint=\"/agent/run\", status=\"429\"}" in prometheus
        assert "ai_requests_total{endpoint=\"/agent/run\", status=\"500\"}" in prometheus

    def test_llm_latency(self):
        m = Metrics()
        m.record_llm("llama3.2:3b", 5.0)
        m.record_llm("llama3.2:3b", 10.0)
        prometheus = m.to_prometheus()
        assert "ai_llm_latency_seconds{model=\"llama3.2:3b\"" in prometheus

    def test_rag_latency(self):
        m = Metrics()
        m.record_rag(0.5)
        prometheus = m.to_prometheus()
        assert "ai_rag_latency_seconds" in prometheus

    def test_tool_latency_and_failures(self):
        m = Metrics()
        m.record_tool("calculator", 0.1, True)
        m.record_tool("web_search", 15.0, False)
        prometheus = m.to_prometheus()
        assert "ai_tool_duration_seconds{tool=\"calculator\", success=\"True\"" in prometheus
        assert "ai_tool_duration_seconds{tool=\"web_search\", success=\"False\"" in prometheus
        assert "ai_tool_failures_total{tool=\"web_search\"}" in prometheus

    def test_rate_limit_events(self):
        m = Metrics()
        m.record_rate_limit()
        m.record_rate_limit()
        prometheus = m.to_prometheus()
        assert "ai_rate_limit_events_total 2" in prometheus

    def test_error_count(self):
        m = Metrics()
        m.record_error("tool_timeout")
        m.record_error("validation_error")
        prometheus = m.to_prometheus()
        assert "ai_errors_total{error_type=\"tool_timeout\"}" in prometheus
        assert "ai_errors_total{error_type=\"validation_error\"}" in prometheus

    def test_percentiles_calculated(self):
        m = Metrics()
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            m.record_request("/test", 200, v)
        prometheus = m.to_prometheus()
        assert "p50=" in prometheus
        assert "p95=" in prometheus
        assert "p99=" in prometheus

    def test_singleton(self):
        """get_metrics should return the same instance."""
        a = get_metrics()
        b = get_metrics()
        assert a is b
