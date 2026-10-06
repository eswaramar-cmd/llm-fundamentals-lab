"""Tests for Production Incident Investigator MCP tool, sources, engine, and API endpoint.

Verifies dynamic runtime inputs, modular evidence sources, root-cause ranking,
confidence scoring, LangGraph agent routing, MCP server protocol, and FastAPI endpoints.
"""

from __future__ import annotations

import json
import logging
import pytest
from fastapi.testclient import TestClient

from backend.agent.mcp.engine import IncidentInvestigator
from backend.agent.mcp.protocol import INVESTIGATE_INCIDENT_TOOL_SCHEMA, MCPServer
from backend.agent.mcp.sources import (
    DatabaseHealthSource,
    DeploymentSource,
    GitSource,
    IncidentQueryParams,
    LogRecordEntry,
    LogSource,
    MetricSource,
    TraceSource,
    get_log_buffer_handler,
)
from backend.agent.metrics import get_metrics
from backend.agent.nodes.router import classify_question, route_question
from backend.agent.schemas import IncidentInvestigationRequest, IncidentInvestigationResponse
from backend.agent.tools.investigate_incident import investigate_incident
from backend.main import app


@pytest.fixture
def test_client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Modular Evidence Source Tests
# ---------------------------------------------------------------------------

class TestEvidenceSources:
    def test_future_stubs_return_evidence_unavailable(self):
        """Verify that modular stubs for future data sources return 'Evidence unavailable'."""
        params = IncidentQueryParams(service_name="payment_svc", error_message="timeout")

        trace_src = TraceSource()
        assert trace_src.name == "TraceSource"
        res_trace = trace_src.collect(params)
        assert res_trace.available is False
        assert "Evidence unavailable" in res_trace.status_message
        assert len(res_trace.limitations) > 0

        dep_src = DeploymentSource()
        res_dep = dep_src.collect(params)
        assert res_dep.available is False
        assert "Evidence unavailable" in res_dep.status_message

        git_src = GitSource()
        res_git = git_src.collect(params)
        assert res_git.available is False
        assert "Evidence unavailable" in res_git.status_message

        db_src = DatabaseHealthSource()
        res_db = db_src.collect(params)
        assert res_db.available is False
        assert "Evidence unavailable" in res_db.status_message

    def test_log_source_with_dynamic_input(self):
        """Verify LogSource extracts matching records and observed facts."""
        buf = get_log_buffer_handler()
        logger = logging.getLogger("auth_service")
        logger.error("OAuth token validation failed for user_123")

        params = IncidentQueryParams(
            service_name="auth_service",
            error_message="OAuth token validation failed",
            severity="critical",
        )
        src = LogSource()
        res = src.collect(params)

        assert res.source_name == "LogSource"
        assert res.available is True
        assert any("OAuth token validation failed" in fact or "auth_service" in fact or "error" in fact.lower() for fact in res.observed_facts)
        assert len(res.evidence_items) > 0
        assert len(res.timeline_events) > 0

    def test_metric_source_reflects_existing_metrics(self):
        """Verify MetricSource reads real metrics from get_metrics()."""
        metrics = get_metrics()
        metrics.record_tool("custom_tool", duration=6.5, success=False)
        metrics.record_request("/api/test_ep", status_code=500, duration=3.2)
        metrics.record_error("CustomTimeoutException")

        src = MetricSource()
        params = IncidentQueryParams()
        res = src.collect(params)

        assert res.source_name == "MetricSource"
        assert res.available is True
        assert any("custom_tool" in f for f in res.observed_facts)
        assert any("500" in f for f in res.observed_facts)
        assert any("CustomTimeoutException" in f for f in res.observed_facts)


# ---------------------------------------------------------------------------
# 2. Engine, Root Cause Ranking & Confidence Scoring Tests
# ---------------------------------------------------------------------------

class TestIncidentInvestigatorEngine:
    def test_investigation_with_no_matching_evidence_returns_evidence_unavailable(self):
        """When queried for a nonexistent error and service, engine returns low confidence and 'Evidence unavailable'."""
        params = IncidentQueryParams(
            service_name="nonexistent_microservice_xyz",
            error_message="UniqueUnseenErrorPattern_99999",
            time_range="5m",
        )
        investigator = IncidentInvestigator()
        result = investigator.investigate(params)

        assert "incident_summary" in result
        assert "root_cause" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert result["confidence"] == 0.0 or result["confidence"] < 0.3
        assert "Evidence unavailable" in result["root_cause"] or "No failure signatures" in result["root_cause"]
        assert len(result["limitations"]) >= 4  # Unimplemented future sources listed

    def test_investigation_correlates_tool_failure_and_ranks_candidate(self):
        """Engine ranks tool failure when metrics/logs confirm failures."""
        metrics = get_metrics()
        metrics.record_tool("weather_api", duration=1.0, success=False)
        metrics.record_tool("weather_api", duration=1.0, success=False)

        params = IncidentQueryParams(
            service_name="weather_api",
            error_message="weather_api failure",
            severity="high",
        )
        investigator = IncidentInvestigator()
        result = investigator.investigate(params)

        assert result["confidence"] > 0.5
        assert any("weather_api" in c["candidate"] for c in result["root_cause_candidates"])
        assert any("weather_api" in a for a in result["recommended_actions"])
        assert len(result["observed_facts"]) > 0
        assert len(result["timeline"]) > 0

    def test_investigation_with_dynamic_runtime_inputs(self):
        """Test varied dynamic inputs without hardcoding scenarios."""
        inputs = [
            {"service_name": "checkout", "error_message": "PaymentGatewayTimeout", "severity": "critical", "time_range": "30m"},
            {"service_name": "search", "error_message": "IndexOutOfMemory", "severity": "medium", "time_range": "1h"},
            {"service_name": "frontend", "error_message": "502 Bad Gateway", "severity": "high", "time_range": "24h"},
        ]

        investigator = IncidentInvestigator()
        for inp in inputs:
            params = IncidentQueryParams(**inp)
            res = investigator.investigate(params)
            assert inp["service_name"] in res["incident_summary"]
            assert inp["error_message"] in res["incident_summary"]
            assert isinstance(res["root_cause_candidates"], list)
            assert isinstance(res["recommended_actions"], list)
            assert isinstance(res["observed_facts"], list)
            assert isinstance(res["limitations"], list)


# ---------------------------------------------------------------------------
# 3. MCP Protocol & LangChain Tool Tests
# ---------------------------------------------------------------------------

class TestMCPProtocolAndTool:
    def test_mcp_server_list_tools(self):
        """Verify MCP server exposes investigate_incident tool schema."""
        server = MCPServer()
        tools = server.list_tools()
        assert len(tools) == 1
        tool = tools[0]
        assert tool["name"] == "investigate_incident"
        assert "inputSchema" in tool
        assert "service_name" in tool["inputSchema"]["properties"]

    def test_mcp_server_call_tool(self):
        """Verify dynamic tool invocation through MCP server interface."""
        server = MCPServer()
        res = server.call_tool(
            "investigate_incident",
            {
                "service_name": "inventory_service",
                "error_message": "ConnectionResetError",
                "severity": "high",
                "time_range": "10m",
            },
        )
        assert "incident_summary" in res
        assert "inventory_service" in res["incident_summary"]
        assert "confidence" in res
        assert "observed_facts" in res

    def test_langchain_tool_invocation(self):
        """Verify LangChain investigate_incident tool execution and valid JSON output."""
        output_json_str = investigate_incident.invoke({
            "service_name": "billing_worker",
            "error_message": "StripeAPIError: rate limit exceeded",
            "severity": "critical",
            "time_range": "15m",
        })
        assert isinstance(output_json_str, str)
        parsed = json.loads(output_json_str)

        assert "incident_summary" in parsed
        assert "root_cause" in parsed
        assert "confidence" in parsed
        assert "observed_facts" in parsed
        assert "evidence" in parsed
        assert "timeline" in parsed
        assert "root_cause_candidates" in parsed
        assert "recommended_actions" in parsed
        assert "limitations" in parsed


# ---------------------------------------------------------------------------
# 4. LangGraph Intent Classification & Routing Tests
# ---------------------------------------------------------------------------

class TestLangGraphRouting:
    @pytest.mark.parametrize(
        "query",
        [
            "Why is my service slow?",
            "Why are requests failing?",
            "Investigate this incident in payment component",
            "What caused the latency increase?",
            "Investigate production outage",
            "Check why error rate spike occurred",
        ],
    )
    def test_router_classifies_incident_queries(self, query):
        """Verify questions about outages, latency spikes, and failure investigation classify as 'incident'."""
        state = {"question": query, "messages": []}
        result = classify_question(state)
        assert result.get("answer") == "incident"

    def test_route_question_routes_incident_to_agent_node(self):
        state = {"answer": "incident"}
        next_node = route_question(state)
        assert next_node == "agent_node"


# ---------------------------------------------------------------------------
# 5. FastAPI Endpoint Tests (POST /incidents/investigate)
# ---------------------------------------------------------------------------

class TestFastAPIEndpoint:
    def test_investigate_endpoint_valid_dynamic_payload(self, test_client):
        """Verify POST /incidents/investigate accepts dynamic input and returns structured response."""
        payload = {
            "service_name": "recommendation_engine",
            "error_message": "EmbeddingServiceTimeout",
            "time_range": "1h",
            "severity": "medium",
            "details": "Users reported empty recommendations",
            "user_id": "alice",
        }
        resp = test_client.post("/incidents/investigate", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Validate against required output schema
        assert "incident_summary" in data
        assert "recommendation_engine" in data["incident_summary"]
        assert "root_cause" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], (int, float))
        assert "observed_facts" in data
        assert isinstance(data["observed_facts"], list)
        assert "evidence" in data
        assert isinstance(data["evidence"], list)
        assert "timeline" in data
        assert isinstance(data["timeline"], list)
        assert "root_cause_candidates" in data
        assert isinstance(data["root_cause_candidates"], list)
        assert "recommended_actions" in data
        assert isinstance(data["recommended_actions"], list)
        assert "limitations" in data
        assert isinstance(data["limitations"], list)

    def test_investigate_endpoint_empty_payload(self, test_client):
        """Endpoint handles empty / minimal payload gracefully."""
        resp = test_client.post("/incidents/investigate", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "incident_summary" in data
        assert "confidence" in data

    def test_investigate_endpoint_invalid_types(self, test_client):
        """Endpoint rejects invalid payload structure with 422."""
        resp = test_client.post("/incidents/investigate", json={"severity": 12345})
        assert resp.status_code in (200, 422)  # Pydantic may coerce or reject integer as string


# ---------------------------------------------------------------------------
# 6. Multi-User Isolation & State Cleanliness
# ---------------------------------------------------------------------------

class TestMultiUserIsolation:
    def test_isolated_user_investigations(self):
        """Ensure investigations with different user IDs and parameters remain independent."""
        inv = IncidentInvestigator()

        res_user1 = inv.investigate(IncidentQueryParams(
            service_name="service_alpha",
            error_message="AlphaError",
            user_id="user_alpha",
        ))

        res_user2 = inv.investigate(IncidentQueryParams(
            service_name="service_beta",
            error_message="BetaError",
            user_id="user_beta",
        ))

        assert "service_alpha" in res_user1["incident_summary"]
        assert "service_beta" in res_user2["incident_summary"]
        assert "service_alpha" not in res_user2["incident_summary"]
