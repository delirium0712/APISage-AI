"""
Test suite for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from api.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def sample_openapi_spec():
    """Sample OpenAPI specification for testing"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "description": "A test API specification",
        },
        "paths": {
            "/users": {
                "get": {
                    "summary": "Get users",
                    "responses": {"200": {"description": "Success"}},
                }
            }
        },
    }


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "llm_available" in data


# NOTE: API key management tests removed - functionality moved to .env configuration


@pytest.mark.asyncio
async def test_analyze_api_no_llm(client, sample_openapi_spec):
    """Test analysis endpoint when LLM is not available"""
    with patch("api.main.llm_manager.is_available", return_value=False):
        response = client.post(
            "/analyze",
            json={
                "openapi_spec": sample_openapi_spec,
                "analysis_depth": "comprehensive",
            },
        )
        assert response.status_code == 503


@pytest.mark.asyncio
async def test_rag_query_empty_question(client):
    """Test RAG query with empty question"""
    response = client.post("/rag-query", json={"question": ""})
    assert response.status_code == 400


def test_create_analysis_prompt():
    """Test analysis prompt creation"""
    from api.main import create_analysis_prompt

    spec = {
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {"/test": {"get": {}}},
    }

    prompt = create_analysis_prompt(spec, "comprehensive", ["security"])
    assert "Test API" in prompt
    assert "/test" in prompt
    assert "security" in prompt.lower()


def test_parse_llm_response():
    """Test LLM response parsing"""
    from api.main import parse_llm_response

    response = """
## Critical Issues
This is critical.

## Security Analysis
No issues found.
"""

    result = parse_llm_response(response)
    assert "sections" in result
    assert "key_findings" in result
    assert "critical_issues" in result["sections"]
    assert "security_analysis" in result["sections"]
