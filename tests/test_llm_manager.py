"""
Test suite for LLM Manager
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import os

from apisage.core.llm_manager import (
    SimpleLLMManager,
    LLMRequest,
    LLMResponse,
    ModelConfig,
)


class TestModelConfig:
    """Test ModelConfig class"""

    def test_is_o1_model(self):
        """Test O1 model detection"""
        assert ModelConfig.is_o1_model("o1-mini") is True
        assert ModelConfig.is_o1_model("o1") is True
        assert ModelConfig.is_o1_model("gpt-4o") is False

    def test_get_model_defaults(self):
        """Test getting model defaults"""
        defaults = ModelConfig.get_model_defaults("o1-mini")
        assert defaults["temperature"] is None
        assert "max_tokens" in defaults

        defaults = ModelConfig.get_model_defaults("gpt-4o")
        assert defaults["temperature"] is not None


class TestLLMManager:
    """Test SimpleLLMManager class"""

    @pytest.fixture
    def mock_env_no_key(self):
        """Mock environment without API key"""
        with patch.dict(os.environ, {}, clear=True):
            yield

    @pytest.fixture
    def mock_env_with_key(self):
        """Mock environment with API key"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            yield

    def test_init_without_api_key(self, mock_env_no_key):
        """Test initialization without API key"""
        manager = SimpleLLMManager()
        assert manager.api_key is None
        assert not manager.is_available()

    def test_init_with_api_key(self, mock_env_with_key):
        """Test initialization with API key"""
        with patch("infrastructure.llm_manager.OPENAI_AVAILABLE", True):
            with patch("infrastructure.llm_manager.AsyncOpenAI") as mock_openai:
                manager = SimpleLLMManager()
                assert manager.api_key == "sk-test123"
                mock_openai.assert_called_once()

    def test_is_valid_model(self):
        """Test model validation"""
        manager = SimpleLLMManager()
        assert manager._is_valid_model("gpt-4o") is True
        assert manager._is_valid_model("invalid-model") is False

    def test_optimize_request_o1_model(self):
        """Test request optimization for O1 models"""
        manager = SimpleLLMManager()
        request = LLMRequest(prompt="test", model="o1-mini", temperature=0.5)

        optimized = manager.optimize_request(request)
        assert optimized.model == "o1-mini"
        assert optimized.temperature == 0.0  # Should be set to 0 for O1

    def test_optimize_request_regular_model(self):
        """Test request optimization for regular models"""
        manager = SimpleLLMManager()
        request = LLMRequest(prompt="test", model="gpt-4o", temperature=0.5)

        optimized = manager.optimize_request(request)
        assert optimized.model == "gpt-4o"
        assert optimized.temperature == 0.5  # Should preserve temperature

    @pytest.mark.asyncio
    async def test_generate_without_client(self):
        """Test generate method without client"""
        manager = SimpleLLMManager()
        request = LLMRequest(prompt="test", model="gpt-4o")

        response = await manager.generate(request)
        assert response.error is not None
        assert "not available" in response.content

    @pytest.mark.asyncio
    async def test_generate_response_wrapper(self):
        """Test generate_response convenience method"""
        manager = SimpleLLMManager()

        # Mock the generate method
        mock_response = LLMResponse(content="test response", model="gpt-4o")
        manager.generate = AsyncMock(return_value=mock_response)

        request = LLMRequest(prompt="test", model="gpt-4o")
        result = await manager.generate_response(request)

        assert result == "test response"

    def test_refresh_api_key_success(self, mock_env_with_key):
        """Test successful API key refresh"""
        with patch("infrastructure.llm_manager.OPENAI_AVAILABLE", True):
            with patch("infrastructure.llm_manager.AsyncOpenAI") as mock_openai:
                manager = SimpleLLMManager()
                result = manager.refresh_api_key()

                assert result is True
                assert manager.api_key == "sk-test123"

    def test_refresh_api_key_no_key(self, mock_env_no_key):
        """Test API key refresh without key"""
        manager = SimpleLLMManager()
        result = manager.refresh_api_key()

        assert result is False
        assert manager.api_key is None


class TestLLMRequest:
    """Test LLMRequest dataclass"""

    def test_default_values(self):
        """Test default values"""
        request = LLMRequest(prompt="test")
        assert request.max_tokens == 4000
        assert request.temperature == 0.3
        assert request.model == "gpt-4o-mini"

    def test_custom_values(self):
        """Test custom values"""
        request = LLMRequest(
            prompt="test", max_tokens=1000, temperature=0.8, model="o1-mini"
        )
        assert request.max_tokens == 1000
        assert request.temperature == 0.8
        assert request.model == "o1-mini"


class TestLLMResponse:
    """Test LLMResponse dataclass"""

    def test_successful_response(self):
        """Test successful response"""
        response = LLMResponse(content="Hello", model="gpt-4o")
        assert response.content == "Hello"
        assert response.model == "gpt-4o"
        assert response.error is None

    def test_error_response(self):
        """Test error response"""
        response = LLMResponse(
            content="Error occurred", model="gpt-4o", error="API failure"
        )
        assert response.error == "API failure"
