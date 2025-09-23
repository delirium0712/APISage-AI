"""
Test suite for security enhancements
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import time

from api.main import app
# Note: API key validation functions removed with endpoint simplification


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


class TestAPIKeyValidation:
    """Test API key validation functions"""

    def test_valid_api_key_format(self):
        """Test valid OpenAI API key format"""
        # Valid key format: sk- followed by 48 characters
        valid_key = "sk-" + "a" * 48
        assert is_valid_openai_api_key(valid_key) is True

    def test_invalid_api_key_format(self):
        """Test invalid API key formats"""
        invalid_keys = [
            "",  # Empty
            "invalid",  # No sk- prefix
            "sk-abc",  # Too short
            "sk-" + "a" * 47,  # 47 characters instead of 48
            "sk-" + "a" * 49,  # 49 characters instead of 48
            None,  # None value
        ]

        for key in invalid_keys:
            assert is_valid_openai_api_key(key) is False

    def test_mask_api_key(self):
        """Test API key masking"""
        # Standard key
        key = "sk-abc123def456ghi789jklmnopqrstuvwxyz123456789"
        masked = mask_api_key(key)
        assert masked.startswith("sk-")
        assert masked.endswith("789")
        assert "*" in masked
        assert len(masked) == len(key)

    def test_mask_short_api_key(self):
        """Test masking short keys"""
        short_key = "short"
        assert mask_api_key(short_key) == "***"
        assert mask_api_key("") == "***"
        assert mask_api_key(None) == "***"


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_basic(self):
        """Test basic rate limiting"""
        client_ip = "192.168.1.1"
        endpoint = "test-endpoint"

        # Should allow up to max_requests
        for i in range(5):
            assert (
                check_rate_limit(client_ip, endpoint, max_requests=5, window_seconds=60)
                is True
            )

        # Should block additional requests
        assert (
            check_rate_limit(client_ip, endpoint, max_requests=5, window_seconds=60)
            is False
        )

    def test_rate_limit_window_expiry(self):
        """Test rate limit window expiry"""
        client_ip = "192.168.1.2"
        endpoint = "test-endpoint"

        # Fill up the rate limit
        for i in range(3):
            assert (
                check_rate_limit(client_ip, endpoint, max_requests=3, window_seconds=1)
                is True
            )

        # Should be blocked now
        assert (
            check_rate_limit(client_ip, endpoint, max_requests=3, window_seconds=1)
            is False
        )

        # Wait for window to expire and try again (in real test, might need to mock time)
        time.sleep(1.1)
        assert (
            check_rate_limit(client_ip, endpoint, max_requests=3, window_seconds=1)
            is True
        )

    def test_rate_limit_different_clients(self):
        """Test rate limiting for different client IPs"""
        endpoint = "test-endpoint"

        # Different clients should have separate rate limits
        assert (
            check_rate_limit("192.168.1.1", endpoint, max_requests=2, window_seconds=60)
            is True
        )
        assert (
            check_rate_limit("192.168.1.2", endpoint, max_requests=2, window_seconds=60)
            is True
        )

        # Each should be able to make max requests
        assert (
            check_rate_limit("192.168.1.1", endpoint, max_requests=2, window_seconds=60)
            is True
        )
        assert (
            check_rate_limit("192.168.1.2", endpoint, max_requests=2, window_seconds=60)
            is True
        )


class TestSecureAPIKeyEndpoint:
    """Test secure API key endpoint implementation"""

    def test_set_api_key_valid_format(self, client):
        """Test setting API key with valid format"""
        valid_key = "sk-" + "a" * 48

        with patch("api.main.llm_manager.refresh_api_key", return_value=True):
            response = client.post("/set-api-key", json={"api_key": valid_key})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["is_set"] is True

    def test_set_api_key_invalid_format(self, client):
        """Test setting API key with invalid format"""
        # This should be caught by pydantic validation
        response = client.post("/set-api-key", json={"api_key": "invalid-key"})
        # Should return 422 for validation error
        assert response.status_code == 422

    def test_set_api_key_empty(self, client):
        """Test setting empty API key"""
        response = client.post("/set-api-key", json={"api_key": ""})
        # Should return 422 for validation error
        assert response.status_code == 422

    def test_set_api_key_rate_limiting(self, client):
        """Test rate limiting on API key endpoint"""
        valid_key = "sk-" + "a" * 48

        with patch("api.main.llm_manager.refresh_api_key", return_value=True):
            # Make requests up to the limit (5 requests per 5 minutes)
            for i in range(5):
                response = client.post("/set-api-key", json={"api_key": valid_key})
                assert response.status_code == 200

            # Next request should be rate limited
            response = client.post("/set-api-key", json={"api_key": valid_key})
            assert response.status_code == 429
            assert "Too many API key requests" in response.json()["detail"]

    def test_set_api_key_validation_failure(self, client):
        """Test API key validation failure in LLM manager"""
        valid_key = "sk-" + "a" * 48

        with patch("api.main.llm_manager.refresh_api_key", return_value=False):
            response = client.post("/set-api-key", json={"api_key": valid_key})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"
            assert data["is_set"] is False
            assert "Failed to validate API key" in data["message"]

    def test_set_api_key_server_error(self, client):
        """Test server error handling"""
        valid_key = "sk-" + "a" * 48

        with patch(
            "api.main.llm_manager.refresh_api_key",
            side_effect=Exception("Server error"),
        ):
            response = client.post("/set-api-key", json={"api_key": valid_key})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"
            assert data["is_set"] is False
            assert data["message"] == "Internal server error"


class TestSecurityLogging:
    """Test security-related logging"""

    def test_api_key_masking_in_logs(self, client):
        """Test that API keys are properly masked in logs"""
        valid_key = "sk-" + "a" * 48

        with patch("api.main.llm_manager.refresh_api_key", return_value=True):
            with patch("api.main.logger") as mock_logger:
                response = client.post("/set-api-key", json={"api_key": valid_key})

                # Check that logger was called with masked key
                assert response.status_code == 200
                mock_logger.info.assert_called()

                # Verify the key is masked in log calls
                log_calls = mock_logger.info.call_args_list
                for call in log_calls:
                    args, kwargs = call
                    if "masked_key" in kwargs:
                        masked_key = kwargs["masked_key"]
                        assert masked_key != valid_key
                        assert masked_key.startswith("sk-")
                        assert "*" in masked_key

    def test_rate_limit_logging(self, client):
        """Test rate limit violation logging"""
        valid_key = "sk-" + "a" * 48

        with patch("api.main.llm_manager.refresh_api_key", return_value=True):
            with patch("api.main.logger") as mock_logger:
                # Exhaust rate limit
                for i in range(5):
                    client.post("/set-api-key", json={"api_key": valid_key})

                # This should trigger rate limit logging
                response = client.post("/set-api-key", json={"api_key": valid_key})

                assert response.status_code == 429
                mock_logger.warning.assert_called_with(
                    "API key endpoint rate limited",
                    client_ip="testclient",
                    endpoint="set-api-key",
                )
