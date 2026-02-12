"""Unit tests for chargebee_mapper.config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chargebee_mapper.config import Config, load_config


class TestConfig:
    """Tests for Config dataclass."""

    def test_basic_init(self):
        """Test basic Config initialization."""
        config = Config(
            api_key="test_key",
            site="test-site",
        )
        
        assert config.api_key == "test_key"
        assert config.site == "test-site"
        assert config.max_concurrency == 20
        assert config.page_size == 100
        assert config.max_retries == 5
        assert config.rate_limit_rpm == 150
        assert config.log_level == "INFO"
        assert config.output_dir == Path("output")

    def test_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            api_key="custom_key",
            site="custom-site",
            max_concurrency=10,
            page_size=50,
            max_retries=3,
            rate_limit_rpm=1000,
            log_level="DEBUG",
            output_dir=Path("/custom/path"),
        )
        
        assert config.max_concurrency == 10
        assert config.page_size == 50
        assert config.max_retries == 3
        assert config.rate_limit_rpm == 1000
        assert config.log_level == "DEBUG"
        assert config.output_dir == Path("/custom/path")

    def test_base_url_property(self):
        """Test base_url property generates correct URL."""
        config = Config(api_key="key", site="my-company")
        assert config.base_url == "https://my-company.chargebee.com/api/v2"

    def test_log_file_property(self):
        """Test log_file property returns correct path."""
        config = Config(
            api_key="key",
            site="site",
            output_dir=Path("/var/log"),
        )
        assert config.log_file == Path("/var/log/chargebee_mapper.log")


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_env(self, mock_env_vars):
        """Test loading config from environment variables."""
        config = load_config()
        
        assert config.api_key == "test_key_abc"
        assert config.site == "test-site-env"
        assert config.max_concurrency == 10
        assert config.page_size == 75
        assert config.max_retries == 4
        assert config.rate_limit_rpm == 200
        assert config.log_level == "WARNING"
        assert config.output_dir == Path("/tmp/chargebee_output")

    def test_load_config_page_size_capped(self):
        """Test that page_size is capped at 100."""
        env_vars = {
            "CHARGEBEE_API_KEY": "key",
            "CHARGEBEE_SITE": "site",
            "CHARGEBEE_PAGE_SIZE": "500",  # Exceeds max
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = load_config()
            assert config.page_size == 100  # Should be capped

    def test_load_config_missing_api_key(self):
        """Test that missing API key causes exit."""
        # Mock load_dotenv to prevent loading actual .env file
        with patch("chargebee_mapper.config.load_dotenv"):
            with patch.dict(os.environ, {"CHARGEBEE_SITE": "site"}, clear=True):
                with pytest.raises(SystemExit) as exc_info:
                    load_config()
                assert exc_info.value.code == 1

    def test_load_config_missing_site(self):
        """Test that missing site causes exit."""
        # Mock load_dotenv to prevent loading actual .env file
        with patch("chargebee_mapper.config.load_dotenv"):
            with patch.dict(os.environ, {"CHARGEBEE_API_KEY": "key"}, clear=True):
                with pytest.raises(SystemExit) as exc_info:
                    load_config()
                assert exc_info.value.code == 1

    def test_load_config_defaults(self):
        """Test that defaults are used when optional vars not set."""
        env_vars = {
            "CHARGEBEE_API_KEY": "key",
            "CHARGEBEE_SITE": "site",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = load_config()
            
            assert config.max_concurrency == 20
            assert config.page_size == 100
            assert config.max_retries == 5
            assert config.rate_limit_rpm == 150
            assert config.log_level == "INFO"
            assert config.output_dir == Path("output")
