"""Pytest fixtures for chargebee_mapper tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chargebee_mapper.config import Config
from chargebee_mapper.entities import EntityDef


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample Config instance for testing."""
    return Config(
        api_key="test_api_key_123",
        site="test-site",
        max_concurrency=5,
        page_size=50,
        max_retries=3,
        rate_limit_rpm=100,
        log_level="DEBUG",
        output_dir=temp_dir,
    )


@pytest.fixture
def sample_entity():
    """Create a sample EntityDef for testing."""
    return EntityDef(
        name="Customer",
        key="customers",
        sdk_resource="Customer",
        response_key="customer",
    )


@pytest.fixture
def dependent_entity():
    """Create a sample dependent EntityDef for testing."""
    return EntityDef(
        name="Attached Item",
        key="attached_items",
        sdk_resource="AttachedItem",
        response_key="attached_item",
        requires_parent=True,
        parent_entity="items",
        parent_id_field="id",
    )


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for config loading tests."""
    env_vars = {
        "CHARGEBEE_API_KEY": "test_key_abc",
        "CHARGEBEE_SITE": "test-site-env",
        "CHARGEBEE_MAX_CONCURRENCY": "10",
        "CHARGEBEE_PAGE_SIZE": "75",
        "CHARGEBEE_MAX_RETRIES": "4",
        "CHARGEBEE_RATE_LIMIT_RPM": "200",
        "CHARGEBEE_LOG_LEVEL": "warning",
        "CHARGEBEE_OUTPUT_DIR": "/tmp/chargebee_output",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars
