"""
Shared test fixtures for integration tests.

This module provides common fixtures used across all framework integration tests,
including API key management, temporary artifact paths, and skip markers.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def gemini_api_key():
    """Get Gemini API key from environment.
    
    Integration tests that require the Gemini API should use this fixture.
    Tests will be skipped if the API key is not set.
    
    Usage:
        @pytest.mark.integration
        def test_with_api(gemini_api_key):
            # gemini_api_key is the API key string
            pass
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return key


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts.
    
    The directory is automatically cleaned up after the test.
    
    Usage:
        def test_something(temp_dir):
            artifact_path = temp_dir / "test.maif"
    """
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def temp_artifact_path(temp_dir):
    """Create a temporary path for a MAIF artifact.
    
    Usage:
        def test_something(temp_artifact_path):
            checkpointer = MAIFCheckpointer(temp_artifact_path)
    """
    return str(temp_dir / "test.maif")


@pytest.fixture
def sample_state():
    """Sample LangGraph state for testing.
    
    Returns a simple state dict that can be used in checkpointer tests.
    """
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "context": {"topic": "greeting"},
    }


@pytest.fixture
def sample_checkpoint(sample_state):
    """Sample LangGraph checkpoint for testing.
    
    Returns a checkpoint dict structure compatible with LangGraph.
    """
    return {
        "v": 1,
        "id": "checkpoint-001",
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": sample_state,
        "channel_versions": {},
        "versions_seen": {},
    }


@pytest.fixture
def sample_config():
    """Sample LangGraph config for testing.
    
    Returns a config dict with thread_id and checkpoint_ns.
    """
    return {
        "configurable": {
            "thread_id": "test-thread-001",
            "checkpoint_ns": "",
        }
    }


# CrewAI-specific fixtures

@pytest.fixture
def sample_crew_callback(temp_artifact_path):
    """Create a MAIFCrewCallback for testing.
    
    Usage:
        def test_something(sample_crew_callback):
            callback = sample_crew_callback
            callback.on_task_complete(task_output)
    """
    try:
        from maif.integrations.crewai import MAIFCrewCallback
        callback = MAIFCrewCallback(temp_artifact_path)
        yield callback
        callback.finalize()
    except ImportError:
        pytest.skip("CrewAI integration not available")


@pytest.fixture
def sample_crew_memory(temp_artifact_path):
    """Create a MAIFCrewMemory for testing.
    
    Usage:
        def test_something(sample_crew_memory):
            memory = sample_crew_memory
            memory.save(content="test")
    """
    try:
        from maif.integrations.crewai import MAIFCrewMemory
        memory = MAIFCrewMemory(temp_artifact_path, auto_finalize=False)
        yield memory
        memory.finalize()
    except ImportError:
        pytest.skip("CrewAI integration not available")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require API keys)"
    )

