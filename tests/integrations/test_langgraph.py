"""
Tests for MAIF LangGraph Integration.

This module contains unit tests and integration tests for the MAIFCheckpointer.
Unit tests can run without external dependencies.
Integration tests require the GEMINI_API_KEY environment variable.

Run unit tests only:
    pytest tests/integrations/test_langgraph.py -m "not integration"

Run all tests including integration:
    GEMINI_API_KEY=your_key pytest tests/integrations/test_langgraph.py
"""

import os
import json
import time
import pytest
import tempfile
import shutil
from pathlib import Path


class TestMAIFCheckpointerUnit:
    """Unit tests for MAIFCheckpointer (no external API required)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "test.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_import(self):
        """Test that MAIFCheckpointer can be imported."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
            assert MAIFCheckpointer is not None
        except ImportError as e:
            if "langgraph" in str(e).lower():
                pytest.skip("LangGraph not installed")
            raise
    
    def test_initialization(self):
        """Test checkpointer initialization."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        assert checkpointer.artifact_path == Path(self.artifact_path)
        assert checkpointer._agent_id == "langgraph_checkpointer"
    
    def test_initialization_with_custom_agent_id(self):
        """Test checkpointer initialization with custom agent ID."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(
            self.artifact_path,
            agent_id="custom-agent"
        )
        
        assert checkpointer._agent_id == "custom-agent"
    
    def test_get_artifact_path(self):
        """Test get_artifact_path method."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        assert checkpointer.get_artifact_path() == self.artifact_path
    
    def test_put_and_get_checkpoint(self):
        """Test storing and retrieving a checkpoint."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        # Create a sample checkpoint
        config = {
            "configurable": {
                "thread_id": "test-thread-001",
                "checkpoint_ns": "",
            }
        }
        
        checkpoint = {
            "v": 1,
            "id": "checkpoint-001",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {
                "messages": [{"role": "user", "content": "Hello"}]
            },
            "channel_versions": {},
            "versions_seen": {},
        }
        
        metadata = {
            "source": "input",
            "step": 0,
        }
        
        # Store the checkpoint
        result_config = checkpointer.put(config, checkpoint, metadata)
        
        assert "configurable" in result_config
        assert result_config["configurable"]["thread_id"] == "test-thread-001"
        assert result_config["configurable"]["checkpoint_id"] == "checkpoint-001"
        
        # Retrieve the checkpoint
        retrieved = checkpointer.get_tuple(result_config)
        
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == "checkpoint-001"
        assert retrieved.checkpoint["channel_values"]["messages"][0]["content"] == "Hello"
        
        # Finalize
        checkpointer.finalize()
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        # Store multiple checkpoints
        config = {
            "configurable": {
                "thread_id": "test-thread-002",
                "checkpoint_ns": "",
            }
        }
        
        for i in range(3):
            checkpoint = {
                "v": 1,
                "id": f"checkpoint-{i:03d}",
                "ts": f"2024-01-0{i+1}T00:00:00Z",
                "channel_values": {"step": i},
                "channel_versions": {},
                "versions_seen": {},
            }
            
            metadata = {"source": "test", "step": i}
            checkpointer.put(config, checkpoint, metadata)
            time.sleep(0.01)  # Ensure different timestamps
        
        # List all checkpoints for the thread
        checkpoints = list(checkpointer.list(config))
        
        assert len(checkpoints) == 3
        # Should be in reverse chronological order
        assert checkpoints[0].checkpoint["id"] == "checkpoint-002"
        assert checkpoints[1].checkpoint["id"] == "checkpoint-001"
        assert checkpoints[2].checkpoint["id"] == "checkpoint-000"
        
        # Test with limit
        limited = list(checkpointer.list(config, limit=2))
        assert len(limited) == 2
        
        checkpointer.finalize()
    
    def test_put_writes(self):
        """Test storing intermediate writes."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        config = {
            "configurable": {
                "thread_id": "test-thread-003",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-001",
            }
        }
        
        writes = [
            ("messages", {"role": "assistant", "content": "Hello!"}),
            ("context", {"updated": True}),
        ]
        
        # This should not raise
        checkpointer.put_writes(config, writes, task_id="node_001")
        
        checkpointer.finalize()
    
    def test_context_manager(self):
        """Test using checkpointer as context manager."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        with MAIFCheckpointer(self.artifact_path) as checkpointer:
            config = {
                "configurable": {
                    "thread_id": "test-thread-004",
                    "checkpoint_ns": "",
                }
            }
            
            checkpoint = {
                "v": 1,
                "id": "checkpoint-ctx",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"test": True},
                "channel_versions": {},
                "versions_seen": {},
            }
            
            checkpointer.put(config, checkpoint, {"source": "test"})
        
        # Artifact should be finalized after context exit
        assert os.path.exists(self.artifact_path)
    
    def test_artifact_creation(self):
        """Test that MAIF artifact is created correctly."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        config = {
            "configurable": {
                "thread_id": "test-thread-005",
                "checkpoint_ns": "",
            }
        }
        
        checkpoint = {
            "v": 1,
            "id": "checkpoint-artifact",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"data": "test"},
            "channel_versions": {},
            "versions_seen": {},
        }
        
        checkpointer.put(config, checkpoint, {"source": "test"})
        checkpointer.finalize()
        
        # Verify artifact exists and can be read
        assert os.path.exists(self.artifact_path)
        
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        # Should have at least session_start, checkpoint, session_end blocks
        assert len(decoder.blocks) >= 2
        
        # Verify integrity
        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"
    
    def test_get_nonexistent_checkpoint(self):
        """Test getting a checkpoint that doesn't exist."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        config = {
            "configurable": {
                "thread_id": "nonexistent-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "nonexistent-checkpoint",
            }
        }
        
        result = checkpointer.get_tuple(config)
        assert result is None
    
    def test_empty_thread_id(self):
        """Test behavior with empty thread_id."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        
        config = {
            "configurable": {
                "thread_id": "",
                "checkpoint_ns": "",
            }
        }
        
        result = checkpointer.get_tuple(config)
        assert result is None
    
    def test_load_existing_checkpoints(self):
        """Test loading checkpoints from existing artifact."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        # Create checkpointer and store checkpoint
        checkpointer1 = MAIFCheckpointer(self.artifact_path)
        
        config = {
            "configurable": {
                "thread_id": "persistent-thread",
                "checkpoint_ns": "",
            }
        }
        
        checkpoint = {
            "v": 1,
            "id": "persistent-checkpoint",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"persistent": True},
            "channel_versions": {},
            "versions_seen": {},
        }
        
        checkpointer1.put(config, checkpoint, {"source": "test"})
        checkpointer1.finalize()
        
        # Create new checkpointer from existing artifact
        checkpointer2 = MAIFCheckpointer(self.artifact_path)
        
        # Should be able to retrieve the checkpoint
        retrieved = checkpointer2.get_tuple({
            "configurable": {
                "thread_id": "persistent-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "persistent-checkpoint",
            }
        })
        
        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["persistent"] is True


class TestMAIFCheckpointerAsync:
    """Test async methods of MAIFCheckpointer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "test_async.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_async_put_and_get(self):
        """Test async put and get methods (runs sync wrappers)."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        import asyncio
        
        async def run_test():
            checkpointer = MAIFCheckpointer(self.artifact_path)
            
            config = {
                "configurable": {
                    "thread_id": "async-thread",
                    "checkpoint_ns": "",
                }
            }
            
            checkpoint = {
                "v": 1,
                "id": "async-checkpoint",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"async": True},
                "channel_versions": {},
                "versions_seen": {},
            }
            
            # Async put
            result_config = await checkpointer.aput(
                config, checkpoint, {"source": "async_test"}
            )
            
            # Async get
            retrieved = await checkpointer.aget_tuple(result_config)
            
            assert retrieved is not None
            assert retrieved.checkpoint["channel_values"]["async"] is True
            
            checkpointer.finalize()
        
        asyncio.get_event_loop().run_until_complete(run_test())
    
    def test_async_list(self):
        """Test async list method (runs sync wrappers)."""
        try:
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        import asyncio
        
        async def run_test():
            checkpointer = MAIFCheckpointer(self.artifact_path)
            
            config = {
                "configurable": {
                    "thread_id": "async-list-thread",
                    "checkpoint_ns": "",
                }
            }
            
            # Store checkpoints
            for i in range(2):
                checkpoint = {
                    "v": 1,
                    "id": f"async-list-{i}",
                    "ts": f"2024-01-0{i+1}T00:00:00Z",
                    "channel_values": {"index": i},
                    "channel_versions": {},
                    "versions_seen": {},
                }
                await checkpointer.aput(config, checkpoint, {"source": "test"})
            
            # Async list
            results = []
            async for item in checkpointer.alist(config):
                results.append(item)
            
            assert len(results) == 2
            
            checkpointer.finalize()
        
        asyncio.get_event_loop().run_until_complete(run_test())


@pytest.mark.integration
class TestMAIFCheckpointerIntegration:
    """Integration tests requiring LangGraph and optionally Gemini API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.temp_dir, "integration.maif")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_with_simple_graph(self):
        """Test checkpointer with a simple LangGraph."""
        try:
            from langgraph.graph import StateGraph, START, END
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        from typing import TypedDict, List, Annotated
        from operator import add
        
        # Define state
        class State(TypedDict):
            messages: Annotated[List[str], add]
            count: int
        
        # Define nodes
        def increment(state: State) -> State:
            return {"messages": ["incremented"], "count": state.get("count", 0) + 1}
        
        def double(state: State) -> State:
            return {"messages": ["doubled"], "count": state.get("count", 0) * 2}
        
        # Build graph
        graph = StateGraph(State)
        graph.add_node("increment", increment)
        graph.add_node("double", double)
        graph.add_edge(START, "increment")
        graph.add_edge("increment", "double")
        graph.add_edge("double", END)
        
        # Compile with MAIF checkpointer
        checkpointer = MAIFCheckpointer(self.artifact_path)
        app = graph.compile(checkpointer=checkpointer)
        
        # Run the graph
        config = {"configurable": {"thread_id": "simple-graph-test"}}
        result = app.invoke({"messages": [], "count": 5}, config=config)
        
        assert result["count"] == 12  # (5 + 1) * 2
        assert "incremented" in result["messages"]
        assert "doubled" in result["messages"]
        
        # Verify checkpoints were saved
        checkpoints = list(checkpointer.list(config))
        assert len(checkpoints) > 0
        
        # Finalize and verify artifact
        checkpointer.finalize()
        
        from maif import MAIFDecoder
        decoder = MAIFDecoder(self.artifact_path)
        decoder.load()
        
        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"
    
    def test_multi_turn_conversation(self):
        """Test checkpointer with multi-turn conversation pattern."""
        try:
            from langgraph.graph import StateGraph, START, END
            from maif.integrations.langgraph import MAIFCheckpointer
        except ImportError:
            pytest.skip("LangGraph not installed")
        
        from typing import TypedDict, List, Annotated
        from operator import add
        
        class ConversationState(TypedDict):
            messages: Annotated[List[dict], add]
        
        def echo_bot(state: ConversationState) -> ConversationState:
            last_message = state["messages"][-1] if state["messages"] else {"content": ""}
            response = {"role": "assistant", "content": f"Echo: {last_message.get('content', '')}"}
            return {"messages": [response]}
        
        graph = StateGraph(ConversationState)
        graph.add_node("echo", echo_bot)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        
        checkpointer = MAIFCheckpointer(self.artifact_path)
        app = graph.compile(checkpointer=checkpointer)
        
        thread_id = "multi-turn-test"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Turn 1
        result1 = app.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config=config
        )
        assert "Echo: Hello" in result1["messages"][-1]["content"]
        
        # Turn 2 - continue the conversation
        result2 = app.invoke(
            {"messages": [{"role": "user", "content": "How are you?"}]},
            config=config
        )
        assert "Echo: How are you?" in result2["messages"][-1]["content"]
        
        # Verify all checkpoints
        checkpoints = list(checkpointer.list(config))
        assert len(checkpoints) >= 2
        
        checkpointer.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

