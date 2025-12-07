#!/usr/bin/env python3
"""
LangGraph + MAIF Integration Quick Start

This example demonstrates using MAIFCheckpointer with LangGraph for
cryptographic provenance tracking of graph state transitions.

Requirements:
    pip install maif[integrations]
    # or
    pip install maif langgraph

Usage:
    python langgraph_quickstart.py

What this example does:
    1. Creates a simple multi-step graph
    2. Uses MAIFCheckpointer for state persistence
    3. Runs the graph with checkpoint tracking
    4. Verifies the resulting MAIF artifact
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import TypedDict, List, Annotated
from operator import add

# Add parent paths for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Run the LangGraph + MAIF example."""
    
    # Check for LangGraph
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError:
        print("Error: LangGraph is required for this example.")
        print("Install it with: pip install langgraph")
        sys.exit(1)
    
    from maif.integrations.langgraph import MAIFCheckpointer
    from maif import MAIFDecoder
    
    # Create temporary directory for the artifact
    temp_dir = tempfile.mkdtemp()
    artifact_path = os.path.join(temp_dir, "langgraph_demo.maif")
    
    print("=" * 60)
    print("LangGraph + MAIF Integration Demo")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Define the state schema
    # -------------------------------------------------------------------------
    
    class AgentState(TypedDict):
        """State for our simple agent."""
        messages: Annotated[List[str], add]  # Accumulate messages
        step_count: int
        data: dict
    
    print("Step 1: Define state schema")
    print("  - messages: List of messages (accumulated)")
    print("  - step_count: Current step number")
    print("  - data: Arbitrary data dict")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Define graph nodes
    # -------------------------------------------------------------------------
    
    def initialize(state: AgentState) -> AgentState:
        """Initialize the agent state."""
        return {
            "messages": ["[init] Agent initialized"],
            "step_count": 1,
            "data": {"initialized": True},
        }
    
    def process(state: AgentState) -> AgentState:
        """Process the current state."""
        current_step = state.get("step_count", 0)
        return {
            "messages": [f"[process] Processing step {current_step}"],
            "step_count": current_step + 1,
            "data": {**state.get("data", {}), "processed": True},
        }
    
    def finalize_node(state: AgentState) -> AgentState:
        """Finalize the agent execution."""
        return {
            "messages": ["[finalize] Agent execution complete"],
            "step_count": state.get("step_count", 0) + 1,
            "data": {**state.get("data", {}), "finalized": True},
        }
    
    print("Step 2: Define graph nodes")
    print("  - initialize: Set up initial state")
    print("  - process: Process current state")
    print("  - finalize_node: Complete execution")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Build the graph
    # -------------------------------------------------------------------------
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("initialize", initialize)
    graph.add_node("process", process)
    graph.add_node("finalize", finalize_node)
    
    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "process")
    graph.add_edge("process", "finalize")
    graph.add_edge("finalize", END)
    
    print("Step 3: Build graph")
    print("  START -> initialize -> process -> finalize -> END")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Create MAIF checkpointer and compile
    # -------------------------------------------------------------------------
    
    print("Step 4: Create MAIF checkpointer")
    print(f"  Artifact path: {artifact_path}")
    print()
    
    checkpointer = MAIFCheckpointer(
        artifact_path=artifact_path,
        agent_id="langgraph_demo"
    )
    
    # Compile graph with checkpointer
    app = graph.compile(checkpointer=checkpointer)
    
    # -------------------------------------------------------------------------
    # Step 5: Run the graph
    # -------------------------------------------------------------------------
    
    print("Step 5: Run the graph")
    print("-" * 40)
    
    initial_state = {
        "messages": [],
        "step_count": 0,
        "data": {},
    }
    
    config = {"configurable": {"thread_id": "demo-session-001"}}
    
    result = app.invoke(initial_state, config=config)
    
    print("Execution result:")
    print(f"  Messages: {result['messages']}")
    print(f"  Step count: {result['step_count']}")
    print(f"  Data: {result['data']}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 6: Run a second session
    # -------------------------------------------------------------------------
    
    print("Step 6: Run second session (different thread)")
    print("-" * 40)
    
    config2 = {"configurable": {"thread_id": "demo-session-002"}}
    result2 = app.invoke(initial_state, config=config2)
    
    print("Second session result:")
    print(f"  Messages: {result2['messages']}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 7: List checkpoints
    # -------------------------------------------------------------------------
    
    print("Step 7: List saved checkpoints")
    print("-" * 40)
    
    # List checkpoints for first session
    checkpoints_1 = list(checkpointer.list({"configurable": {"thread_id": "demo-session-001"}}))
    print(f"Session 1 checkpoints: {len(checkpoints_1)}")
    
    checkpoints_2 = list(checkpointer.list({"configurable": {"thread_id": "demo-session-002"}}))
    print(f"Session 2 checkpoints: {len(checkpoints_2)}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 8: Finalize and verify artifact
    # -------------------------------------------------------------------------
    
    print("Step 8: Finalize and verify artifact")
    print("-" * 40)
    
    checkpointer.finalize()
    print("Artifact finalized (signed and sealed)")
    print()
    
    # Verify the artifact
    decoder = MAIFDecoder(artifact_path)
    decoder.load()
    
    is_valid, errors = decoder.verify_integrity()
    print(f"Integrity verification: {'PASSED' if is_valid else 'FAILED'}")
    
    if not is_valid:
        print(f"Errors: {errors}")
    
    print(f"Total blocks in artifact: {len(decoder.blocks)}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 9: Inspect artifact contents
    # -------------------------------------------------------------------------
    
    print("Step 9: Inspect artifact contents")
    print("-" * 40)
    
    for i, block in enumerate(decoder.blocks[:10]):  # Show first 10 blocks
        block_type = block.metadata.get("type", "unknown")
        timestamp = block.metadata.get("timestamp", "N/A")
        thread_id = block.metadata.get("thread_id", "N/A")
        
        print(f"Block {i}: type={block_type}, thread={thread_id}")
    
    if len(decoder.blocks) > 10:
        print(f"... and {len(decoder.blocks) - 10} more blocks")
    
    print()
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    print("=" * 60)
    print("Demo complete!")
    print()
    print(f"Artifact saved to: {artifact_path}")
    print()
    print("Key takeaways:")
    print("  1. MAIFCheckpointer provides drop-in checkpoint persistence")
    print("  2. All state transitions are cryptographically signed")
    print("  3. Multiple sessions can be tracked in one artifact")
    print("  4. Artifact integrity can be verified at any time")
    print("=" * 60)
    
    # Optional: Clean up temp directory
    # import shutil
    # shutil.rmtree(temp_dir)
    
    return artifact_path


if __name__ == "__main__":
    main()

