#!/usr/bin/env python3
"""
MAIF + LangGraph Quickstart

The simplest possible example showing how to add cryptographic provenance
to any LangGraph application with just 2 lines of code change.

Before (SqliteSaver):
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver.from_conn_string(":memory:")

After (MAIF):
    from maif.integrations.langgraph import MAIFCheckpointer
    checkpointer = MAIFCheckpointer("state.maif")

That's it! Your graph now has:
- Ed25519 signatures on every state change
- Hash-chained blocks for tamper detection
- Full audit trail for compliance
"""

import sys
from pathlib import Path

# Add parent path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TypedDict, Annotated, List
from operator import add

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# MAIF import - this is the only change you need!
from maif.integrations.langgraph import MAIFCheckpointer


# Define your state (same as any LangGraph app)
class ChatState(TypedDict):
    messages: Annotated[List[str], add]
    turn_count: int


# Define your nodes (same as any LangGraph app)
def user_input(state: ChatState) -> ChatState:
    """Simulate user input."""
    return {
        "messages": [f"User: Hello! (turn {state['turn_count'] + 1})"],
        "turn_count": state["turn_count"] + 1,
    }


def assistant_response(state: ChatState) -> ChatState:
    """Simulate assistant response."""
    return {
        "messages": [f"Assistant: Hi there! I see this is turn {state['turn_count']}."],
        "turn_count": state["turn_count"],
    }


def should_continue(state: ChatState) -> str:
    """Decide whether to continue the conversation."""
    if state["turn_count"] >= 3:
        return "end"
    return "continue"


def main():
    # Build your graph (same as any LangGraph app)
    graph = StateGraph(ChatState)
    graph.add_node("user", user_input)
    graph.add_node("assistant", assistant_response)
    
    graph.add_edge(START, "user")
    graph.add_edge("user", "assistant")
    graph.add_conditional_edges(
        "assistant",
        should_continue,
        {"continue": "user", "end": END}
    )
    
    # === THE ONLY CHANGE: Use MAIFCheckpointer instead of SqliteSaver ===
    checkpointer = MAIFCheckpointer("quickstart_state.maif")
    
    # Compile with the checkpointer
    app = graph.compile(checkpointer=checkpointer)
    
    # Run the conversation
    print("Running conversation with MAIF provenance tracking...")
    print("-" * 50)
    
    result = app.invoke(
        {"messages": [], "turn_count": 0},
        config={"configurable": {"thread_id": "quickstart-demo"}}
    )
    
    # Show results
    print("\nConversation:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    # Finalize (seals the artifact)
    checkpointer.finalize()
    
    # Verify the provenance
    print("\n" + "-" * 50)
    print("Verifying cryptographic provenance...")
    
    from maif import MAIFDecoder
    decoder = MAIFDecoder("quickstart_state.maif")
    decoder.load()
    
    is_valid, errors = decoder.verify_integrity()
    
    print(f"\nArtifact: quickstart_state.maif")
    print(f"Blocks: {len(decoder.blocks)}")
    print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")
    
    if is_valid:
        print("\nYour LangGraph now has cryptographic provenance!")
        print("Every state change is signed and tamper-evident.")
    
    # Cleanup
    import os
    os.remove("quickstart_state.maif")


if __name__ == "__main__":
    main()

