"""
MAIF AWS Strands SDK Integration

Provides MAIF-backed callback handler for AWS Strands Agents SDK:
- MAIFStrandsCallback: Callback handler for agent provenance tracking
- create_composite_handler: Helper to create combined MAIF + printing handlers

The Strands Agents SDK (strands-agents) is an open source framework from AWS
for building AI agents with a model-driven approach.

Quick Start:
    from strands import Agent
    from maif.integrations.strands import MAIFStrandsCallback

    callback = MAIFStrandsCallback("agent.maif")
    agent = Agent(
        tools=[...],
        callback_handler=callback,
    )

    response = agent("What is the weather?")
    callback.finalize()

With Context Manager:
    from maif.integrations.strands import MAIFStrandsCallback

    with MAIFStrandsCallback("agent.maif") as callback:
        agent = Agent(callback_handler=callback)
        agent("Hello!")
    # Artifact automatically finalized

Combined with PrintingCallbackHandler:
    from maif.integrations.strands import create_composite_handler

    handler = create_composite_handler("agent.maif")
    agent = Agent(callback_handler=handler)
    response = agent("Hello!")

    # Access the MAIF callback to finalize
    maif_callback = handler.handlers[1]  # Second handler
    maif_callback.finalize()

All agent invocations, tool calls, and LLM responses are captured
in the MAIF artifact with full provenance and tamper-evident audit trails.

Note: Requires the strands-agents package:
    pip install strands-agents
"""

# Import the callback handler - works without strands-agents installed
# The callback is just a callable that can work with any compatible framework
from maif.integrations.strands.callback import (
    MAIFStrandsCallback,
    create_composite_handler,
    STRANDS_AVAILABLE,
)

__all__ = [
    "MAIFStrandsCallback",
    "create_composite_handler",
    "STRANDS_AVAILABLE",
]
