"""
MAIF AWS Strands SDK Integration

Provides MAIF-backed callback handler for AWS Strands Agents SDK:
- MAIFStrandsCallback: Callback handler for agent provenance

Status: Not yet implemented

Usage (when implemented):
    from strands import Agent
    from maif.integrations.strands import MAIFStrandsCallback
    
    callback = MAIFStrandsCallback("agent.maif")
    agent = Agent(callback_handler=callback)
    response = agent("What is the weather?")
    callback.finalize()

See maif/integrations/INTEGRATION_PLAN.md for implementation details.
"""

# Placeholder - to be implemented
# from maif.integrations.strands.callback import MAIFStrandsCallback

__all__ = [
    # "MAIFStrandsCallback",
]


def __getattr__(name: str):
    """Raise informative error for unimplemented components."""
    if name in ("MAIFStrandsCallback",):
        raise NotImplementedError(
            f"{name} is not yet implemented. "
            "See maif/integrations/INTEGRATION_PLAN.md for implementation details."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

