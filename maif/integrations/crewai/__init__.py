"""
MAIF CrewAI Integration

Provides MAIF-backed provenance tracking for CrewAI multi-agent workflows.
All agent actions, task completions, and inter-agent communications are
automatically logged to MAIF artifacts with cryptographic signatures.

Usage:
    from crewai import Crew, Agent, Task
    from maif.integrations.crewai import MAIFCrewCallback

    callback = MAIFCrewCallback("crew_session.maif")
    
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2],
        task_callback=callback.on_task_complete,
        step_callback=callback.on_step,
    )
    
    result = crew.kickoff()
    callback.finalize()

All agent reasoning steps, tool usage, and task outputs are captured
in the MAIF artifact with full provenance and tamper-evident audit trails.
"""

from maif.integrations.crewai.callback import (
    MAIFCrewCallback,
    MAIFTaskCallback,
    MAIFStepCallback,
)

try:
    from maif.integrations.crewai.memory import MAIFCrewMemory
    _MEMORY_AVAILABLE = True
except ImportError:
    MAIFCrewMemory = None  # type: ignore
    _MEMORY_AVAILABLE = False

__all__ = [
    "MAIFCrewCallback",
    "MAIFTaskCallback", 
    "MAIFStepCallback",
]

if _MEMORY_AVAILABLE:
    __all__.append("MAIFCrewMemory")
