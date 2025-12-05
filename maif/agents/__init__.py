"""
MAIF Agents Module

Contains multi-agent support:
- Agent framework for AI agents
- Multi-agent coordination
- Distributed processing
- Lifecycle management
"""

try:
    from .agentic_framework import (
        MAIFAgent,
        PerceptionSystem,
        ReasoningSystem,
        ExecutionSystem,
        MemorySystem,
        AgentState,
    )
except ImportError:
    MAIFAgent = None
    PerceptionSystem = None
    ReasoningSystem = None
    ExecutionSystem = None
    MemorySystem = None
    AgentState = None

try:
    from .multi_agent import (
        MultiAgentCoordinator,
        AgentCapabilities,
        MAIFExchangeProtocol,
    )
except ImportError:
    MultiAgentCoordinator = None
    AgentCapabilities = None
    MAIFExchangeProtocol = None

try:
    from .distributed import DistributedCoordinator, DistributedMAIF
except ImportError:
    DistributedCoordinator = None
    DistributedMAIF = None

try:
    from .lifecycle_management import LifecycleManager, LifecyclePolicy
except ImportError:
    LifecycleManager = None
    LifecyclePolicy = None

try:
    from .lifecycle_management_enhanced import EnhancedLifecycleManager
except ImportError:
    EnhancedLifecycleManager = None

__all__ = [
    # Agent framework
    "MAIFAgent",
    "PerceptionSystem",
    "ReasoningSystem",
    "ExecutionSystem",
    "MemorySystem",
    "AgentState",
    # Multi-agent
    "MultiAgentCoordinator",
    "AgentCapabilities",
    "MAIFExchangeProtocol",
    # Distributed
    "DistributedCoordinator",
    "DistributedMAIF",
    # Lifecycle
    "LifecycleManager",
    "LifecyclePolicy",
    "EnhancedLifecycleManager",
]

