"""
MAIF Agents Module

Contains multi-agent support:
- Agent framework for AI agents
- Multi-agent coordination with state machine
- Event-driven architecture
- Inter-agent communication
- Task scheduling and orchestration
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
        AgentState as AgenticAgentState,
    )
except ImportError:
    MAIFAgent = None
    PerceptionSystem = None
    ReasoningSystem = None
    ExecutionSystem = None
    MemorySystem = None
    AgenticAgentState = None

try:
    from .multi_agent import (
        # New multi-agent framework classes
        MultiAgentOrchestrator,
        MultiAgentCoordinator,  # Alias for backward compatibility
        AgentStateMachine,
        AgentState,
        StateTransition,
        StateTransitionError,
        EventBus,
        Event,
        MessageBroker,
        Message,
        Blackboard,
        BlackboardEntry,
        TaskDependencyGraph,
        Task,
        AgentInfo,
        # Legacy classes for backward compatibility
        AgentCapabilities,
        MAIFExchangeProtocol,
        SemanticAlignment,
        SemanticAlignmentEngine,
        ExchangeMessage,
        ExchangeProtocolVersion,
        MessageType,
    )
except ImportError:
    MultiAgentOrchestrator = None
    MultiAgentCoordinator = None
    AgentStateMachine = None
    AgentState = None
    StateTransition = None
    StateTransitionError = None
    EventBus = None
    Event = None
    MessageBroker = None
    Message = None
    Blackboard = None
    BlackboardEntry = None
    TaskDependencyGraph = None
    Task = None
    AgentInfo = None
    AgentCapabilities = None
    MAIFExchangeProtocol = None
    SemanticAlignment = None
    SemanticAlignmentEngine = None
    ExchangeMessage = None
    ExchangeProtocolVersion = None
    MessageType = None

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
    "AgenticAgentState",
    # Multi-agent orchestration (new)
    "MultiAgentOrchestrator",
    "MultiAgentCoordinator",  # Alias for backward compatibility
    "AgentStateMachine",
    "AgentState",
    "StateTransition",
    "StateTransitionError",
    # Event-driven architecture
    "EventBus",
    "Event",
    # Inter-agent communication
    "MessageBroker",
    "Message",
    # Shared blackboard memory
    "Blackboard",
    "BlackboardEntry",
    # Task dependencies
    "TaskDependencyGraph",
    "Task",
    "AgentInfo",
    # Legacy multi-agent (backward compatibility)
    "AgentCapabilities",
    "MAIFExchangeProtocol",
    "SemanticAlignment",
    "SemanticAlignmentEngine",
    "ExchangeMessage",
    "ExchangeProtocolVersion",
    "MessageType",
    # Distributed
    "DistributedCoordinator",
    "DistributedMAIF",
    # Lifecycle
    "LifecycleManager",
    "LifecyclePolicy",
    "EnhancedLifecycleManager",
]

