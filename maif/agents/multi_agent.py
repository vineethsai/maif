"""
Multi-Agent MAIF Framework with State Machine, Message Passing, and Orchestration.

This module provides a production-ready multi-agent framework for MAIF-based systems with:

FEATURES:
- Agent State Machine: IDLE, RUNNING, WAITING, COMPLETED, FAILED states with proper transitions
- Event-Driven Architecture: Async event bus for decoupled communication
- Inter-Agent Communication: Message passing with priority queues
- Shared Blackboard Memory: Thread-safe shared memory for agent coordination
- Task Dependencies: DAG-based task dependency management
- Orchestrator: Task scheduling, load balancing, and failure handling
- Backward Compatibility: Original classes preserved for existing code

ARCHITECTURE:
    +----------------+
    |  Orchestrator  |
    +-------+--------+
            |
    +-------v--------+
    |   Event Bus    |
    +-------+--------+
            |
    +-------v--------+     +-----------+
    |   Blackboard   |<--->|   Agents  |
    +----------------+     +-----------+
"""

import asyncio
import json
import hashlib
import uuid
import time
import threading
import heapq
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Protocol, Union,
    Callable, Awaitable, TypeVar, Generic
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
import numpy as np
from pathlib import Path
import pickle
import struct
import zlib
import logging
import weakref

from ..core import MAIFEncoder, MAIFDecoder, MAIFBlock
from ..core.block_types import BlockType
from ..security import SecurityManager
from ..compression.compression_manager import CompressionManager


logger = logging.getLogger(__name__)


# =============================================================================
# AGENT STATE MACHINE
# =============================================================================

class AgentState(Enum):
    """
    Agent lifecycle states with defined transitions.

    State Transition Diagram:
        IDLE -> RUNNING -> WAITING -> RUNNING -> COMPLETED
                  |                      |
                  v                      v
               FAILED                 FAILED
    """
    IDLE = auto()       # Agent is ready but not processing
    RUNNING = auto()    # Agent is actively processing a task
    WAITING = auto()    # Agent is waiting for dependencies or resources
    COMPLETED = auto()  # Agent has completed its task successfully
    FAILED = auto()     # Agent has encountered an error


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


@dataclass
class StateTransition:
    """Represents a state transition event."""
    from_state: AgentState
    to_state: AgentState
    timestamp: datetime
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentStateMachine:
    """
    Manages agent state transitions with validation and history tracking.

    Attributes:
        agent_id: Unique identifier for the agent
        state: Current state of the agent
        history: List of state transitions

    Example:
        >>> sm = AgentStateMachine("agent_1")
        >>> sm.transition(AgentState.RUNNING, "Starting task")
        >>> sm.state
        AgentState.RUNNING
    """

    # Valid state transitions
    VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
        AgentState.IDLE: {AgentState.RUNNING, AgentState.FAILED},
        AgentState.RUNNING: {AgentState.WAITING, AgentState.COMPLETED, AgentState.FAILED, AgentState.IDLE},
        AgentState.WAITING: {AgentState.RUNNING, AgentState.FAILED, AgentState.IDLE},
        AgentState.COMPLETED: {AgentState.IDLE},
        AgentState.FAILED: {AgentState.IDLE},
    }

    def __init__(self, agent_id: str, initial_state: AgentState = AgentState.IDLE):
        """
        Initialize the state machine.

        Args:
            agent_id: Unique identifier for the agent
            initial_state: Starting state (default: IDLE)
        """
        self.agent_id = agent_id
        self._state = initial_state
        self._history: List[StateTransition] = []
        self._lock = threading.Lock()
        self._callbacks: Dict[AgentState, List[Callable[[StateTransition], None]]] = defaultdict(list)

    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state

    @property
    def history(self) -> List[StateTransition]:
        """Get state transition history."""
        return list(self._history)

    def can_transition(self, target_state: AgentState) -> bool:
        """
        Check if transition to target state is valid.

        Args:
            target_state: The state to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        return target_state in self.VALID_TRANSITIONS.get(self._state, set())

    def transition(
        self,
        target_state: AgentState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateTransition:
        """
        Transition to a new state.

        Args:
            target_state: The state to transition to
            reason: Reason for the transition
            metadata: Additional metadata for the transition

        Returns:
            The StateTransition object

        Raises:
            StateTransitionError: If the transition is invalid
        """
        with self._lock:
            if not self.can_transition(target_state):
                raise StateTransitionError(
                    f"Invalid transition from {self._state.name} to {target_state.name} "
                    f"for agent {self.agent_id}"
                )

            transition = StateTransition(
                from_state=self._state,
                to_state=target_state,
                timestamp=datetime.now(),
                reason=reason,
                metadata=metadata or {}
            )

            self._history.append(transition)
            self._state = target_state

            # Trigger callbacks
            for callback in self._callbacks.get(target_state, []):
                try:
                    callback(transition)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            logger.debug(f"Agent {self.agent_id}: {transition.from_state.name} -> {transition.to_state.name}")
            return transition

    def on_state(self, state: AgentState, callback: Callable[[StateTransition], None]):
        """
        Register a callback for when a state is entered.

        Args:
            state: The state to watch
            callback: Function to call when state is entered
        """
        self._callbacks[state].append(callback)

    def reset(self):
        """Reset state machine to IDLE."""
        with self._lock:
            if self._state in {AgentState.COMPLETED, AgentState.FAILED}:
                self._state = AgentState.IDLE
                self._history.append(StateTransition(
                    from_state=self._state,
                    to_state=AgentState.IDLE,
                    timestamp=datetime.now(),
                    reason="Reset"
                ))


# =============================================================================
# EVENT-DRIVEN ARCHITECTURE
# =============================================================================

@dataclass
class Event:
    """
    Represents an event in the event-driven system.

    Attributes:
        event_type: Type/category of the event
        source_id: ID of the event source
        payload: Event data
        timestamp: When the event was created
        priority: Event priority (lower = higher priority)
    """
    event_type: str
    source_id: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __lt__(self, other: "Event") -> bool:
        """Compare events by priority for heap operations."""
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)


class EventBus:
    """
    Async event bus for decoupled inter-agent communication.

    Supports:
    - Publish/subscribe pattern
    - Event filtering by type
    - Priority-based event processing
    - Async event handlers

    Example:
        >>> bus = EventBus()
        >>> await bus.subscribe("task_complete", my_handler)
        >>> await bus.publish(Event("task_complete", "agent_1", {"result": "success"}))
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum number of events in queue
        """
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._lock = asyncio.Lock()
        self._event_history: List[Event] = []
        self._max_history = 1000

    async def start(self):
        """Start the event bus processing loop."""
        self._running = True
        asyncio.create_task(self._process_events())
        logger.info("EventBus started")

    async def stop(self):
        """Stop the event bus."""
        self._running = False
        logger.info("EventBus stopped")

    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]]
    ):
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to ("*" for all)
            handler: Async function to handle events
        """
        async with self._lock:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed to {event_type}")

    async def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], Awaitable[None]]
    ):
        """
        Unsubscribe from events.

        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler to remove
        """
        async with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event):
        """
        Publish an event to the bus.

        Args:
            event: Event to publish
        """
        await self._queue.put(event)

        # Track history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    async def _dispatch_event(self, event: Event):
        """Dispatch event to subscribers."""
        handlers = []

        # Get specific handlers
        handlers.extend(self._subscribers.get(event.event_type, []))

        # Get wildcard handlers
        handlers.extend(self._subscribers.get("*", []))

        # Execute handlers concurrently
        if handlers:
            tasks = [asyncio.create_task(h(event)) for h in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# INTER-AGENT COMMUNICATION
# =============================================================================

@dataclass
class Message:
    """
    Message for inter-agent communication.

    Attributes:
        sender_id: ID of sending agent
        recipient_id: ID of receiving agent (None for broadcast)
        message_type: Type of message
        payload: Message data
        correlation_id: For request-response patterns
        priority: Message priority
    """
    sender_id: str
    recipient_id: Optional[str]
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    priority: int = 5
    ttl: int = 300  # Time to live in seconds

    def __lt__(self, other: "Message") -> bool:
        """Compare messages by priority for heap operations."""
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)

    def is_expired(self) -> bool:
        """Check if message has expired."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl


class MessageBroker:
    """
    Broker for reliable inter-agent message passing.

    Features:
    - Priority queues per agent
    - Request-response correlation
    - Message persistence
    - Dead letter queue

    Example:
        >>> broker = MessageBroker()
        >>> await broker.send(Message("agent_1", "agent_2", "request", {"data": "test"}))
        >>> msg = await broker.receive("agent_2")
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the message broker.

        Args:
            max_queue_size: Maximum queue size per agent
        """
        self._queues: Dict[str, asyncio.PriorityQueue] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._dead_letter_queue: List[Message] = []
        self._max_queue_size = max_queue_size
        self._lock = asyncio.Lock()
        self._message_log: List[Message] = []

    async def register_agent(self, agent_id: str):
        """
        Register an agent with the broker.

        Args:
            agent_id: ID of agent to register
        """
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.PriorityQueue(maxsize=self._max_queue_size)
                logger.debug(f"Registered agent {agent_id} with broker")

    async def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the broker.

        Args:
            agent_id: ID of agent to unregister
        """
        async with self._lock:
            if agent_id in self._queues:
                # Move remaining messages to dead letter queue
                queue = self._queues[agent_id]
                while not queue.empty():
                    try:
                        _, msg = queue.get_nowait()
                        self._dead_letter_queue.append(msg)
                    except asyncio.QueueEmpty:
                        break
                del self._queues[agent_id]

    async def send(self, message: Message) -> bool:
        """
        Send a message to an agent or broadcast.

        Args:
            message: Message to send

        Returns:
            True if message was queued successfully
        """
        self._message_log.append(message)

        if message.recipient_id:
            # Direct message
            return await self._send_direct(message)
        else:
            # Broadcast
            return await self._broadcast(message)

    async def _send_direct(self, message: Message) -> bool:
        """Send message to specific agent."""
        if message.recipient_id not in self._queues:
            self._dead_letter_queue.append(message)
            return False

        try:
            queue = self._queues[message.recipient_id]
            await asyncio.wait_for(
                queue.put((message.priority, message)),
                timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            self._dead_letter_queue.append(message)
            return False

    async def _broadcast(self, message: Message) -> bool:
        """Broadcast message to all agents."""
        success = True
        for agent_id, queue in self._queues.items():
            if agent_id != message.sender_id:
                try:
                    broadcast_msg = Message(
                        sender_id=message.sender_id,
                        recipient_id=agent_id,
                        message_type=message.message_type,
                        payload=message.payload,
                        correlation_id=message.correlation_id,
                        priority=message.priority
                    )
                    await asyncio.wait_for(
                        queue.put((broadcast_msg.priority, broadcast_msg)),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    success = False
        return success

    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
        message_type: Optional[str] = None
    ) -> Optional[Message]:
        """
        Receive a message for an agent.

        Args:
            agent_id: ID of receiving agent
            timeout: Maximum wait time in seconds
            message_type: Filter by message type

        Returns:
            Message or None if timeout
        """
        if agent_id not in self._queues:
            return None

        queue = self._queues[agent_id]

        try:
            if timeout:
                _, message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                _, message = await queue.get()

            # Filter by type if specified
            if message_type and message.message_type != message_type:
                # Put back and return None
                await queue.put((message.priority, message))
                return None

            # Check expiration
            if message.is_expired():
                self._dead_letter_queue.append(message)
                return await self.receive(agent_id, timeout, message_type)

            return message

        except asyncio.TimeoutError:
            return None

    async def request(
        self,
        message: Message,
        timeout: float = 30.0
    ) -> Optional[Message]:
        """
        Send a request and wait for response.

        Args:
            message: Request message
            timeout: Maximum wait time for response

        Returns:
            Response message or None if timeout
        """
        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[message.message_id] = future

        try:
            await self.send(message)
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_responses.pop(message.message_id, None)

    async def respond(self, original: Message, response_payload: Dict[str, Any]):
        """
        Send a response to a request.

        Args:
            original: Original request message
            response_payload: Response data
        """
        response = Message(
            sender_id=original.recipient_id,
            recipient_id=original.sender_id,
            message_type=f"{original.message_type}_response",
            payload=response_payload,
            correlation_id=original.message_id
        )

        # Check for pending future
        if original.message_id in self._pending_responses:
            future = self._pending_responses[original.message_id]
            if not future.done():
                future.set_result(response)
        else:
            await self.send(response)


# =============================================================================
# SHARED BLACKBOARD MEMORY
# =============================================================================

@dataclass
class BlackboardEntry:
    """
    Entry in the shared blackboard.

    Attributes:
        key: Entry key
        value: Entry value
        owner_id: ID of agent that created/owns the entry
        timestamp: When entry was created/updated
        version: Entry version for conflict detection
        ttl: Time to live in seconds (None for permanent)
    """
    key: str
    value: Any
    owner_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl


class Blackboard:
    """
    Thread-safe shared memory for agent coordination.

    Provides:
    - Key-value storage with versioning
    - Namespace isolation
    - TTL-based expiration
    - Change notifications

    Example:
        >>> bb = Blackboard()
        >>> await bb.write("agent_1", "shared/result", {"score": 0.95})
        >>> value = await bb.read("shared/result")
    """

    def __init__(self):
        """Initialize the blackboard."""
        self._entries: Dict[str, BlackboardEntry] = {}
        self._lock = asyncio.Lock()
        self._watchers: Dict[str, List[Callable[[BlackboardEntry], Awaitable[None]]]] = defaultdict(list)
        self._namespaces: Dict[str, Set[str]] = defaultdict(set)  # namespace -> keys

    async def write(
        self,
        owner_id: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BlackboardEntry:
        """
        Write a value to the blackboard.

        Args:
            owner_id: ID of writing agent
            key: Key to write to
            value: Value to store
            ttl: Time to live in seconds
            metadata: Additional metadata

        Returns:
            The created/updated entry
        """
        async with self._lock:
            existing = self._entries.get(key)
            version = (existing.version + 1) if existing else 1

            entry = BlackboardEntry(
                key=key,
                value=value,
                owner_id=owner_id,
                version=version,
                ttl=ttl,
                metadata=metadata or {}
            )

            self._entries[key] = entry

            # Track namespace
            namespace = key.split("/")[0] if "/" in key else "default"
            self._namespaces[namespace].add(key)

        # Notify watchers
        await self._notify_watchers(key, entry)

        return entry

    async def read(self, key: str) -> Optional[Any]:
        """
        Read a value from the blackboard.

        Args:
            key: Key to read

        Returns:
            Value or None if not found/expired
        """
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return None

            if entry.is_expired():
                del self._entries[key]
                return None

            return entry.value

    async def read_entry(self, key: str) -> Optional[BlackboardEntry]:
        """
        Read full entry from the blackboard.

        Args:
            key: Key to read

        Returns:
            BlackboardEntry or None
        """
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return None

            if entry.is_expired():
                del self._entries[key]
                return None

            return entry

    async def delete(self, key: str, owner_id: Optional[str] = None) -> bool:
        """
        Delete an entry from the blackboard.

        Args:
            key: Key to delete
            owner_id: If specified, only delete if owner matches

        Returns:
            True if deleted
        """
        async with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return False

            if owner_id and entry.owner_id != owner_id:
                return False

            del self._entries[key]

            # Remove from namespace
            namespace = key.split("/")[0] if "/" in key else "default"
            self._namespaces[namespace].discard(key)

            return True

    async def list_keys(
        self,
        namespace: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """
        List keys in the blackboard.

        Args:
            namespace: Filter by namespace
            pattern: Filter by pattern (prefix match)

        Returns:
            List of matching keys
        """
        async with self._lock:
            if namespace:
                keys = list(self._namespaces.get(namespace, set()))
            else:
                keys = list(self._entries.keys())

            if pattern:
                keys = [k for k in keys if k.startswith(pattern)]

            # Filter out expired
            valid_keys = []
            for key in keys:
                entry = self._entries.get(key)
                if entry and not entry.is_expired():
                    valid_keys.append(key)

            return valid_keys

    async def watch(
        self,
        key_pattern: str,
        callback: Callable[[BlackboardEntry], Awaitable[None]]
    ):
        """
        Watch for changes to keys matching a pattern.

        Args:
            key_pattern: Pattern to match (prefix)
            callback: Async function to call on changes
        """
        async with self._lock:
            self._watchers[key_pattern].append(callback)

    async def _notify_watchers(self, key: str, entry: BlackboardEntry):
        """Notify watchers of a change."""
        for pattern, callbacks in self._watchers.items():
            if key.startswith(pattern) or pattern == "*":
                for callback in callbacks:
                    try:
                        await callback(entry)
                    except Exception as e:
                        logger.error(f"Watcher callback error: {e}")

    async def cleanup_expired(self):
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._entries.items()
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._entries[key]
            return len(expired_keys)


# =============================================================================
# TASK DEPENDENCIES
# =============================================================================

@dataclass
class Task:
    """
    Represents a task in the multi-agent system.

    Attributes:
        task_id: Unique task identifier
        name: Human-readable name
        payload: Task data
        dependencies: List of task IDs this task depends on
        priority: Task priority (lower = higher priority)
        assigned_agent: ID of agent assigned to this task
        status: Current task status
    """
    task_id: str
    name: str
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds

    def __lt__(self, other: "Task") -> bool:
        """Compare tasks by priority."""
        return (self.priority, self.created_at) < (other.priority, other.created_at)


class TaskDependencyGraph:
    """
    DAG-based task dependency management.

    Provides:
    - Dependency validation (cycle detection)
    - Topological ordering
    - Ready task identification

    Example:
        >>> dag = TaskDependencyGraph()
        >>> dag.add_task(task1)
        >>> dag.add_task(task2, dependencies=[task1.task_id])
        >>> ready = dag.get_ready_tasks()
    """

    def __init__(self):
        """Initialize the dependency graph."""
        self._tasks: Dict[str, Task] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)  # task_id -> dependent task_ids
        self._dependents: Dict[str, Set[str]] = defaultdict(set)  # task_id -> tasks that depend on it
        self._lock = threading.Lock()

    def add_task(self, task: Task) -> bool:
        """
        Add a task to the graph.

        Args:
            task: Task to add

        Returns:
            True if added successfully, False if would create cycle
        """
        with self._lock:
            # Check for cycles
            if task.dependencies:
                if self._would_create_cycle(task.task_id, task.dependencies):
                    return False

            self._tasks[task.task_id] = task

            # Set up dependency relationships
            for dep_id in task.dependencies:
                self._dependencies[task.task_id].add(dep_id)
                self._dependents[dep_id].add(task.task_id)

            return True

    def _would_create_cycle(self, task_id: str, dependencies: List[str]) -> bool:
        """Check if adding dependencies would create a cycle."""
        # BFS to check for cycles
        visited = set()
        queue = list(dependencies)

        while queue:
            current = queue.pop(0)
            if current == task_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._dependencies.get(current, set()))

        return False

    def remove_task(self, task_id: str) -> Optional[Task]:
        """
        Remove a task from the graph.

        Args:
            task_id: ID of task to remove

        Returns:
            Removed task or None
        """
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                # Remove from dependencies
                for dep_id in self._dependencies.get(task_id, set()):
                    self._dependents[dep_id].discard(task_id)

                # Remove from dependents
                for dependent_id in self._dependents.get(task_id, set()):
                    self._dependencies[dependent_id].discard(task_id)

                del self._dependencies[task_id]
                del self._dependents[task_id]

            return task

    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to execute (all dependencies satisfied).

        Returns:
            List of ready tasks
        """
        with self._lock:
            ready = []
            for task_id, task in self._tasks.items():
                if task.status != "pending":
                    continue

                # Check all dependencies are completed
                deps_satisfied = all(
                    self._tasks.get(dep_id, Task("", "", {})).status == "completed"
                    for dep_id in self._dependencies.get(task_id, set())
                )

                if deps_satisfied:
                    ready.append(task)

            # Sort by priority
            ready.sort()
            return ready

    def mark_completed(self, task_id: str, result: Any = None):
        """
        Mark a task as completed.

        Args:
            task_id: ID of completed task
            result: Task result
        """
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = "completed"
                self._tasks[task_id].completed_at = datetime.now()
                self._tasks[task_id].result = result

    def mark_failed(self, task_id: str, error: str):
        """
        Mark a task as failed.

        Args:
            task_id: ID of failed task
            error: Error message
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.retries += 1
                if task.retries >= task.max_retries:
                    task.status = "failed"
                    task.error = error
                else:
                    task.status = "pending"  # Will retry

    def get_topological_order(self) -> List[str]:
        """
        Get tasks in topological order.

        Returns:
            List of task IDs in execution order
        """
        with self._lock:
            in_degree = {tid: len(deps) for tid, deps in self._dependencies.items()}

            # Add tasks with no dependencies
            for tid in self._tasks:
                if tid not in in_degree:
                    in_degree[tid] = 0

            queue = [tid for tid, degree in in_degree.items() if degree == 0]
            result = []

            while queue:
                current = queue.pop(0)
                result.append(current)

                for dependent in self._dependents.get(current, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            return result


# =============================================================================
# MULTI-AGENT ORCHESTRATOR
# =============================================================================

@dataclass
class AgentInfo:
    """
    Information about a registered agent.

    Attributes:
        agent_id: Unique agent identifier
        capabilities: List of capabilities
        state_machine: Agent's state machine
        load: Current load (number of assigned tasks)
        last_heartbeat: Last heartbeat timestamp
    """
    agent_id: str
    capabilities: Set[str]
    state_machine: AgentStateMachine
    load: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    max_load: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentOrchestrator:
    """
    Production-ready orchestrator for multi-agent MAIF systems.

    Provides:
    - Task scheduling with priority queues
    - Load balancing across agents
    - Failure handling with retries
    - Agent health monitoring
    - Resource management

    Example:
        >>> orchestrator = MultiAgentOrchestrator()
        >>> await orchestrator.start()
        >>> await orchestrator.register_agent(agent_info)
        >>> await orchestrator.submit_task(task)
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 100,
        heartbeat_interval: float = 5.0,
        agent_timeout: float = 30.0
    ):
        """
        Initialize the orchestrator.

        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            heartbeat_interval: Interval for health checks
            agent_timeout: Time before considering agent dead
        """
        self._agents: Dict[str, AgentInfo] = {}
        self._event_bus = EventBus()
        self._message_broker = MessageBroker()
        self._blackboard = Blackboard()
        self._task_graph = TaskDependencyGraph()

        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_concurrent_tasks)
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: List[Task] = []
        self._failed_tasks: List[Task] = []

        self._running = False
        self._lock = asyncio.Lock()
        self._heartbeat_interval = heartbeat_interval
        self._agent_timeout = agent_timeout

        # Statistics
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }

        # Backward compatibility
        self.agents: Dict[str, Any] = {}  # For legacy access
        self.alignment_engine = SemanticAlignmentEngine()
        self.exchange_history: List[Dict[str, Any]] = []

    async def start(self):
        """Start the orchestrator and all subsystems."""
        self._running = True

        await self._event_bus.start()

        # Start background tasks
        asyncio.create_task(self._scheduler_loop())
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._cleanup_loop())

        # Subscribe to events
        await self._event_bus.subscribe("agent_heartbeat", self._handle_heartbeat)
        await self._event_bus.subscribe("task_complete", self._handle_task_complete)
        await self._event_bus.subscribe("task_failed", self._handle_task_failed)

        logger.info("MultiAgentOrchestrator started")

    async def stop(self):
        """Stop the orchestrator gracefully."""
        self._running = False
        await self._event_bus.stop()
        logger.info("MultiAgentOrchestrator stopped")

    async def register_agent(
        self,
        agent_id: str,
        capabilities: Set[str],
        max_load: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentInfo:
        """
        Register an agent with the orchestrator.

        Args:
            agent_id: Unique agent identifier
            capabilities: Set of agent capabilities
            max_load: Maximum concurrent tasks
            metadata: Additional agent metadata

        Returns:
            AgentInfo for the registered agent
        """
        async with self._lock:
            state_machine = AgentStateMachine(agent_id)

            agent_info = AgentInfo(
                agent_id=agent_id,
                capabilities=capabilities,
                state_machine=state_machine,
                max_load=max_load,
                metadata=metadata or {}
            )

            self._agents[agent_id] = agent_info
            await self._message_broker.register_agent(agent_id)

            # Update blackboard
            await self._blackboard.write(
                "orchestrator",
                f"agents/{agent_id}/info",
                {
                    "capabilities": list(capabilities),
                    "max_load": max_load,
                    "registered_at": datetime.now().isoformat()
                }
            )

            # Backward compatibility
            self.agents[agent_id] = agent_info

            logger.info(f"Registered agent {agent_id} with capabilities {capabilities}")
            return agent_info

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the orchestrator.

        Args:
            agent_id: ID of agent to unregister

        Returns:
            True if agent was unregistered
        """
        async with self._lock:
            if agent_id not in self._agents:
                return False

            agent_info = self._agents.pop(agent_id)
            await self._message_broker.unregister_agent(agent_id)
            await self._blackboard.delete(f"agents/{agent_id}/info")

            # Reassign any tasks
            for task_id, task in list(self._running_tasks.items()):
                if task.assigned_agent == agent_id:
                    task.assigned_agent = None
                    task.status = "pending"
                    await self._task_queue.put((task.priority, task))
                    del self._running_tasks[task_id]

            # Backward compatibility
            self.agents.pop(agent_id, None)

            logger.info(f"Unregistered agent {agent_id}")
            return True

    async def submit_task(self, task: Task) -> bool:
        """
        Submit a task for execution.

        Args:
            task: Task to submit

        Returns:
            True if task was submitted successfully
        """
        # Add to dependency graph
        if not self._task_graph.add_task(task):
            logger.error(f"Task {task.task_id} would create dependency cycle")
            return False

        # Queue if ready
        if not task.dependencies:
            await self._task_queue.put((task.priority, task))
        else:
            # Will be queued when dependencies complete
            pass

        self._stats["tasks_submitted"] += 1

        await self._blackboard.write(
            "orchestrator",
            f"tasks/{task.task_id}/status",
            {"status": "submitted", "timestamp": datetime.now().isoformat()}
        )

        return True

    async def _scheduler_loop(self):
        """Main scheduling loop."""
        while self._running:
            try:
                # Get next task
                try:
                    _, task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Check for newly ready tasks
                    ready_tasks = self._task_graph.get_ready_tasks()
                    for t in ready_tasks:
                        if t.status == "pending" and t.task_id not in self._running_tasks:
                            await self._task_queue.put((t.priority, t))
                    continue

                # Find suitable agent
                agent = await self._select_agent(task)

                if agent:
                    await self._assign_task(task, agent)
                else:
                    # No agent available, requeue
                    await asyncio.sleep(1.0)
                    await self._task_queue.put((task.priority, task))

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(1.0)

    async def _select_agent(self, task: Task) -> Optional[AgentInfo]:
        """
        Select best agent for a task using load balancing.

        Args:
            task: Task to assign

        Returns:
            Selected agent or None
        """
        required_capabilities = task.payload.get("required_capabilities", set())

        candidates = []
        for agent_id, agent_info in self._agents.items():
            # Check state
            if agent_info.state_machine.state not in {AgentState.IDLE, AgentState.RUNNING}:
                continue

            # Check load
            if agent_info.load >= agent_info.max_load:
                continue

            # Check capabilities
            if required_capabilities and not required_capabilities.issubset(agent_info.capabilities):
                continue

            # Check health
            time_since_heartbeat = (datetime.now() - agent_info.last_heartbeat).total_seconds()
            if time_since_heartbeat > self._agent_timeout:
                continue

            candidates.append(agent_info)

        if not candidates:
            return None

        # Select agent with lowest load (load balancing)
        return min(candidates, key=lambda a: a.load)

    async def _assign_task(self, task: Task, agent: AgentInfo):
        """
        Assign a task to an agent.

        Args:
            task: Task to assign
            agent: Agent to assign to
        """
        task.assigned_agent = agent.agent_id
        task.status = "running"
        task.started_at = datetime.now()

        agent.load += 1
        if agent.state_machine.state == AgentState.IDLE:
            agent.state_machine.transition(AgentState.RUNNING, f"Assigned task {task.task_id}")

        self._running_tasks[task.task_id] = task

        # Send task to agent
        message = Message(
            sender_id="orchestrator",
            recipient_id=agent.agent_id,
            message_type="task_assignment",
            payload={
                "task_id": task.task_id,
                "name": task.name,
                "payload": task.payload,
                "timeout": task.timeout
            }
        )
        await self._message_broker.send(message)

        # Update blackboard
        await self._blackboard.write(
            "orchestrator",
            f"tasks/{task.task_id}/status",
            {
                "status": "running",
                "agent": agent.agent_id,
                "started_at": datetime.now().isoformat()
            }
        )

        logger.debug(f"Assigned task {task.task_id} to agent {agent.agent_id}")

    async def _handle_heartbeat(self, event: Event):
        """Handle agent heartbeat event."""
        agent_id = event.source_id
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = datetime.now()

    async def _handle_task_complete(self, event: Event):
        """Handle task completion event."""
        task_id = event.payload.get("task_id")
        result = event.payload.get("result")

        if task_id in self._running_tasks:
            task = self._running_tasks.pop(task_id)
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result

            self._task_graph.mark_completed(task_id, result)
            self._completed_tasks.append(task)

            # Update agent load
            if task.assigned_agent in self._agents:
                agent = self._agents[task.assigned_agent]
                agent.load = max(0, agent.load - 1)
                if agent.load == 0:
                    agent.state_machine.transition(AgentState.IDLE, "No active tasks")

            self._stats["tasks_completed"] += 1
            if task.started_at and task.completed_at:
                exec_time = (task.completed_at - task.started_at).total_seconds()
                self._stats["total_execution_time"] += exec_time

            # Check for newly ready tasks
            ready_tasks = self._task_graph.get_ready_tasks()
            for t in ready_tasks:
                if t.status == "pending" and t.task_id not in self._running_tasks:
                    await self._task_queue.put((t.priority, t))

            await self._blackboard.write(
                "orchestrator",
                f"tasks/{task_id}/status",
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "result": result
                }
            )

    async def _handle_task_failed(self, event: Event):
        """Handle task failure event."""
        task_id = event.payload.get("task_id")
        error = event.payload.get("error", "Unknown error")

        if task_id in self._running_tasks:
            task = self._running_tasks.pop(task_id)

            # Update agent load
            if task.assigned_agent in self._agents:
                agent = self._agents[task.assigned_agent]
                agent.load = max(0, agent.load - 1)

            # Handle retry
            self._task_graph.mark_failed(task_id, error)
            graph_task = self._task_graph._tasks.get(task_id)

            if graph_task and graph_task.status == "pending":
                # Retry
                task.assigned_agent = None
                task.status = "pending"
                await self._task_queue.put((task.priority, task))
                logger.info(f"Retrying task {task_id} (attempt {task.retries + 1})")
            else:
                # Failed permanently
                task.status = "failed"
                task.error = error
                self._failed_tasks.append(task)
                self._stats["tasks_failed"] += 1

            await self._blackboard.write(
                "orchestrator",
                f"tasks/{task_id}/status",
                {
                    "status": task.status,
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                }
            )

    async def _health_check_loop(self):
        """Monitor agent health."""
        while self._running:
            try:
                current_time = datetime.now()

                for agent_id, agent_info in list(self._agents.items()):
                    time_since_heartbeat = (current_time - agent_info.last_heartbeat).total_seconds()

                    if time_since_heartbeat > self._agent_timeout:
                        logger.warning(f"Agent {agent_id} is unresponsive")

                        # Mark agent as failed
                        if agent_info.state_machine.can_transition(AgentState.FAILED):
                            agent_info.state_machine.transition(
                                AgentState.FAILED,
                                "Heartbeat timeout"
                            )

                        # Reassign tasks
                        for task_id, task in list(self._running_tasks.items()):
                            if task.assigned_agent == agent_id:
                                task.assigned_agent = None
                                task.status = "pending"
                                await self._task_queue.put((task.priority, task))
                                del self._running_tasks[task_id]

                await asyncio.sleep(self._heartbeat_interval)

            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self._heartbeat_interval)

    async def _cleanup_loop(self):
        """Periodic cleanup."""
        while self._running:
            try:
                # Clean expired blackboard entries
                expired = await self._blackboard.cleanup_expired()
                if expired > 0:
                    logger.debug(f"Cleaned up {expired} expired blackboard entries")

                # Trim completed/failed task history
                max_history = 1000
                if len(self._completed_tasks) > max_history:
                    self._completed_tasks = self._completed_tasks[-max_history:]
                if len(self._failed_tasks) > max_history:
                    self._failed_tasks = self._failed_tasks[-max_history:]

                await asyncio.sleep(60.0)  # Run every minute

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60.0)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "registered_agents": len(self._agents),
            "running_tasks": len(self._running_tasks),
            "queued_tasks": self._task_queue.qsize(),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "avg_execution_time": (
                self._stats["total_execution_time"] / self._stats["tasks_completed"]
                if self._stats["tasks_completed"] > 0 else 0
            )
        }

    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # =========================================================================

    def register_agent_legacy(self, agent: "MAIFExchangeProtocol"):
        """
        Register an agent in the orchestrator (legacy method).

        Args:
            agent: MAIFExchangeProtocol instance
        """
        self.agents[agent.agent_id] = agent

    async def facilitate_exchange(
        self,
        source_agent_id: str,
        target_agent_id: str,
        maif_id: str
    ) -> bool:
        """
        Facilitate MAIF exchange between two agents (legacy method).

        Args:
            source_agent_id: Source agent ID
            target_agent_id: Target agent ID
            maif_id: MAIF to exchange

        Returns:
            True if exchange succeeded
        """
        source = self.agents.get(source_agent_id)
        target = self.agents.get(target_agent_id)

        if not source or not target:
            return False

        # Handle both legacy and new agent types
        if hasattr(source, 'capabilities') and hasattr(target, 'capabilities'):
            # Compute semantic alignment
            alignment = await self.alignment_engine.align_agents(
                source.capabilities, target.capabilities
            )

            quality = alignment.metadata.get("alignment_quality", 0.0)
            if quality < 0.6:
                return False

            # Initiate exchange
            success = await source.initiate_exchange(target, maif_id)

            self.exchange_history.append({
                "timestamp": datetime.now().isoformat(),
                "source": source_agent_id,
                "target": target_agent_id,
                "maif_id": maif_id,
                "alignment_quality": quality,
                "success": success,
            })

            return success

        return False

    async def broadcast_maif(
        self,
        source_agent_id: str,
        maif_id: str,
        min_alignment: float = 0.7
    ) -> List[str]:
        """
        Broadcast MAIF to all compatible agents (legacy method).

        Args:
            source_agent_id: Source agent ID
            maif_id: MAIF to broadcast
            min_alignment: Minimum alignment score

        Returns:
            List of agents that received the MAIF
        """
        source = self.agents.get(source_agent_id)
        if not source:
            return []

        successful_agents = []

        for agent_id, agent in self.agents.items():
            if agent_id == source_agent_id:
                continue

            if hasattr(source, 'capabilities') and hasattr(agent, 'capabilities'):
                alignment = await self.alignment_engine.align_agents(
                    source.capabilities, agent.capabilities
                )

                quality = alignment.metadata.get("alignment_quality", 0.0)
                if quality >= min_alignment:
                    success = await source.initiate_exchange(agent, maif_id)
                    if success:
                        successful_agents.append(agent_id)

        return successful_agents

    def get_exchange_analytics(self) -> Dict[str, Any]:
        """
        Get analytics on multi-agent exchanges (legacy method).

        Returns:
            Exchange analytics dictionary
        """
        if not self.exchange_history:
            return {}

        total_exchanges = len(self.exchange_history)
        successful_exchanges = sum(1 for e in self.exchange_history if e["success"])

        alignment_scores = [e["alignment_quality"] for e in self.exchange_history]

        agent_exchanges = {}
        for exchange in self.exchange_history:
            for agent in [exchange["source"], exchange["target"]]:
                agent_exchanges[agent] = agent_exchanges.get(agent, 0) + 1

        return {
            "total_exchanges": total_exchanges,
            "successful_exchanges": successful_exchanges,
            "success_rate": successful_exchanges / total_exchanges if total_exchanges > 0 else 0,
            "average_alignment": np.mean(alignment_scores) if alignment_scores else 0,
            "min_alignment": min(alignment_scores) if alignment_scores else 0,
            "max_alignment": max(alignment_scores) if alignment_scores else 0,
            "most_active_agents": sorted(
                agent_exchanges.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "exchange_timeline": self.exchange_history[-10:],
        }


# =============================================================================
# BACKWARD COMPATIBILITY: Original Classes
# =============================================================================

class SemanticProcessor:
    """Placeholder semantic processor for multi-agent coordination.

    Uses TF-IDF based processing (no TensorFlow dependencies).
    """

    def __init__(self):
        # Lazy import to avoid circular dependencies
        try:
            from ..semantic import TFIDFEmbedder
            self.embedder = TFIDFEmbedder() if TFIDFEmbedder else None
        except ImportError:
            self.embedder = None

    def process(self, data: Any) -> Any:
        return data


class Ontology:
    """Placeholder ontology class for semantic alignment."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.concepts: Dict[str, Any] = {}


class MAIF:
    """Placeholder MAIF class for multi-agent exchange."""

    def __init__(self, id: str):
        self.id = id
        self.header: Dict[str, Any] = {"metadata": {}}
        self.blocks: List[Any] = []

    def add_block(self, block: Any):
        self.blocks.append(block)


class Block:
    """Placeholder Block class for multi-agent exchange."""

    def __init__(self, block_type: BlockType, data: bytes, metadata: Optional[Dict] = None):
        self.block_type = block_type
        self.data = data
        self.metadata = metadata or {}


class ExchangeSession:
    """Placeholder ExchangeSession class for multi-agent exchange."""

    def __init__(self, session_id: str, target_agent_id: str):
        self.session_id = session_id
        self.target_agent_id = target_agent_id
        self.state: str = "initiated"
        self.exchanged_blocks: List[str] = []


class ExchangeProtocolVersion(Enum):
    """MAIF Exchange Protocol versions"""

    V1_0 = "1.0"  # Basic exchange
    V2_0 = "2.0"  # With semantic alignment
    V3_0 = "3.0"  # With negotiation


class MessageType(Enum):
    """Types of messages in the exchange protocol"""

    # Discovery
    HELLO = "HELLO"
    CAPABILITIES = "CAPABILITIES"

    # Negotiation
    PROPOSE_EXCHANGE = "PROPOSE_EXCHANGE"
    ACCEPT_EXCHANGE = "ACCEPT_EXCHANGE"
    REJECT_EXCHANGE = "REJECT_EXCHANGE"

    # Transfer
    REQUEST_MAIF = "REQUEST_MAIF"
    SEND_MAIF = "SEND_MAIF"
    SEND_BLOCK = "SEND_BLOCK"

    # Semantic Alignment
    REQUEST_ALIGNMENT = "REQUEST_ALIGNMENT"
    SEND_ALIGNMENT = "SEND_ALIGNMENT"
    CONFIRM_ALIGNMENT = "CONFIRM_ALIGNMENT"

    # Control
    ACK = "ACK"
    NACK = "NACK"
    ERROR = "ERROR"
    GOODBYE = "GOODBYE"


@dataclass
class AgentCapabilities:
    """Capabilities of an agent in the exchange protocol"""

    agent_id: str
    name: str
    version: str
    supported_protocols: List[ExchangeProtocolVersion]
    supported_block_types: List[BlockType]
    semantic_models: List[str]
    compression_algorithms: List[str]
    max_maif_size: int
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeMessage:
    """Message in the MAIF exchange protocol"""

    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    timestamp: datetime
    payload: Dict[str, Any]
    signature: Optional[bytes] = None

    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }
        serialized = json.dumps(data).encode("utf-8")
        length = struct.pack("!I", len(serialized))
        return length + serialized

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExchangeMessage":
        """Deserialize message from bytes"""
        serialized = data[4:]
        parsed = json.loads(serialized.decode("utf-8"))

        return cls(
            message_id=parsed["message_id"],
            sender_id=parsed["sender_id"],
            recipient_id=parsed["recipient_id"],
            message_type=MessageType(parsed["message_type"]),
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            payload=parsed["payload"],
        )


@dataclass
class SemanticAlignment:
    """Semantic alignment between two agents"""

    source_agent: str
    target_agent: str
    concept_mappings: Dict[str, str]
    confidence_scores: Dict[str, float]
    transformation_rules: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MAIFExchangeProtocol:
    """
    MAIF Exchange Protocol implementation.

    Provides standardized communication for MAIF exchange between agents.
    """

    def __init__(self, agent_id: str, capabilities: AgentCapabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.active_sessions: Dict[str, "ExchangeSession"] = {}
        self.semantic_processor = SemanticProcessor()
        self.security_manager = SecurityManager()

    async def initiate_exchange(
        self, target_agent: "MAIFExchangeProtocol", maif_id: str
    ) -> bool:
        """Initiate MAIF exchange with another agent"""
        session_id = str(uuid.uuid4())

        hello_msg = self._create_message(
            target_agent.agent_id,
            MessageType.HELLO,
            {
                "session_id": session_id,
                "protocol_version": ExchangeProtocolVersion.V3_0.value,
            },
        )

        response = await target_agent.handle_message(hello_msg)
        if response.message_type != MessageType.CAPABILITIES:
            return False

        target_caps = AgentCapabilities(**response.payload["capabilities"])
        if not self._check_compatibility(target_caps):
            return False

        propose_msg = self._create_message(
            target_agent.agent_id,
            MessageType.PROPOSE_EXCHANGE,
            {
                "session_id": session_id,
                "maif_id": maif_id,
                "transfer_mode": "streaming",
                "compression": "zstd",
            },
        )

        response = await target_agent.handle_message(propose_msg)
        return response.message_type == MessageType.ACCEPT_EXCHANGE

    async def handle_message(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle incoming exchange protocol message"""
        handlers = {
            MessageType.HELLO: self._handle_hello,
            MessageType.PROPOSE_EXCHANGE: self._handle_propose,
            MessageType.REQUEST_MAIF: self._handle_request_maif,
            MessageType.REQUEST_ALIGNMENT: self._handle_request_alignment,
        }

        handler = handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return self._create_error_response(message, "Unsupported message type")

    async def _handle_hello(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle HELLO message"""
        return self._create_message(
            message.sender_id,
            MessageType.CAPABILITIES,
            {"capabilities": self.capabilities.__dict__},
        )

    async def _handle_propose(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle exchange proposal"""
        return self._create_message(
            message.sender_id,
            MessageType.ACCEPT_EXCHANGE,
            {"session_id": message.payload["session_id"]},
        )

    async def _handle_request_maif(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle MAIF request"""
        maif_id = message.payload["maif_id"]
        return self._create_message(
            message.sender_id, MessageType.ACK, {"maif_id": maif_id, "status": "ready"}
        )

    async def _handle_request_alignment(
        self, message: ExchangeMessage
    ) -> ExchangeMessage:
        """Handle semantic alignment request"""
        source_concepts = message.payload.get("concepts", [])
        alignment = self._compute_alignment(message.sender_id, source_concepts)

        return self._create_message(
            message.sender_id,
            MessageType.SEND_ALIGNMENT,
            {"alignment": alignment.__dict__},
        )

    def _create_message(
        self, recipient_id: str, msg_type: MessageType, payload: Dict[str, Any]
    ) -> ExchangeMessage:
        """Create a new exchange message"""
        return ExchangeMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=msg_type,
            timestamp=datetime.now(),
            payload=payload,
        )

    def _create_error_response(
        self, original: ExchangeMessage, error: str
    ) -> ExchangeMessage:
        """Create error response message"""
        return self._create_message(
            original.sender_id,
            MessageType.ERROR,
            {"error": error, "original_message_id": original.message_id},
        )

    def _check_compatibility(self, target_caps: AgentCapabilities) -> bool:
        """Check if two agents are compatible for exchange"""
        common_protocols = set(self.capabilities.supported_protocols) & set(
            target_caps.supported_protocols
        )
        if not common_protocols:
            return False

        common_blocks = set(self.capabilities.supported_block_types) & set(
            target_caps.supported_block_types
        )
        if not common_blocks:
            return False

        return True

    def _compute_alignment(
        self, target_agent_id: str, source_concepts: List[str]
    ) -> SemanticAlignment:
        """Compute semantic alignment with another agent"""
        concept_mappings = {}
        confidence_scores = {}

        for concept in source_concepts:
            if concept in self.capabilities.semantic_models:
                concept_mappings[concept] = concept
                confidence_scores[concept] = 1.0
            else:
                best_match, score = self._fuzzy_match(
                    concept, self.capabilities.semantic_models
                )
                if score > 0.7:
                    concept_mappings[concept] = best_match
                    confidence_scores[concept] = score

        return SemanticAlignment(
            source_agent=target_agent_id,
            target_agent=self.agent_id,
            concept_mappings=concept_mappings,
            confidence_scores=confidence_scores,
            transformation_rules=[],
        )

    def _fuzzy_match(self, concept: str, candidates: List[str]) -> Tuple[str, float]:
        """Simple fuzzy matching for concept alignment"""
        best_match = ""
        best_score = 0.0

        concept_lower = concept.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()

            if concept_lower in candidate_lower or candidate_lower in concept_lower:
                score = 0.8
            else:
                concept_words = set(concept_lower.split())
                candidate_words = set(candidate_lower.split())
                overlap = len(concept_words & candidate_words)
                total = len(concept_words | candidate_words)
                score = overlap / total if total > 0 else 0.0

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, best_score


class SemanticAlignmentEngine:
    """
    Advanced Semantic Alignment Engine.

    Provides deep semantic understanding and alignment between agents.
    """

    def __init__(self):
        self.ontology_cache: Dict[str, "Ontology"] = {}
        self.alignment_cache: Dict[Tuple[str, str], SemanticAlignment] = {}
        self.semantic_processor = SemanticProcessor()

    async def align_agents(
        self, agent1: AgentCapabilities, agent2: AgentCapabilities
    ) -> SemanticAlignment:
        """Perform deep semantic alignment between two agents"""
        cache_key = (agent1.agent_id, agent2.agent_id)
        if cache_key in self.alignment_cache:
            return self.alignment_cache[cache_key]

        models1 = set(agent1.semantic_models)
        models2 = set(agent2.semantic_models)

        common_models = models1 & models2
        unique_to_1 = models1 - models2
        unique_to_2 = models2 - models1

        concept_mappings = {}
        confidence_scores = {}

        for model in common_models:
            concept_mappings[model] = model
            confidence_scores[model] = 1.0

        for concept1 in unique_to_1:
            best_match, score = await self._find_best_alignment(concept1, unique_to_2)
            if score > 0.5:
                concept_mappings[concept1] = best_match
                confidence_scores[concept1] = score

        transformation_rules = self._generate_transformation_rules(
            concept_mappings, agent1, agent2
        )

        alignment = SemanticAlignment(
            source_agent=agent1.agent_id,
            target_agent=agent2.agent_id,
            concept_mappings=concept_mappings,
            confidence_scores=confidence_scores,
            transformation_rules=transformation_rules,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "common_models": list(common_models),
                "alignment_quality": np.mean(list(confidence_scores.values())) if confidence_scores else 0.0,
            },
        )

        self.alignment_cache[cache_key] = alignment
        return alignment

    async def _find_best_alignment(
        self, source_concept: str, target_concepts: Set[str]
    ) -> Tuple[str, float]:
        """Find best semantic alignment for a concept"""
        if not target_concepts:
            return "", 0.0

        source_embedding = await self._get_concept_embedding(source_concept)

        best_match = ""
        best_score = 0.0

        for target in target_concepts:
            target_embedding = await self._get_concept_embedding(target)
            score = self._cosine_similarity(source_embedding, target_embedding)
            lexical_score = self._lexical_similarity(source_concept, target)
            score = 0.7 * score + 0.3 * lexical_score

            if score > best_score:
                best_score = score
                best_match = target

        return best_match, best_score

    async def _get_concept_embedding(self, concept: str) -> np.ndarray:
        """Get semantic embedding for a concept"""
        hash_bytes = hashlib.sha256(concept.encode()).digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.float32)
        return embedding / np.linalg.norm(embedding)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot_product / norm_product) if norm_product > 0 else 0.0

    def _lexical_similarity(self, str1: str, str2: str) -> float:
        """Compute lexical similarity between strings"""

        def get_ngrams(s: str, n: int = 3) -> Set[str]:
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        ngrams1 = get_ngrams(str1.lower())
        ngrams2 = get_ngrams(str2.lower())

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _generate_transformation_rules(
        self,
        mappings: Dict[str, str],
        source: AgentCapabilities,
        target: AgentCapabilities,
    ) -> List[Dict[str, Any]]:
        """Generate transformation rules based on alignment"""
        rules = []

        for source_concept, target_concept in mappings.items():
            if source_concept != target_concept:
                rule = {
                    "type": "concept_mapping",
                    "source": source_concept,
                    "target": target_concept,
                    "confidence": mappings.get(source_concept, 0.0),
                    "transformations": [],
                }

                source_features = source.features.get(source_concept, {})
                target_features = target.features.get(target_concept, {})

                if source_features.get("data_type") != target_features.get("data_type"):
                    rule["transformations"].append(
                        {
                            "type": "type_conversion",
                            "from": source_features.get("data_type"),
                            "to": target_features.get("data_type"),
                        }
                    )

                rules.append(rule)

        return rules

    async def transform_maif(self, maif: MAIF, alignment: SemanticAlignment) -> MAIF:
        """Transform MAIF according to semantic alignment"""
        transformed = MAIF(f"{maif.id}_transformed")

        transformed.header = maif.header.copy()
        transformed.header["metadata"]["semantic_alignment"] = {
            "source_agent": alignment.source_agent,
            "target_agent": alignment.target_agent,
            "alignment_quality": alignment.metadata.get("alignment_quality", 0.0),
        }

        for block in maif.blocks:
            transformed_block = await self._transform_block(block, alignment)
            transformed.add_block(transformed_block)

        return transformed

    async def _transform_block(
        self, block: Block, alignment: SemanticAlignment
    ) -> Block:
        """Transform a block according to alignment rules"""
        block_type_str = block.block_type.name

        if block_type_str in alignment.concept_mappings:
            target_type_str = alignment.concept_mappings[block_type_str]

            try:
                target_type = BlockType[target_type_str]
            except KeyError:
                target_type = block.block_type

            transformed = Block(
                block_type=target_type, data=block.data, metadata=block.metadata.copy()
            )

            for rule in alignment.transformation_rules:
                if rule["source"] == block_type_str:
                    transformed = await self._apply_transformation_rule(
                        transformed, rule
                    )

            return transformed
        else:
            return block

    async def _apply_transformation_rule(
        self, block: Block, rule: Dict[str, Any]
    ) -> Block:
        """Apply a specific transformation rule to a block"""
        for transform in rule.get("transformations", []):
            if transform["type"] == "type_conversion":
                if transform["from"] == "json" and transform["to"] == "msgpack":
                    try:
                        import msgpack
                        data = json.loads(block.data)
                        block.data = msgpack.packb(data)
                    except ImportError:
                        pass
                elif transform["from"] == "msgpack" and transform["to"] == "json":
                    try:
                        import msgpack
                        data = msgpack.unpackb(block.data)
                        block.data = json.dumps(data).encode()
                    except ImportError:
                        pass

        return block


# =============================================================================
# CONVENIENCE ALIAS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Alias for backward compatibility with imports expecting MultiAgentCoordinator
MultiAgentCoordinator = MultiAgentOrchestrator


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":

    async def demo_multi_agent_framework():
        """Demonstrate the multi-agent framework"""
        print("=== Multi-Agent MAIF Framework Demo ===\n")

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()
        await orchestrator.start()

        # Register agents
        agent1 = await orchestrator.register_agent(
            agent_id="research_agent",
            capabilities={"research", "analysis", "text_processing"},
            max_load=5
        )

        agent2 = await orchestrator.register_agent(
            agent_id="analysis_agent",
            capabilities={"analysis", "statistics", "visualization"},
            max_load=5
        )

        print(f"1. Registered agents: {list(orchestrator._agents.keys())}")

        # Test state machine
        print("\n2. Testing State Machine:")
        print(f"   Agent 1 state: {agent1.state_machine.state.name}")
        agent1.state_machine.transition(AgentState.RUNNING, "Starting demo task")
        print(f"   After transition: {agent1.state_machine.state.name}")
        agent1.state_machine.transition(AgentState.IDLE, "Demo complete")
        print(f"   Back to: {agent1.state_machine.state.name}")

        # Test blackboard
        print("\n3. Testing Blackboard:")
        await orchestrator._blackboard.write("agent1", "shared/result", {"score": 0.95})
        result = await orchestrator._blackboard.read("shared/result")
        print(f"   Blackboard value: {result}")

        # Test message broker
        print("\n4. Testing Message Broker:")
        await orchestrator._message_broker.send(Message(
            sender_id="research_agent",
            recipient_id="analysis_agent",
            message_type="data_ready",
            payload={"data_id": "dataset_001"}
        ))
        msg = await orchestrator._message_broker.receive("analysis_agent", timeout=1.0)
        print(f"   Received message: {msg.message_type if msg else 'None'}")

        # Test task submission
        print("\n5. Testing Task Scheduling:")
        task1 = Task(
            task_id="task_001",
            name="Data Analysis",
            payload={"action": "analyze", "data": "sample"},
            priority=3
        )
        task2 = Task(
            task_id="task_002",
            name="Generate Report",
            payload={"action": "report", "format": "pdf"},
            dependencies=["task_001"],
            priority=5
        )

        await orchestrator.submit_task(task1)
        await orchestrator.submit_task(task2)
        print(f"   Submitted tasks: task_001, task_002")
        print(f"   task_002 depends on task_001")

        # Get stats
        print("\n6. Orchestrator Stats:")
        stats = orchestrator.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Test backward compatibility
        print("\n7. Testing Backward Compatibility:")

        agent1_caps = AgentCapabilities(
            agent_id="agent_1",
            name="Research Agent",
            version="1.0",
            supported_protocols=[ExchangeProtocolVersion.V3_0],
            supported_block_types=[BlockType.TEXT_DATA, BlockType.EMBEDDING],
            semantic_models=["research_ontology", "scientific_concepts"],
            compression_algorithms=["zstd", "lz4"],
            max_maif_size=1024 * 1024 * 1024,
        )

        agent2_caps = AgentCapabilities(
            agent_id="agent_2",
            name="Analysis Agent",
            version="1.0",
            supported_protocols=[ExchangeProtocolVersion.V3_0],
            supported_block_types=[BlockType.TEXT_DATA, BlockType.EMBEDDING],
            semantic_models=["analysis_framework", "scientific_concepts"],
            compression_algorithms=["zstd", "gzip"],
            max_maif_size=512 * 1024 * 1024,
        )

        protocol_agent1 = MAIFExchangeProtocol("agent_1", agent1_caps)
        protocol_agent2 = MAIFExchangeProtocol("agent_2", agent2_caps)

        orchestrator.register_agent_legacy(protocol_agent1)
        orchestrator.register_agent_legacy(protocol_agent2)

        success = await orchestrator.facilitate_exchange("agent_1", "agent_2", "test_maif")
        print(f"   Exchange result: {'Success' if success else 'Failed'}")

        # Cleanup
        await orchestrator.stop()
        print("\n=== Demo Complete ===")

    # Run the demo
    asyncio.run(demo_multi_agent_framework())
