"""
MAIF Callback Handler for AWS Strands Agents SDK.

Tracks agent execution, tool invocations, and LLM calls with
cryptographic provenance in MAIF artifacts.

The Strands Agents SDK uses a callback handler pattern where
handlers receive keyword arguments containing event data during
agent execution.
"""

import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

# Conditional import for Strands Agents SDK (strands-agents package)
# Note: There are other packages named 'strands', so we check for specific imports
STRANDS_AVAILABLE = False
Agent = None
PrintingCallbackHandler = None
CompositeCallbackHandler = None

try:
    # Check for strands-agents specific imports
    from strands import Agent
    from strands.handlers.callback_handler import (
        PrintingCallbackHandler,
        CompositeCallbackHandler,
    )
    STRANDS_AVAILABLE = True
except (ImportError, AttributeError):
    # Either strands-agents not installed or wrong strands package
    pass

from maif.integrations._base import BaseMAIFCallback, EventType, MAIFProvenanceTracker
from maif.integrations._utils import (
    safe_serialize,
    extract_error_info,
    generate_run_id,
    truncate_string,
    safe_get_attr,
)

logger = logging.getLogger(__name__)


class MAIFStrandsCallback(BaseMAIFCallback):
    """MAIF-backed callback handler for AWS Strands Agents SDK.

    This callback handler tracks all agent execution events including:
    - Agent invocation start/end
    - Tool invocations with inputs/outputs
    - LLM text generation (streaming chunks)
    - Errors and force stops

    The handler follows the Strands callback interface, accepting **kwargs
    with event data and logging them to a MAIF artifact.

    Usage:
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
        with MAIFStrandsCallback("agent.maif") as callback:
            agent = Agent(callback_handler=callback)
            agent("Hello!")
        # Automatically finalized

    Combined with PrintingCallbackHandler:
        from strands.handlers.callback_handler import (
            PrintingCallbackHandler,
            CompositeCallbackHandler,
        )
        from maif.integrations.strands import MAIFStrandsCallback

        maif_callback = MAIFStrandsCallback("agent.maif")
        combined = CompositeCallbackHandler(
            PrintingCallbackHandler(),
            maif_callback,
        )
        agent = Agent(callback_handler=combined)

    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Optional identifier for this callback handler
        track_text_chunks: Whether to log individual text chunks (can be verbose)
        track_tool_inputs: Whether to log tool input parameters
        max_text_length: Maximum length for text content in logs

    Attributes:
        tracker: The underlying MAIFProvenanceTracker
        invocation_count: Number of agent invocations
        tool_call_count: Number of tool calls made
        error_count: Number of errors encountered
    """

    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
        track_text_chunks: bool = False,
        track_tool_inputs: bool = True,
        max_text_length: int = 5000,
    ):
        """Initialize the MAIF Strands callback handler.

        Args:
            artifact_path: Path to the MAIF artifact file
            agent_id: Identifier for this handler (default: "strands_agent")
            track_text_chunks: Log individual streaming text chunks
            track_tool_inputs: Log tool invocation inputs
            max_text_length: Max characters for text content

        Note:
            This callback can be instantiated without strands-agents installed.
            It follows the Strands callback interface (__call__ with **kwargs)
            and will work with any compatible agent framework.
        """
        super().__init__(
            artifact_path=artifact_path,
            agent_id=agent_id or "strands_agent",
        )

        self.track_text_chunks = track_text_chunks
        self.track_tool_inputs = track_tool_inputs
        self.max_text_length = max_text_length

        # Counters
        self.invocation_count = 0
        self.tool_call_count = 0
        self.error_count = 0
        self._text_chunk_count = 0

        # State tracking
        self._current_run_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._accumulated_text: str = ""
        self._seen_tool_ids: Set[str] = set()
        self._active_tools: Dict[str, Dict[str, Any]] = {}

    def get_framework_name(self) -> str:
        """Return the framework name."""
        return "strands"

    def __call__(self, **kwargs: Any) -> None:
        """Handle callback events from Strands agent.

        This method is called by the Strands agent with various kwargs
        containing event information. The kwargs can include:

        - init_event_loop: Boolean, agent loop initialization
        - start_event_loop: Boolean, new cycle starting
        - data: String, text generation chunk
        - current_tool_use: Dict, tool invocation info
        - message: Dict, complete message
        - result: Any, agent completion result
        - complete: Boolean, indicates final response chunk
        - force_stop: Boolean, agent was force stopped
        - force_stop_reason: String, reason for stopping
        - reasoningText: String, model reasoning content

        Args:
            **kwargs: Event data from the Strands agent
        """
        try:
            self._handle_event(**kwargs)
        except Exception as e:
            logger.warning(f"Error in MAIF callback: {e}")
            # Don't re-raise to avoid breaking the agent

    def _handle_event(self, **kwargs: Any) -> None:
        """Internal event handler with full processing logic."""

        # Handle event loop initialization
        if kwargs.get("init_event_loop"):
            self._on_agent_start()
            return

        # Handle new cycle start
        if kwargs.get("start_event_loop"):
            self._on_cycle_start()
            return

        # Handle force stop
        if kwargs.get("force_stop"):
            self._on_force_stop(kwargs.get("force_stop_reason"))
            return

        # Handle tool invocation
        current_tool = kwargs.get("current_tool_use")
        if current_tool and isinstance(current_tool, dict):
            self._on_tool_event(current_tool)

        # Handle text generation
        if "data" in kwargs:
            self._on_text_data(kwargs["data"], kwargs.get("complete", False))

        # Handle reasoning text (if present)
        if "reasoningText" in kwargs:
            self._on_reasoning_text(kwargs["reasoningText"])

        # Handle complete message
        if "message" in kwargs:
            self._on_message(kwargs["message"])

        # Handle result (completion)
        if "result" in kwargs:
            self._on_result(kwargs["result"])

    def _on_agent_start(self) -> None:
        """Handle agent invocation start."""
        self.invocation_count += 1
        self._current_run_id = generate_run_id()
        self._start_time = time.time()
        self._accumulated_text = ""
        self._seen_tool_ids.clear()
        self._active_tools.clear()
        self._text_chunk_count = 0

        self.tracker.log_event(
            event_type=EventType.AGENT_START,
            data={
                "invocation_number": self.invocation_count,
            },
            metadata={
                "framework": "strands",
                "event_subtype": "agent_start",
            },
            run_id=self._current_run_id,
        )

        logger.debug(f"Strands agent started, run_id={self._current_run_id}")

    def _on_cycle_start(self) -> None:
        """Handle new event loop cycle start."""
        # This indicates a new iteration in the agent loop
        # We don't log every cycle to avoid verbosity, but track it
        pass

    def _on_force_stop(self, reason: Optional[str] = None) -> None:
        """Handle agent force stop event."""
        self.error_count += 1
        duration = time.time() - self._start_time if self._start_time else 0

        self.tracker.log_event(
            event_type=EventType.AGENT_ERROR,
            data={
                "force_stop": True,
                "reason": reason or "unknown",
                "duration_seconds": duration,
                "tool_calls": self.tool_call_count,
            },
            metadata={
                "framework": "strands",
                "event_subtype": "force_stop",
            },
            run_id=self._current_run_id,
        )

        logger.debug(f"Strands agent force stopped: {reason}")

    def _on_tool_event(self, tool_info: Dict[str, Any]) -> None:
        """Handle tool invocation events.

        The Strands SDK provides tool info progressively, so we need
        to track when tools start and complete.

        Args:
            tool_info: Dictionary with tool invocation details
        """
        tool_use_id = tool_info.get("toolUseId", "")
        tool_name = tool_info.get("name", "")

        # Skip if we've already logged this tool start
        if not tool_use_id:
            return

        # Check if this is a new tool invocation
        if tool_use_id not in self._seen_tool_ids and tool_name:
            self._seen_tool_ids.add(tool_use_id)
            self.tool_call_count += 1

            # Log tool start
            tool_data = {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "tool_number": self.tool_call_count,
            }

            if self.track_tool_inputs:
                tool_input = tool_info.get("input", {})
                tool_data["input"] = truncate_string(
                    safe_serialize(tool_input), self.max_text_length
                )

            self._active_tools[tool_use_id] = {
                "name": tool_name,
                "start_time": time.time(),
            }

            self.tracker.log_event(
                event_type=EventType.TOOL_START,
                data=tool_data,
                metadata={
                    "framework": "strands",
                    "event_subtype": "tool_start",
                    "tool_name": tool_name,
                },
                run_id=self._current_run_id,
            )

            logger.debug(f"Strands tool started: {tool_name}")

    def _on_text_data(self, data: str, complete: bool = False) -> None:
        """Handle text generation data.

        Args:
            data: Text chunk from the model
            complete: Whether this is the final chunk
        """
        if not data:
            return

        self._accumulated_text += data
        self._text_chunk_count += 1

        # Optionally log individual chunks
        if self.track_text_chunks:
            self.tracker.log_event(
                event_type=EventType.LLM_END,
                data={
                    "chunk": truncate_string(data, 500),
                    "chunk_number": self._text_chunk_count,
                    "is_complete": complete,
                },
                metadata={
                    "framework": "strands",
                    "event_subtype": "text_chunk",
                },
                run_id=self._current_run_id,
            )

    def _on_reasoning_text(self, reasoning: str) -> None:
        """Handle reasoning text from the model.

        Args:
            reasoning: Model's reasoning content
        """
        if not reasoning:
            return

        self.tracker.log_event(
            event_type=EventType.AGENT_ACTION,
            data={
                "reasoning": truncate_string(reasoning, self.max_text_length),
            },
            metadata={
                "framework": "strands",
                "event_subtype": "reasoning",
            },
            run_id=self._current_run_id,
        )

    def _on_message(self, message: Any) -> None:
        """Handle complete message event.

        Args:
            message: Complete message object/dict
        """
        if not message:
            return

        # Extract message info
        if isinstance(message, dict):
            role = message.get("role", "unknown")
            content = message.get("content", "")
        else:
            role = safe_get_attr(message, "role", "unknown")
            content = safe_get_attr(message, "content", "")

        # Serialize content if it's complex
        if not isinstance(content, str):
            content = safe_serialize(content)

        self.tracker.log_event(
            event_type=EventType.LLM_END,
            data={
                "role": role,
                "content": truncate_string(content, self.max_text_length),
            },
            metadata={
                "framework": "strands",
                "event_subtype": "message",
            },
            run_id=self._current_run_id,
        )

    def _on_result(self, result: Any) -> None:
        """Handle agent completion result.

        Args:
            result: The final result from the agent
        """
        duration = time.time() - self._start_time if self._start_time else 0

        # Log any active tools as completed
        for tool_id, tool_info in self._active_tools.items():
            tool_duration = time.time() - tool_info.get("start_time", time.time())
            self.tracker.log_event(
                event_type=EventType.TOOL_END,
                data={
                    "tool_name": tool_info.get("name", "unknown"),
                    "tool_use_id": tool_id,
                    "duration_seconds": tool_duration,
                },
                metadata={
                    "framework": "strands",
                    "event_subtype": "tool_end",
                },
                run_id=self._current_run_id,
            )
        self._active_tools.clear()

        # Extract result info
        result_data: Dict[str, Any] = {
            "duration_seconds": duration,
            "tool_calls": self.tool_call_count,
            "text_chunks": self._text_chunk_count,
        }

        if result is not None:
            if isinstance(result, str):
                result_data["output"] = truncate_string(result, self.max_text_length)
            elif isinstance(result, dict):
                result_data["output"] = truncate_string(
                    safe_serialize(result), self.max_text_length
                )
            else:
                # Try to get common attributes
                output_text = safe_get_attr(result, "text", None)
                if output_text is None:
                    output_text = safe_get_attr(result, "content", None)
                if output_text is None:
                    output_text = str(result)
                result_data["output"] = truncate_string(output_text, self.max_text_length)

        # Log accumulated response if we have text
        if self._accumulated_text and not self.track_text_chunks:
            result_data["full_response"] = truncate_string(
                self._accumulated_text, self.max_text_length
            )

        self.tracker.log_event(
            event_type=EventType.AGENT_END,
            data=result_data,
            metadata={
                "framework": "strands",
                "event_subtype": "agent_end",
            },
            run_id=self._current_run_id,
        )

        logger.debug(
            f"Strands agent completed, duration={duration:.2f}s, "
            f"tools={self.tool_call_count}"
        )

    def log_tool_result(
        self,
        tool_use_id: str,
        tool_name: str,
        result: Any,
        error: Optional[Exception] = None,
    ) -> None:
        """Manually log a tool result.

        Use this method when you need to explicitly log tool completion
        with result data, for example when wrapping tools with custom logic.

        Args:
            tool_use_id: The tool use ID from Strands
            tool_name: Name of the tool
            result: The tool's result
            error: Optional exception if the tool failed
        """
        # Calculate duration if we tracked the start
        duration = None
        if tool_use_id in self._active_tools:
            tool_info = self._active_tools.pop(tool_use_id)
            duration = time.time() - tool_info.get("start_time", time.time())

        if error:
            self.error_count += 1
            self.tracker.log_event(
                event_type=EventType.TOOL_ERROR,
                data={
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "error": extract_error_info(error),
                    "duration_seconds": duration,
                },
                metadata={
                    "framework": "strands",
                    "event_subtype": "tool_error",
                },
                run_id=self._current_run_id,
            )
        else:
            self.tracker.log_event(
                event_type=EventType.TOOL_END,
                data={
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "result": truncate_string(
                        safe_serialize(result) if not isinstance(result, str) else result,
                        self.max_text_length
                    ),
                    "duration_seconds": duration,
                },
                metadata={
                    "framework": "strands",
                    "event_subtype": "tool_end",
                },
                run_id=self._current_run_id,
            )

    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """Log an error event.

        Use this to manually log errors that occur during agent execution.

        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
        """
        self.error_count += 1

        self.tracker.log_event(
            event_type=EventType.AGENT_ERROR,
            data={
                "error": extract_error_info(error),
                "context": context,
            },
            metadata={
                "framework": "strands",
                "event_subtype": "error",
            },
            run_id=self._current_run_id,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get callback statistics.

        Returns:
            Dictionary with invocation/tool/error counts
        """
        return {
            "invocations": self.invocation_count,
            "tool_calls": self.tool_call_count,
            "errors": self.error_count,
            "current_run_id": self._current_run_id,
        }

    def reset_statistics(self) -> None:
        """Reset all counters for a new session."""
        self.invocation_count = 0
        self.tool_call_count = 0
        self.error_count = 0
        self._text_chunk_count = 0
        self._current_run_id = None
        self._start_time = None
        self._accumulated_text = ""
        self._seen_tool_ids.clear()
        self._active_tools.clear()

    def __enter__(self) -> "MAIFStrandsCallback":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - finalize the artifact."""
        if exc_val is not None:
            # Log the error before finalizing
            self.log_error(exc_val, context="Context manager exit")
        self.finalize()


def create_composite_handler(
    artifact_path: Union[str, Path],
    include_printing: bool = True,
    verbose_tool_use: bool = True,
    **maif_kwargs: Any,
) -> "CompositeCallbackHandler":
    """Create a composite handler with MAIF tracking and optional printing.

    This is a convenience function to create a handler that both logs
    to MAIF and prints to stdout (like the default Strands behavior).

    Usage:
        from maif.integrations.strands import create_composite_handler

        handler = create_composite_handler("agent.maif")
        agent = Agent(callback_handler=handler)

        # After agent execution
        handler.handlers[1].finalize()  # Finalize MAIF callback

    Args:
        artifact_path: Path to the MAIF artifact file
        include_printing: Include PrintingCallbackHandler for stdout
        verbose_tool_use: Pass to PrintingCallbackHandler
        **maif_kwargs: Additional kwargs for MAIFStrandsCallback

    Returns:
        CompositeCallbackHandler with MAIF and optionally printing handlers
    """
    if not STRANDS_AVAILABLE:
        raise ImportError(
            "Strands Agents SDK is required. "
            "Install with: pip install strands-agents"
        )

    handlers: List[Callable[..., None]] = []

    if include_printing:
        handlers.append(PrintingCallbackHandler(verbose_tool_use=verbose_tool_use))

    maif_callback = MAIFStrandsCallback(artifact_path, **maif_kwargs)
    handlers.append(maif_callback)

    return CompositeCallbackHandler(*handlers)
