"""
MAIF Callback Handlers for CrewAI

Provides callback implementations for tracking CrewAI agent workflows
with cryptographic provenance in MAIF artifacts.

This module implements the callback interfaces expected by CrewAI's
`step_callback` and `task_callback` parameters, automatically logging:
- Agent reasoning steps (thought, action, observation cycle)
- Task completions with outputs
- Tool invocations and results
- Crew-level events (kickoff, completion)
"""

import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from maif.integrations._base import BaseMAIFCallback, EventType, MAIFProvenanceTracker
from maif.integrations._utils import (
    safe_serialize,
    extract_error_info,
    generate_run_id,
    truncate_string,
    safe_get_attr,
)

logger = logging.getLogger(__name__)


# Type aliases for CrewAI callback signatures
# These match CrewAI's expected callback signatures
TaskCallbackFn = Callable[[Any], None]
StepCallbackFn = Callable[[Any], None]


class MAIFCrewCallback(BaseMAIFCallback):
    """Main callback handler for CrewAI crews.
    
    This class provides both task-level and step-level callbacks that can
    be passed to CrewAI's Crew constructor. It tracks the entire lifecycle
    of a crew run including:
    
    - Crew kickoff and completion
    - Individual task executions
    - Agent reasoning steps (ReAct cycle)
    - Tool usage and results
    - Agent delegation events
    
    Usage:
        from crewai import Crew, Agent, Task
        from maif.integrations.crewai import MAIFCrewCallback
        
        callback = MAIFCrewCallback("crew_session.maif")
        
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            task_callback=callback.on_task_complete,
            step_callback=callback.on_step,
        )
        
        result = crew.kickoff()
        callback.finalize()
    
    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Optional identifier for the callback handler
        track_tokens: Whether to track token usage (if available)
        track_timing: Whether to track execution timing
        
    Attributes:
        tracker: The underlying MAIFProvenanceTracker
        task_count: Number of tasks completed
        step_count: Number of steps processed
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
        track_tokens: bool = True,
        track_timing: bool = True,
    ):
        super().__init__(
            artifact_path=artifact_path,
            agent_id=agent_id or "crewai_callback",
        )
        
        self.track_tokens = track_tokens
        self.track_timing = track_timing
        
        # Counters for summary
        self.task_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        self.error_count = 0
        
        # Timing tracking
        self._start_time: Optional[float] = None
        self._task_start_times: Dict[str, float] = {}
        
        # Run context
        self._current_run_id: Optional[str] = None
        self._crew_name: Optional[str] = None
    
    def get_framework_name(self) -> str:
        """Return the framework name."""
        return "crewai"
    
    # =========================================================================
    # Crew Lifecycle Methods
    # =========================================================================
    
    def on_crew_start(
        self,
        crew_name: Optional[str] = None,
        agents: Optional[List[Any]] = None,
        tasks: Optional[List[Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log crew kickoff event.
        
        Call this manually at the start of a crew run to capture the
        initial configuration and inputs.
        
        Args:
            crew_name: Name of the crew
            agents: List of agents in the crew
            tasks: List of tasks to execute
            inputs: Input parameters for the crew
            
        Returns:
            The run ID for this crew execution
        """
        self._start_time = time.time()
        self._current_run_id = generate_run_id()
        self._crew_name = crew_name
        
        # Extract agent info
        agent_info = []
        if agents:
            for agent in agents:
                agent_info.append({
                    "role": safe_get_attr(agent, "role", "unknown"),
                    "goal": truncate_string(safe_get_attr(agent, "goal", ""), 500),
                    "backstory": truncate_string(safe_get_attr(agent, "backstory", ""), 500),
                    "allow_delegation": safe_get_attr(agent, "allow_delegation", False),
                    "tools": [safe_get_attr(t, "name", str(t)) for t in safe_get_attr(agent, "tools", [])],
                })
        
        # Extract task info
        task_info = []
        if tasks:
            for task in tasks:
                task_info.append({
                    "description": truncate_string(safe_get_attr(task, "description", ""), 500),
                    "expected_output": truncate_string(safe_get_attr(task, "expected_output", ""), 500),
                    "agent": safe_get_attr(safe_get_attr(task, "agent", None), "role", "unassigned"),
                })
        
        self.tracker.log_event(
            event_type=EventType.AGENT_START,
            data={
                "crew_name": crew_name,
                "agents": agent_info,
                "tasks": task_info,
                "inputs": inputs or {},
                "num_agents": len(agents) if agents else 0,
                "num_tasks": len(tasks) if tasks else 0,
            },
            metadata={
                "framework": "crewai",
                "event_subtype": "crew_kickoff",
            },
            run_id=self._current_run_id,
        )
        
        return self._current_run_id
    
    def on_crew_end(
        self,
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Log crew completion event.
        
        Call this manually after crew.kickoff() completes to capture
        the final result and summary statistics.
        
        Args:
            result: The crew's final result (CrewOutput)
            error: Optional exception if the crew failed
        """
        duration = time.time() - self._start_time if self._start_time else 0
        
        if error:
            self.error_count += 1
            self.tracker.log_event(
                event_type=EventType.AGENT_ERROR,
                data={
                    "crew_name": self._crew_name,
                    "error": extract_error_info(error),
                    "duration_seconds": duration,
                    "tasks_completed": self.task_count,
                    "steps_executed": self.step_count,
                },
                metadata={
                    "framework": "crewai",
                    "event_subtype": "crew_error",
                },
                run_id=self._current_run_id,
            )
        else:
            # Extract result info
            result_data = {}
            if result is not None:
                result_data = {
                    "raw": truncate_string(safe_get_attr(result, "raw", str(result)), 5000),
                    "tasks_output": self._extract_tasks_output(result),
                }
                
                # Token usage if available
                if self.track_tokens:
                    token_usage = safe_get_attr(result, "token_usage", None)
                    if token_usage:
                        result_data["token_usage"] = {
                            "total_tokens": safe_get_attr(token_usage, "total_tokens", 0),
                            "prompt_tokens": safe_get_attr(token_usage, "prompt_tokens", 0),
                            "completion_tokens": safe_get_attr(token_usage, "completion_tokens", 0),
                        }
            
            self.tracker.log_event(
                event_type=EventType.AGENT_END,
                data={
                    "crew_name": self._crew_name,
                    "result": result_data,
                    "duration_seconds": duration,
                    "summary": {
                        "tasks_completed": self.task_count,
                        "steps_executed": self.step_count,
                        "tool_calls": self.tool_call_count,
                        "errors": self.error_count,
                    },
                },
                metadata={
                    "framework": "crewai",
                    "event_subtype": "crew_complete",
                },
                run_id=self._current_run_id,
            )
    
    def _extract_tasks_output(self, result: Any) -> List[Dict[str, Any]]:
        """Extract task outputs from crew result."""
        outputs = []
        tasks_output = safe_get_attr(result, "tasks_output", [])
        
        for task_output in tasks_output:
            outputs.append({
                "description": truncate_string(safe_get_attr(task_output, "description", ""), 500),
                "raw": truncate_string(safe_get_attr(task_output, "raw", ""), 2000),
                "agent": safe_get_attr(task_output, "agent", "unknown"),
            })
        
        return outputs
    
    # =========================================================================
    # Task Callback (for Crew task_callback parameter)
    # =========================================================================
    
    def on_task_complete(self, task_output: Any) -> None:
        """Callback for task completion events.
        
        This method should be passed to Crew's `task_callback` parameter.
        It's called automatically by CrewAI when a task finishes.
        
        Args:
            task_output: The TaskOutput object from CrewAI containing:
                - description: Task description
                - raw: Raw output string
                - pydantic: Pydantic model output (if configured)
                - json_dict: JSON output (if configured)
                - agent: Agent that executed the task
                - output_format: Format of the output
        """
        self.task_count += 1
        
        # Calculate task duration if we tracked the start
        task_desc = safe_get_attr(task_output, "description", "") or ""
        task_key = truncate_string(str(task_desc), 100)
        duration = None
        if task_key in self._task_start_times:
            duration = time.time() - self._task_start_times.pop(task_key)
        
        # Extract output based on format
        raw_output = safe_get_attr(task_output, "raw", "") or ""
        output_data = {
            "raw": truncate_string(str(raw_output), 5000),
        }
        
        # Check for structured output
        pydantic_output = safe_get_attr(task_output, "pydantic", None)
        if pydantic_output:
            try:
                output_data["pydantic"] = pydantic_output.model_dump() if hasattr(pydantic_output, "model_dump") else str(pydantic_output)
            except Exception:
                output_data["pydantic"] = str(pydantic_output)
        
        json_output = safe_get_attr(task_output, "json_dict", None)
        if json_output:
            output_data["json"] = json_output
        
        self.tracker.log_event(
            event_type=EventType.TASK_END,
            data={
                "task_description": truncate_string(str(task_desc), 1000),
                "output": output_data,
                "agent": safe_get_attr(task_output, "agent", "unknown") or "unknown",
                "output_format": safe_get_attr(task_output, "output_format", "raw") or "raw",
                "duration_seconds": duration,
                "task_number": self.task_count,
            },
            metadata={
                "framework": "crewai",
                "event_subtype": "task_complete",
            },
            run_id=self._current_run_id,
        )
        
        logger.debug(f"CrewAI task completed: {task_key[:50]}...")
    
    def on_task_start(self, task: Any) -> None:
        """Log task start event.
        
        Call this manually before a task starts if you want to track
        task duration. CrewAI doesn't provide a built-in task start callback.
        
        Args:
            task: The Task object about to be executed
        """
        task_desc = safe_get_attr(task, "description", "")
        task_key = truncate_string(task_desc, 100)
        self._task_start_times[task_key] = time.time()
        
        self.tracker.log_event(
            event_type=EventType.TASK_START,
            data={
                "task_description": truncate_string(task_desc, 1000),
                "expected_output": truncate_string(safe_get_attr(task, "expected_output", ""), 500),
                "agent": safe_get_attr(safe_get_attr(task, "agent", None), "role", "unassigned"),
                "tools": [safe_get_attr(t, "name", str(t)) for t in safe_get_attr(task, "tools", [])],
            },
            metadata={
                "framework": "crewai",
                "event_subtype": "task_start",
            },
            run_id=self._current_run_id,
        )
    
    # =========================================================================
    # Step Callback (for Crew step_callback parameter)
    # =========================================================================
    
    def on_step(self, step_output: Any) -> None:
        """Callback for agent step events (ReAct cycle).
        
        This method should be passed to Crew's `step_callback` parameter.
        It's called automatically by CrewAI for each agent reasoning step.
        
        A step typically includes:
        - Thought: Agent's reasoning about what to do
        - Action: The tool or action chosen
        - Action Input: Parameters for the action
        - Observation: Result of the action
        
        Args:
            step_output: The step output object from CrewAI containing
                the agent's thought process and action results.
        """
        self.step_count += 1
        
        # Extract step components
        thought = safe_get_attr(step_output, "thought", "")
        action = safe_get_attr(step_output, "action", "")
        action_input = safe_get_attr(step_output, "action_input", "")
        observation = safe_get_attr(step_output, "observation", "")
        
        # Determine event type based on content
        event_type = EventType.AGENT_ACTION
        event_subtype = "agent_step"
        
        # Check if this is a tool call
        if action and action not in ["Final Answer", "Delegate work", ""]:
            self.tool_call_count += 1
            event_type = EventType.TOOL_END
            event_subtype = "tool_result"
        
        self.tracker.log_event(
            event_type=event_type,
            data={
                "thought": truncate_string(str(thought), 2000),
                "action": str(action),
                "action_input": truncate_string(str(action_input), 2000),
                "observation": truncate_string(str(observation), 2000),
                "step_number": self.step_count,
            },
            metadata={
                "framework": "crewai",
                "event_subtype": event_subtype,
            },
            run_id=self._current_run_id,
        )
        
        logger.debug(f"CrewAI step {self.step_count}: action={action}")
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get callback statistics.
        
        Returns:
            Dictionary with task/step/error counts and timing info
        """
        duration = time.time() - self._start_time if self._start_time else 0
        return {
            "tasks_completed": self.task_count,
            "steps_executed": self.step_count,
            "tool_calls": self.tool_call_count,
            "errors": self.error_count,
            "duration_seconds": duration,
            "run_id": self._current_run_id,
        }
    
    def reset_statistics(self) -> None:
        """Reset all counters and state for a new run."""
        self.task_count = 0
        self.step_count = 0
        self.tool_call_count = 0
        self.error_count = 0
        self._start_time = None
        self._task_start_times.clear()
        self._current_run_id = None
        self._crew_name = None


class MAIFTaskCallback:
    """Standalone task callback for simpler usage.
    
    Use this if you only need to track task completions without
    the full step-level tracking.
    
    Usage:
        from maif.integrations.crewai import MAIFTaskCallback
        
        task_callback = MAIFTaskCallback("tasks.maif")
        
        crew = Crew(
            agents=[...],
            tasks=[...],
            task_callback=task_callback,
        )
    
    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Optional identifier
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
    ):
        self._tracker = MAIFProvenanceTracker(
            artifact_path=artifact_path,
            agent_id=agent_id or "crewai_task_callback",
            auto_finalize=False,
        )
        self._task_count = 0
        self._run_id = generate_run_id()
    
    def __call__(self, task_output: Any) -> None:
        """Handle task completion callback.
        
        This method is called directly by CrewAI when passed as task_callback.
        
        Args:
            task_output: The TaskOutput from CrewAI
        """
        self._task_count += 1
        
        self._tracker.log_event(
            event_type=EventType.TASK_END,
            data={
                "task_description": truncate_string(
                    safe_get_attr(task_output, "description", ""), 1000
                ),
                "output": truncate_string(
                    safe_get_attr(task_output, "raw", ""), 5000
                ),
                "agent": safe_get_attr(task_output, "agent", "unknown"),
                "task_number": self._task_count,
            },
            metadata={
                "framework": "crewai",
                "event_subtype": "task_complete",
            },
            run_id=self._run_id,
        )
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact."""
        self._tracker.finalize()
    
    def get_artifact_path(self) -> str:
        """Get the artifact path."""
        return self._tracker.get_artifact_path()


class MAIFStepCallback:
    """Standalone step callback for tracking agent reasoning.
    
    Use this if you only need to track agent steps without
    task-level tracking.
    
    Usage:
        from maif.integrations.crewai import MAIFStepCallback
        
        step_callback = MAIFStepCallback("steps.maif")
        
        crew = Crew(
            agents=[...],
            tasks=[...],
            step_callback=step_callback,
        )
    
    Args:
        artifact_path: Path to the MAIF artifact file
        agent_id: Optional identifier
    """
    
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
    ):
        self._tracker = MAIFProvenanceTracker(
            artifact_path=artifact_path,
            agent_id=agent_id or "crewai_step_callback",
            auto_finalize=False,
        )
        self._step_count = 0
        self._run_id = generate_run_id()
    
    def __call__(self, step_output: Any) -> None:
        """Handle step callback.
        
        This method is called directly by CrewAI when passed as step_callback.
        
        Args:
            step_output: The step output from CrewAI
        """
        self._step_count += 1
        
        self._tracker.log_event(
            event_type=EventType.AGENT_ACTION,
            data={
                "thought": truncate_string(
                    str(safe_get_attr(step_output, "thought", "")), 2000
                ),
                "action": str(safe_get_attr(step_output, "action", "")),
                "action_input": truncate_string(
                    str(safe_get_attr(step_output, "action_input", "")), 2000
                ),
                "observation": truncate_string(
                    str(safe_get_attr(step_output, "observation", "")), 2000
                ),
                "step_number": self._step_count,
            },
            metadata={
                "framework": "crewai",
                "event_subtype": "agent_step",
            },
            run_id=self._run_id,
        )
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact."""
        self._tracker.finalize()
    
    def get_artifact_path(self) -> str:
        """Get the artifact path."""
        return self._tracker.get_artifact_path()

