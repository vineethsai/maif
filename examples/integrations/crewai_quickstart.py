#!/usr/bin/env python3
"""
MAIF + CrewAI Quickstart

The simplest possible example showing how to add cryptographic provenance
to any CrewAI application with just a few lines of code.

Before (No Provenance):
    crew = Crew(agents=[...], tasks=[...])
    result = crew.kickoff()

After (With MAIF):
    from maif.integrations.crewai import MAIFCrewCallback
    
    callback = MAIFCrewCallback("crew_audit.maif")
    crew = Crew(
        agents=[...], tasks=[...],
        task_callback=callback.on_task_complete,
        step_callback=callback.on_step,
    )
    result = crew.kickoff()
    callback.finalize()

That's it! Your crew now has:
- Ed25519 signatures on every action
- Hash-chained blocks for tamper detection
- Full audit trail for compliance
"""

import sys
from pathlib import Path

# Add parent path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# MAIF import - this is what you need!
from maif.integrations.crewai import MAIFCrewCallback


def run_demo_crew():
    """Run a demo crew with MAIF provenance tracking.
    
    This demo simulates a CrewAI crew without requiring LLM API keys.
    """
    
    # === Step 1: Create the MAIF callback ===
    callback = MAIFCrewCallback("quickstart_crew.maif")
    
    # Log crew start
    callback.on_crew_start(
        crew_name="Quickstart Demo Crew",
        agents=None,  # In real usage, pass your agent objects
        tasks=None,   # In real usage, pass your task objects
    )
    
    print("Running CrewAI demo with MAIF provenance tracking...")
    print("-" * 50)
    
    # === Step 2: Simulate agent steps (these would happen automatically) ===
    
    # Simulate researcher agent steps
    class MockStep:
        def __init__(self, thought, action, action_input, observation):
            self.thought = thought
            self.action = action
            self.action_input = action_input
            self.observation = observation
    
    steps = [
        MockStep(
            thought="I need to research the topic",
            action="search",
            action_input="AI security best practices",
            observation="Found 5 relevant sources"
        ),
        MockStep(
            thought="Let me analyze the findings",
            action="analyze",
            action_input="Analyzing security patterns",
            observation="Identified key recommendations"
        ),
    ]
    
    print("\n[AGENT] Executing steps:")
    for step in steps:
        callback.on_step(step)
        print(f"  - {step.action}: {step.thought[:40]}...")
    
    # === Step 3: Log task completion ===
    
    class MockTaskOutput:
        description = "Research AI security best practices"
        raw = "Key findings: Input validation, least privilege, audit logging"
        agent = "researcher"
        output_format = "raw"
    
    callback.on_task_complete(MockTaskOutput())
    print("\n[TASK] Completed: Research task")
    
    # === Step 4: Log crew completion ===
    
    class MockCrewResult:
        raw = "Research complete with security recommendations"
        tasks_output = [MockTaskOutput()]
    
    callback.on_crew_end(result=MockCrewResult())
    print("[CREW] Finished")
    
    # === Step 5: Finalize the artifact ===
    callback.finalize()
    
    # === Verify the provenance ===
    print("\n" + "-" * 50)
    print("Verifying cryptographic provenance...")
    
    from maif import MAIFDecoder
    decoder = MAIFDecoder("quickstart_crew.maif")
    decoder.load()
    
    is_valid, errors = decoder.verify_integrity()
    
    # Show stats
    stats = callback.get_statistics()
    
    print(f"\nArtifact: quickstart_crew.maif")
    print(f"Blocks: {len(decoder.blocks)}")
    print(f"Tasks completed: {stats.get('tasks_completed', 0)}")
    print(f"Steps executed: {stats.get('steps_executed', 0)}")
    print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")
    
    if is_valid:
        print("\nYour CrewAI now has cryptographic provenance!")
        print("Every agent action is signed and tamper-evident.")
    
    # Cleanup
    import os
    os.remove("quickstart_crew.maif")
    
    return is_valid


def run_real_crew_example():
    """Example of how to use MAIF with a real CrewAI crew.
    
    This shows the pattern - uncomment and fill in your agents/tasks.
    """
    print("""
# Real CrewAI Example (requires crewai and LLM API key):

from crewai import Agent, Task, Crew
from maif.integrations.crewai import MAIFCrewCallback

# Create your agents
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information",
    backstory="You are an expert researcher",
)

writer = Agent(
    role="Technical Writer", 
    goal="Create clear documentation",
    backstory="You excel at technical writing",
)

# Create your tasks
research_task = Task(
    description="Research the topic thoroughly",
    expected_output="A detailed research summary",
    agent=researcher,
)

writing_task = Task(
    description="Write documentation from research",
    expected_output="Clear technical documentation",
    agent=writer,
)

# === ADD MAIF PROVENANCE ===
callback = MAIFCrewCallback("my_crew_audit.maif")

# Create the crew with callbacks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    task_callback=callback.on_task_complete,  # <-- Add this
    step_callback=callback.on_step,            # <-- Add this
)

# Run the crew
result = crew.kickoff()

# Finalize the audit trail
callback.finalize()

print(f"Audit saved to: my_crew_audit.maif")
""")


if __name__ == "__main__":
    print("=" * 60)
    print("  MAIF + CrewAI Quickstart")
    print("=" * 60)
    print()
    
    # Run the demo
    success = run_demo_crew()
    
    print("\n" + "=" * 60)
    print("  Usage Pattern")
    print("=" * 60)
    run_real_crew_example()
    
    sys.exit(0 if success else 1)

