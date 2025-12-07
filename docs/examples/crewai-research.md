# CrewAI Research Crew with Provenance

Complete example of a multi-agent research crew with cryptographic audit trails.

## Introduction

This example demonstrates a research workflow using CrewAI with MAIF provenance tracking:

- **CrewAI** for multi-agent orchestration
- **MAIF** for cryptographic provenance and audit trails
- **Two specialized agents** (Researcher and Writer)
- **Complete execution logging** with tamper detection

The system performs research on a topic and produces documentation, while maintaining a complete record of every agent action, reasoning step, and task completion.

## Architecture

### Agent Roles

**1. Researcher Agent**
- Searches for information on the given topic
- Analyzes and synthesizes findings
- Produces research summaries
- All reasoning steps logged to MAIF

**2. Writer Agent**
- Receives research findings
- Creates structured documentation
- Formats output for readability
- Task completion logged to MAIF

### Data Flow

```
Input Topic
    |
    v
Researcher Agent (research task)
    |-- step: thought -> action -> observation (logged)
    |-- step: thought -> action -> observation (logged)
    |-- task complete (logged)
    |
    v
Writer Agent (writing task)
    |-- step: thought -> action -> observation (logged)
    |-- task complete (logged)
    |
    v
Final Documentation + MAIF Artifact
```

## Implementation

### Complete Example

```python
from crewai import Agent, Task, Crew
from maif.integrations.crewai import MAIFCrewCallback

# Create MAIF callback for provenance tracking
callback = MAIFCrewCallback("research_session.maif")

# Define the Researcher agent
researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct thorough research and provide accurate, comprehensive information",
    backstory="""You are an experienced research analyst with expertise in 
    gathering and synthesizing information from various sources. You are known 
    for your attention to detail and ability to identify key insights.""",
    verbose=True,
    allow_delegation=False,
)

# Define the Writer agent
writer = Agent(
    role="Technical Writer",
    goal="Create clear, well-structured documentation from research findings",
    backstory="""You are a skilled technical writer who excels at transforming 
    complex research into accessible documentation. You focus on clarity, 
    organization, and reader comprehension.""",
    verbose=True,
    allow_delegation=False,
)

# Define the research task
research_task = Task(
    description="""Research the topic: {topic}
    
    Provide a comprehensive analysis including:
    1. Key concepts and definitions
    2. Current state and trends
    3. Important considerations
    4. Relevant examples or case studies
    """,
    expected_output="A detailed research summary with key findings and insights",
    agent=researcher,
)

# Define the writing task
writing_task = Task(
    description="""Based on the research provided, create documentation that:
    
    1. Introduces the topic clearly
    2. Explains key concepts
    3. Discusses important considerations
    4. Provides actionable insights
    5. Includes a summary section
    """,
    expected_output="Well-structured documentation suitable for technical readers",
    agent=writer,
)

# Create the crew with MAIF callbacks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    task_callback=callback.on_task_complete,
    step_callback=callback.on_step,
    verbose=True,
)

# Execute with provenance tracking
callback.on_crew_start(
    crew_name="Research Documentation Crew",
    agents=crew.agents,
    tasks=crew.tasks,
    inputs={"topic": "AI Agent Security Best Practices"},
)

try:
    result = crew.kickoff(inputs={"topic": "AI Agent Security Best Practices"})
    callback.on_crew_end(result=result)
except Exception as e:
    callback.on_crew_end(error=e)
finally:
    callback.finalize()

# Print execution statistics
stats = callback.get_statistics()
print(f"\nExecution Statistics:")
print(f"  Tasks completed: {stats['tasks_completed']}")
print(f"  Steps executed: {stats['steps_executed']}")
print(f"  Tool calls: {stats['tool_calls']}")
print(f"  Duration: {stats['duration_seconds']:.2f}s")
```

### Verifying the Audit Trail

After execution, verify the integrity of the recorded events:

```python
from maif import MAIFDecoder

# Load and verify the artifact
decoder = MAIFDecoder("research_session.maif")
decoder.load()

# Check cryptographic integrity
is_valid, errors = decoder.verify_integrity()
print(f"Artifact integrity: {'VALID' if is_valid else 'INVALID'}")

if not is_valid:
    for error in errors:
        print(f"  Error: {error}")

# Inspect the audit trail
print(f"\nAudit Trail ({len(decoder.blocks)} events):")
for i, block in enumerate(decoder.blocks):
    event_type = block.metadata.get("type", "unknown")
    subtype = block.metadata.get("event_subtype", "")
    timestamp = block.metadata.get("timestamp", "")
    
    label = f"{event_type}"
    if subtype:
        label += f" ({subtype})"
    
    print(f"  {i+1}. {label}")
```

### Using Persistent Memory

Add persistent memory that survives across sessions:

```python
from maif.integrations.crewai import MAIFCrewMemory

# Create memory store
memory = MAIFCrewMemory("crew_memory.maif")

# Store insights from research
memory.save(
    content="AI agents should implement principle of least privilege",
    agent="researcher",
    tags=["security", "best-practice"],
    importance=0.9,
)

memory.save(
    content="Input validation is critical for agent tool usage",
    agent="researcher", 
    tags=["security", "input-validation"],
    importance=0.85,
)

# In a later session, retrieve relevant memories
security_insights = memory.search("security best practices")
for insight in security_insights:
    print(f"- {insight['content']} (importance: {insight['importance']})")

memory.finalize()
```

## Events Logged

The MAIF artifact captures these events:

| Event | Description | Data Captured |
|-------|-------------|---------------|
| `agent_start` | Crew kickoff | Crew name, agents, tasks, inputs |
| `agent_action` | Reasoning step | Thought, action, input, observation |
| `tool_end` | Tool invocation | Tool name, result |
| `task_end` | Task completion | Description, output, agent |
| `agent_end` | Crew completion | Result, statistics, duration |

## Inspecting Specific Events

Extract and analyze specific event types:

```python
import json
from maif import MAIFDecoder

decoder = MAIFDecoder("research_session.maif")
decoder.load()

# Find all task completions
print("Task Completions:")
for block in decoder.blocks:
    if block.metadata.get("type") == "task_end":
        data = json.loads(block.data.decode("utf-8"))
        task_data = data.get("data", {})
        print(f"  Task: {task_data.get('task_description', '')[:50]}...")
        print(f"  Agent: {task_data.get('agent')}")
        print()

# Find all reasoning steps
print("Agent Reasoning Steps:")
step_count = 0
for block in decoder.blocks:
    if block.metadata.get("type") == "agent_action":
        step_count += 1
        data = json.loads(block.data.decode("utf-8"))
        step_data = data.get("data", {})
        action = step_data.get("action", "unknown")
        print(f"  Step {step_count}: {action}")

print(f"\nTotal reasoning steps: {step_count}")
```

## Error Handling

Handle errors gracefully while preserving the audit trail:

```python
from crewai import Crew
from maif.integrations.crewai import MAIFCrewCallback

callback = MAIFCrewCallback("session.maif")

try:
    callback.on_crew_start(crew_name="My Crew")
    result = crew.kickoff()
    callback.on_crew_end(result=result)
    
except Exception as e:
    # Log the error to the artifact
    callback.on_crew_end(error=e)
    print(f"Crew execution failed: {e}")
    
finally:
    # Always finalize to seal the artifact
    callback.finalize()
```

The error is recorded in the artifact with full stack trace, enabling post-mortem analysis.

## Requirements

```
maif[integrations]
crewai>=0.30.0
```

Note: CrewAI requires Python 3.10 or higher.

## Related

- [CrewAI Integration Guide](../guide/integrations/crewai.md)
- [LangGraph RAG Example](./langgraph-rag.md)
- [MAIF Security Model](../guide/security-model.md)

