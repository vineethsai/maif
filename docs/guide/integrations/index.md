# Framework Integrations

MAIF provides drop-in integrations for popular AI agent frameworks, enabling cryptographic provenance tracking with minimal code changes.

## Available Integrations

| Framework | Status | Description |
|-----------|--------|-------------|
| [LangGraph](./langgraph.md) | Available | State checkpointer with provenance |
| [LangChain](./langchain.md) | Coming Soon | Callbacks, VectorStore, Memory |
| [CrewAI](./crewai.md) | Coming Soon | Crew/Agent callbacks, Memory |
| [Strands SDK](./strands.md) | Coming Soon | AWS Strands agent callbacks |

## Installation

Install MAIF with all integrations:

```bash
pip install maif[integrations]
```

Or install specific framework support:

```bash
# LangGraph only
pip install maif langgraph

# LangChain only
pip install maif langchain-core

# CrewAI only
pip install maif crewai
```

## Quick Start

Each integration follows the same pattern: create a MAIF-backed handler, attach it to your framework, and finalize when done.

### LangGraph Example

```python
from langgraph.graph import StateGraph
from maif.integrations.langgraph import MAIFCheckpointer

checkpointer = MAIFCheckpointer("state.maif")
app = graph.compile(checkpointer=checkpointer)
result = app.invoke(state, config)
checkpointer.finalize()
```

### LangChain Example (Coming Soon)

```python
from langchain_core.language_models import BaseLLM
from maif.integrations.langchain import MAIFCallbackHandler

handler = MAIFCallbackHandler("session.maif")
llm.invoke("Hello", config={"callbacks": [handler]})
handler.finalize()
```

### CrewAI Example (Coming Soon)

```python
from crewai import Crew
from maif.integrations.crewai import MAIFCrewCallback

callback = MAIFCrewCallback("crew.maif")
crew = Crew(
    agents=[...],
    tasks=[...],
    task_callback=callback.on_task_complete,
)
crew.kickoff()
callback.finalize()
```

## Common Patterns

### Context Manager

All integrations support context manager usage for automatic finalization:

```python
with MAIFCheckpointer("state.maif") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    result = app.invoke(state, config)
# Automatically finalized
```

### Inspecting Provenance

After running your agent, inspect the audit trail:

```python
from maif import MAIFDecoder

decoder = MAIFDecoder("session.maif")
decoder.load()

# Verify integrity
is_valid, errors = decoder.verify_integrity()

# Inspect events
for block in decoder.blocks:
    print(f"{block.metadata.get('type')}: {block.metadata.get('timestamp')}")
```

### Multi-Session Support

Use thread IDs to manage multiple sessions in one artifact:

```python
# Session 1
app.invoke(state, {"configurable": {"thread_id": "user-alice"}})

# Session 2
app.invoke(state, {"configurable": {"thread_id": "user-bob"}})
```

## Architecture

All integrations share a common architecture:

```
Framework Event -> MAIF Callback -> MAIFProvenanceTracker -> MAIFEncoder -> .maif file
```

The base classes in `maif.integrations` provide:

- **EventType**: Standardized event types across frameworks
- **MAIFProvenanceTracker**: Core logging functionality
- **BaseMAIFCallback**: Abstract base for framework callbacks
- **Utility functions**: Safe serialization, timestamp formatting

## Contributing

To add support for a new framework, see the [Integration Plan](../../../maif/integrations/INTEGRATION_PLAN.md) for detailed implementation guidance.

