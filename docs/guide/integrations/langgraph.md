# LangGraph Integration

This guide covers integrating MAIF with LangGraph for cryptographic provenance tracking of graph state transitions.

## Overview

The MAIF LangGraph integration provides a drop-in checkpointer that stores all graph state in a MAIF artifact. Every state transition is cryptographically signed and linked, creating a tamper-evident audit trail of your agent's execution.

**Key Benefits:**

- Tamper-evident state history via hash chains
- Ed25519 signatures on all checkpoints
- Full execution audit trail for compliance
- Thread-safe checkpoint management
- Compatible with multi-turn conversations

## Installation

```bash
# Install MAIF with LangGraph integration
pip install maif[integrations]

# Or install LangGraph separately
pip install maif langgraph
```

**Requirements:**

- Python 3.9+
- LangGraph 0.2.0+
- langgraph-checkpoint 1.0.0+

## Quick Start

```python
from langgraph.graph import StateGraph, START, END
from maif.integrations.langgraph import MAIFCheckpointer
from typing import TypedDict, List, Annotated
from operator import add


# Define your state
class MyState(TypedDict):
    messages: Annotated[List[str], add]
    count: int


# Define nodes
def process(state: MyState) -> MyState:
    return {"messages": ["processed"], "count": state["count"] + 1}


# Build graph
graph = StateGraph(MyState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compile with MAIF checkpointer
checkpointer = MAIFCheckpointer("graph_state.maif")
app = graph.compile(checkpointer=checkpointer)

# Run with thread_id for checkpoint tracking
result = app.invoke(
    {"messages": [], "count": 0},
    config={"configurable": {"thread_id": "my-session"}}
)

# Finalize when done (seals the artifact)
checkpointer.finalize()
```

## API Reference

### MAIFCheckpointer

```python
class MAIFCheckpointer(BaseCheckpointSaver):
    def __init__(
        self,
        artifact_path: Union[str, Path],
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the MAIF checkpointer.
        
        Args:
            artifact_path: Path to the MAIF artifact file
            agent_id: Optional identifier for the checkpointer (default: "langgraph_checkpointer")
        """
```

#### Methods

##### get_tuple(config)

Retrieve a checkpoint by configuration.

```python
config = {
    "configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": "checkpoint-001",  # Optional, gets latest if omitted
    }
}

checkpoint_tuple = checkpointer.get_tuple(config)
if checkpoint_tuple:
    print(checkpoint_tuple.checkpoint)
    print(checkpoint_tuple.metadata)
```

##### put(config, checkpoint, metadata, new_versions=None)

Store a checkpoint.

```python
result_config = checkpointer.put(
    config={"configurable": {"thread_id": "my-thread"}},
    checkpoint={"id": "cp-001", "channel_values": {...}},
    metadata={"source": "input", "step": 0},
)
```

##### list(config, filter=None, before=None, limit=None)

List checkpoints matching criteria.

```python
# List all checkpoints for a thread
for cp in checkpointer.list({"configurable": {"thread_id": "my-thread"}}):
    print(cp.checkpoint["id"])

# List with limit
recent = list(checkpointer.list(config, limit=5))
```

##### put_writes(config, writes, task_id)

Store intermediate writes (node outputs before checkpoint).

```python
checkpointer.put_writes(
    config={"configurable": {"thread_id": "my-thread", "checkpoint_id": "cp-001"}},
    writes=[("messages", {"role": "assistant", "content": "Hello"})],
    task_id="node_001",
)
```

##### finalize()

Finalize the MAIF artifact. Call this when you're done using the checkpointer.

```python
checkpointer.finalize()
```

##### get_artifact_path()

Get the path to the MAIF artifact file.

```python
path = checkpointer.get_artifact_path()
print(f"Artifact stored at: {path}")
```

### Async Methods

All methods have async counterparts:

- `aget_tuple(config)` - Async version of `get_tuple`
- `aput(config, checkpoint, metadata)` - Async version of `put`
- `alist(config, ...)` - Async generator version of `list`
- `aput_writes(config, writes, task_id)` - Async version of `put_writes`

```python
# Async usage
checkpoint = await checkpointer.aget_tuple(config)
result = await checkpointer.aput(config, checkpoint, metadata)

async for cp in checkpointer.alist(config):
    print(cp.checkpoint["id"])
```

## Usage Patterns

### Multi-Turn Conversations

```python
from langgraph.graph import StateGraph, START, END
from maif.integrations.langgraph import MAIFCheckpointer

# ... define state and nodes ...

checkpointer = MAIFCheckpointer("conversation.maif")
app = graph.compile(checkpointer=checkpointer)

thread_id = "user-session-123"
config = {"configurable": {"thread_id": thread_id}}

# Turn 1
result1 = app.invoke({"messages": [{"role": "user", "content": "Hi"}]}, config)

# Turn 2 - continues from previous state
result2 = app.invoke({"messages": [{"role": "user", "content": "Tell me more"}]}, config)

# Turn 3
result3 = app.invoke({"messages": [{"role": "user", "content": "Thanks!"}]}, config)

checkpointer.finalize()
```

### Context Manager

```python
with MAIFCheckpointer("session.maif") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    result = app.invoke(initial_state, config)
# Automatically finalized on exit
```

### Resuming from Existing Artifact

```python
# First run
checkpointer1 = MAIFCheckpointer("persistent.maif")
app = graph.compile(checkpointer=checkpointer1)
app.invoke(state, {"configurable": {"thread_id": "session-1"}})
checkpointer1.finalize()

# Later - resume from saved state
checkpointer2 = MAIFCheckpointer("persistent.maif")
# Checkpoints are automatically loaded from the existing artifact

# Get the latest checkpoint
checkpoint = checkpointer2.get_tuple({"configurable": {"thread_id": "session-1"}})
if checkpoint:
    print(f"Resuming from: {checkpoint.checkpoint['id']}")
```

### Inspecting the Audit Trail

```python
from maif import MAIFDecoder

# After running your graph
decoder = MAIFDecoder("graph_state.maif")
decoder.load()

# Verify integrity
is_valid, errors = decoder.verify_integrity()
print(f"Artifact valid: {is_valid}")

# Inspect blocks
for block in decoder.blocks:
    print(f"Block type: {block.metadata.get('type')}")
    print(f"Timestamp: {block.metadata.get('timestamp')}")
```

## Events Logged

The MAIFCheckpointer logs the following event types:

| Event Type | Description |
|------------|-------------|
| `SESSION_START` | Checkpointer initialized |
| `STATE_CHECKPOINT` | Full state checkpoint saved |
| `NODE_END` | Intermediate node writes (via `put_writes`) |
| `SESSION_END` | Checkpointer finalized |

Each event includes:

- Timestamp
- Thread ID
- Checkpoint ID
- Parent checkpoint reference
- Full state data (serialized)
- Cryptographic signature

## Performance Considerations

- **Memory Usage**: Checkpoints are cached in memory for fast lookups. For very long sessions with many checkpoints, consider periodic finalization.

- **Artifact Size**: Each checkpoint stores full state. For large states, monitor artifact size.

- **Thread Safety**: The checkpointer is thread-safe and can be used from multiple threads.

- **Async Performance**: Async methods currently delegate to sync implementations. For high-throughput scenarios, consider batching operations.

## Troubleshooting

### Import Error: LangGraph not installed

```
ImportError: LangGraph is required for MAIFCheckpointer
```

**Solution:** Install LangGraph with `pip install langgraph`

### Checkpoint Not Found

```python
result = checkpointer.get_tuple(config)
# result is None
```

**Possible causes:**

1. Thread ID doesn't match
2. Checkpoint ID doesn't exist
3. Artifact file doesn't exist or is corrupted

**Solution:** Verify thread_id and check artifact exists

### Integrity Check Failed

```python
is_valid, errors = decoder.verify_integrity()
# is_valid is False
```

**Possible causes:**

1. Artifact was modified externally
2. Incomplete write (crash during save)
3. Artifact corrupted

**Solution:** The integrity failure indicates tampering or corruption. Investigate the source and restore from backup if needed.

## Example: RAG Agent with Provenance

See `examples/integrations/langgraph_quickstart.py` for a complete example of using MAIFCheckpointer with a retrieval-augmented generation agent.

## Related

- [MAIF Overview](../getting-started.md)
- [Provenance Tracking](../provenance.md)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)

