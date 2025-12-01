# Hello World Agent

The simplest possible MAIF agent.

## Code

```python
from maif_sdk import create_client, create_artifact

# Create agent with memory
client = create_client("hello-agent")
memory = create_artifact("hello-memory", client)

# Add content with built-in features
memory.add_text("Hello, MAIF world!", encrypt=True)
memory.save("hello.maif", sign=True)

print("✅ Your first AI agent memory is ready!")
```

## Explanation

1. **create_client**: Initializes a MAIF client.
2. **create_artifact**: Creates a new memory artifact.
3. **add_text**: Adds text content, optionally encrypted.
4. **save**: Persists the artifact to disk, optionally signed.
