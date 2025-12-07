# AWS Strands SDK Integration

::: warning Coming Soon
The AWS Strands SDK integration is currently in development. Check back soon for updates.
:::

## Overview

The Strands SDK integration will provide MAIF-backed provenance tracking for AWS Strands agent workflows.

## Planned Features

- **MAIFStrandsCallback**: Track all agent actions and tool invocations
- **MAIFMemoryProvider**: Persistent memory with cryptographic provenance
- **MAIFToolWrapper**: Wrap Strands tools with automatic logging

## Expected Usage

```python
from maif.integrations.strands import MAIFStrandsCallback

# Coming soon
callback = MAIFStrandsCallback("strands_session.maif")
```

## Contributing

Interested in helping build this integration? See the [Integration Plan](../../../maif/integrations/INTEGRATION_PLAN.md) for implementation guidance.

## Related

- [LangGraph Integration](./langgraph.md) - Available now
- [LangChain Integration](./langchain.md) - Available now
- [CrewAI Integration](./crewai.md) - Available now

