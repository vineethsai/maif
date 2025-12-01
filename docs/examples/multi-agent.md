# Multi-Agent Consortium

This example demonstrates how multiple specialized agents can collaborate using MAIF to produce a comprehensive artifact with full version history, content tracking, and forensic analysis capabilities.

## Overview

The Multi-Agent Consortium example simulates a complex planning scenario: "How do I walk from California to Nepal in a meaningful way - where I have infinite ability to swim, and don't need to sleep".

### Participating Agents

The consortium includes:
1. **GeographyAgent**: Analyzes terrain and routes.
2. **CulturalAgent**: Provides cultural insights and meaningful experiences.
3. **LogisticsAgent**: Handles practical considerations.
4. **SafetyAgent**: Assesses risks and safety measures.
5. **CoordinatorAgent**: Orchestrates the collaboration and synthesizes results.

### Enhanced Features

- **Version History Tracking**: All content changes are tracked with full version history.
- **Content Evolution**: Iterative refinement of contributions based on feedback.
- **Cross-Agent Dependencies**: Management of dependencies between different agents' outputs.
- **Forensic Analysis**: Analysis of collaboration patterns and contribution history.
- **Privacy & Security**: Granular privacy controls and security verifications.
- **Semantic Embeddings**: Searchability through semantic understanding.

## Running the Demo

To run the demo:

```bash
python examples/multi_agent_consortium_demo.py
```

## Implementation Details

Each agent is implemented as a subclass of `BaseAgent`, which handles MAIF integration:

```python
class BaseAgent:
    def __init__(self, agent_id, agent_type, specialization, shared_maif=None):
        # ... initialization ...
        self.maif = shared_maif if shared_maif is not None else create_maif(agent_id, enable_privacy=True)

    def contribute(self, query, context=None):
        # ... generate contribution ...
        self._store_contribution(contribution)
        return contribution
```

The agents collaborate by sharing a `shared_maif` instance or by exchanging data through the coordinator.
