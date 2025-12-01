# Bedrock Agent Swarm

This example demonstrates how to use multiple AWS Bedrock models in a swarm of agents that share the same MAIF storage. It showcases how different models (Claude, Titan, Jurassic, Command) can collaborate on tasks, including healthcare-related ethical analysis.

## Overview

The Bedrock Agent Swarm uses the `BedrockAgentSwarm` class to manage a collection of agents, each powered by a different Foundation Model from AWS Bedrock.

### Key Features

- **Multi-Model Support**: Integrates Anthropic Claude, Amazon Titan, AI21 Jurassic, and Cohere Command.
- **Shared Storage**: All agents share a common MAIF workspace for persistent memory.
- **Task Aggregation**: Supports various methods to combine results from multiple agents:
  - `vote`: Simple consensus voting.
  - `weighted_vote`: Weighted voting based on provider reliability or specialization.
  - `ensemble`: Combining outputs into a single coherent response.
  - `semantic_merge`: Merging results based on semantic similarity.

## Healthcare Scenario

One of the key scenarios demonstrated is the ethical analysis of AI in healthcare.

```python
{
    "task_id": "task_4",
    "type": "all",
    "data": "What are the ethical implications of using AI in healthcare?",
    "input_type": "text",
    "goal": "analyze ethical implications",
    "aggregation": "weighted_vote",
    "provider_weights": {
        "anthropic": 1.0,  # Higher weight for Claude models on ethical questions
        "amazon": 0.8,
        "ai21": 0.7,
        "cohere": 0.7
    }
}
```

In this scenario, the swarm leverages the strengths of different models, giving higher weight to Anthropic's Claude for its strong reasoning on ethical topics.

## Running the Demo

To run the demo, ensure you have AWS credentials configured with access to Bedrock models.

```bash
python examples/bedrock_swarm_demo.py
```

## Code Structure

The demo initializes a swarm and adds agents with specific models:

```python
# Create agent swarm
swarm = BedrockAgentSwarm(str(workspace))

# Add agents
swarm.add_agent_with_model(
    "claude_agent",
    BedrockModelProvider.ANTHROPIC,
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "us-east-1"
)
# ... add other agents
```

It then submits tasks and processes the results using the specified aggregation strategy.
