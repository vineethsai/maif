# Examples

Real-world examples demonstrating MAIF capabilities.

## Featured Example

### LangGraph Multi-Agent RAG System

Production-ready multi-agent research assistant with cryptographic provenance.

**Location**: `examples/langgraph/`

**Features**:
- Five specialized agents (Init, Retrieve, Synthesize, Fact-Check, Citation)
- Real ChromaDB vector search with 384-dim embeddings
- Gemini API for generation and verification
- LLM-based fact-checking with iterative refinement
- Complete audit trail in MAIF artifacts
- Interactive console interface
- Multi-turn conversation support

**Quick Start**:
```bash
cd examples/langgraph
echo "GEMINI_API_KEY=your_key" > .env
pip install -r requirements_enhanced.txt
python3 create_kb_enhanced.py
python3 demo_enhanced.py
```

**Documentation**: See [LangGraph RAG Guide](./langgraph-rag.md) for complete details.

---

## Available Examples

### Hello World
**[Get Started →](./hello-world.md)**

The simplest possible MAIF agent. Perfect for understanding the basics.

**What you'll learn:**
- Creating MAIF artifacts
- Adding text blocks
- Saving and loading
- Basic verification

**Time**: 5 minutes

---

### Multi-Agent System
**[View Example →](./multi-agent.md)**

Multiple agents collaborating through shared MAIF artifacts.

**What you'll learn:**
- Agent coordination
- Shared memory patterns
- Provenance tracking
- Multi-agent workflows

**Time**: 15 minutes

---

### Privacy & Security
**[View Example →](./privacy-demo.md)**

Privacy-preserving agent with encryption and anonymization.

**What you'll learn:**
- AES-GCM encryption
- Differential privacy
- Data anonymization
- Access control

**Time**: 10 minutes

---

### Streaming Data
**[View Example →](./streaming.md)**

High-throughput streaming with memory-mapped I/O.

**What you'll learn:**
- Streaming operations
- Memory-mapped I/O
- Performance optimization
- Large file handling

**Time**: 15 minutes

---

### Financial Agent
**[View Example →](./financial-agent.md)**

Privacy-compliant financial transaction analysis.

**What you'll learn:**
- Regulatory compliance
- Transaction analysis
- Audit trails
- Risk scoring

**Time**: 20 minutes

---

### Distributed Processing
**[View Example →](./distributed.md)**

Distributed agent systems with MAIF synchronization.

**What you'll learn:**
- Distributed coordination
- State synchronization
- Network protocols
- Fault tolerance

**Time**: 25 minutes

---

## Quick Start Examples

### Hello World Agent (30 seconds)

The simplest possible MAIF agent:

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

### Privacy-Enabled Chat Agent (2 minutes)

A more realistic agent with memory and privacy:

```python
from maif_sdk import create_client, create_artifact
from maif.privacy import PrivacyLevel, EncryptionMode

class PrivateChatAgent:
    def __init__(self, agent_id: str):
        self.client = create_client(agent_id, enable_privacy=True)
        self.memory = create_artifact(f"{agent_id}-chat", self.client)
    
    def chat(self, message: str, user_id: str) -> str:
        # Store message with privacy protection
        self.memory.add_text(
            message,
            title="User Message",
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            encryption_mode=EncryptionMode.AES_GCM,
            anonymize=True,  # Remove PII automatically
            metadata={"user_id": user_id, "type": "user_input"}
        )
        
        # Search for relevant context
        context = self.memory.search(message, top_k=3)
        
        # Generate response (integrate your LLM here)
        response = f"I understand you're asking about: {message}"
        
        # Store response with same privacy level
        self.memory.add_text(
            response,
            title="Agent Response", 
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            metadata={"user_id": user_id, "type": "agent_response"}
        )
        
        return response

# Usage
agent = PrivateChatAgent("support-bot")
response = agent.chat("How do I reset my password?", "user123")
print(response)
```

## Example Categories

### By Experience Level

**Beginner:**
- [Hello World](./hello-world.md) - Your first MAIF agent
- [Privacy Demo](./privacy-demo.md) - Basic privacy features

**Intermediate:**
- [Multi-Agent](./multi-agent.md) - Agent coordination
- [Streaming](./streaming.md) - High-performance I/O
- [Financial Agent](./financial-agent.md) - Production patterns

**Advanced:**
- [LangGraph RAG](./langgraph-rag.md) - Complete multi-agent system
- [Distributed](./distributed.md) - Distributed systems

### By Use Case

**AI/ML Applications:**
- [LangGraph RAG](./langgraph-rag.md) - Research assistant with fact-checking
- [Multi-Agent](./multi-agent.md) - Collaborative agents

**Enterprise:**
- [Financial Agent](./financial-agent.md) - Regulatory compliance
- [Privacy Demo](./privacy-demo.md) - Data protection

**Performance:**
- [Streaming](./streaming.md) - High throughput
- [Distributed](./distributed.md) - Scale-out architecture

## Running the Examples

All examples follow the same pattern:

```bash
# 1. Navigate to repository root
cd /path/to/maifscratch-1

# 2. Install dependencies (if needed)
pip install -e .

# 3. Run the example
python3 examples/<category>/<example_file>.py
```

For the LangGraph example:
```bash
cd examples/langgraph
pip install -r requirements_enhanced.txt
python3 demo_enhanced.py
```

## Example Structure

Each example includes:
- ✅ **Complete, runnable code**
- ✅ **Comprehensive error handling**
- ✅ **Performance optimizations**
- ✅ **Security best practices**
- ✅ **Testing and validation**
- ✅ **Detailed documentation**

## Contributing Examples

Have a great example to share? We welcome contributions!

1. Create your example in `examples/<category>/`
2. Add documentation in `docs/examples/`
3. Include README with usage instructions
4. Submit a pull request

## Support

- **Documentation**: See the [User Guide](../guide/)
- **API Reference**: Check the [API docs](../api/)
- **Issues**: Report problems on [GitHub](https://github.com/vineethsai/maifscratch-1/issues)

---

*Every example is designed to be production-ready. Copy, modify, and deploy with confidence.*
