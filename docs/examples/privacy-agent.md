# Privacy-Enabled Chat Agent

A more realistic agent with memory and privacy.

## Code

```python
from maif_sdk import create_client, create_artifact
from maif import PrivacyLevel, EncryptionMode

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

## Key Concepts

- **Privacy Levels**: Classify data sensitivity (e.g., CONFIDENTIAL).
- **Encryption**: Encrypt data at rest using AES-GCM.
- **Anonymization**: Automatically detect and redact PII.
- **Metadata**: Attach structured metadata for retrieval.
