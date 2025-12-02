# AWS Macie Privacy Integration

This example demonstrates how MAIF integrates with AWS Macie to provide advanced privacy protection and PII (Personally Identifiable Information) detection.

## Overview

The Privacy Demo showcases the `MaciePrivacyEngine`, which uses AWS Macie to scan content for sensitive information before it is stored or processed.

### Key Features

- **PII Detection**: Automatically detects sensitive data like names, addresses, credit card numbers, etc.
- **Privacy Levels**: Supports different privacy levels (e.g., PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED).
- **Anonymization**: Can redact or mask sensitive information based on policy.
- **Compliance Logging**: Logs privacy violations and actions for audit purposes.

## Running the Demo

To run the demo, ensure you have AWS credentials configured and Macie enabled in your region (optional, but recommended for full functionality).

```bash
python examples/aws_macie_privacy_demo.py
```

## Code Example

```python
# Initialize Privacy Engine
privacy_engine = MaciePrivacyEngine(
    region_name="us-east-1",
    enable_automated_discovery=True
)

# Create Privacy Policy
policy = PrivacyPolicy(
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES256,
    anonymization_required=True
)

# Process Content
content = "My credit card number is 4532-xxxx-xxxx-xxxx"
processed_content = privacy_engine.process_content(content, policy)
```
