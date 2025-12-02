# Distributed Processing with AWS

This example demonstrates how MAIF leverages AWS services for distributed processing and storage, enabling scalable agent architectures.

## Overview

The Distributed Demo shows how to use `AWSDistributedManager` to coordinate tasks across multiple compute nodes or Lambda functions.

### Key Features

- **Distributed Task Execution**: Offload heavy tasks to AWS Lambda or ECS.
- **State Synchronization**: Keep agent state synchronized across distributed nodes using DynamoDB and S3.
- **Scalability**: Scale agent operations horizontally.

## Running the Demo

To run the demo:

```bash
python examples/aws_distributed_demo.py
```

## Code Example

```python
# Initialize Distributed Manager
dist_manager = AWSDistributedManager(
    table_name="maif-state-table",
    bucket_name="maif-storage-bucket"
)

# Distribute a task
task_id = dist_manager.submit_task(
    task_type="process_data",
    payload={"data": "..."}
)

# Retrieve result
result = dist_manager.get_result(task_id)
```
