# Streaming Data with AWS Kinesis

This example demonstrates how MAIF handles high-throughput streaming data using AWS Kinesis.

## Overview

The Streaming Demo showcases the `KinesisStreamManager`, which allows agents to ingest and process real-time data streams.

### Key Features

- **Real-time Ingestion**: Ingest data from Kinesis streams with low latency.
- **Stream Processing**: Process records as they arrive.
- **Checkpointing**: Reliable processing with checkpointing to DynamoDB.
- **Scalability**: Handle high-volume streams by sharding.

## Running the Demo

To run the demo:

```bash
python examples/aws_kinesis_streaming_demo.py
```

## Code Example

```python
# Initialize Stream Manager
stream_manager = KinesisStreamManager(
    stream_name="maif-data-stream",
    region_name="us-east-1"
)

# Put record
stream_manager.put_record(
    data={"sensor_id": "A1", "value": 98.6},
    partition_key="A1"
)

# Process records
for record in stream_manager.get_records():
    process(record)
```
