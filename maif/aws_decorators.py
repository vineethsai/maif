"""
AWS Decorators for MAIF Agentic Framework
=========================================

Provides simple decorator-based APIs for integrating the MAIF agentic framework
with AWS services like Bedrock, KMS, S3, Lambda, and more.
"""

import functools
import asyncio
import boto3
from typing import Optional, Dict, Any, Callable, List, Type, Union
from pathlib import Path
import json
import time


from .agentic_framework import (
    MAIFAgent, AgentState, PerceptionSystem,
    ReasoningSystem, ExecutionSystem
)

from maif_sdk.artifact import Artifact as MAIFArtifact
from maif_sdk.types import SecurityLevel
from maif_sdk.client import MAIFClient

# Conditional AWS imports
try:
    from .aws_bedrock_integration import BedrockClient, MAIFBedrockIntegration
    from .aws_kms_integration import create_kms_verifier, sign_block_data_with_kms
    from .aws_xray_integration import MAIFXRayIntegration, xray_trace, xray_subsegment
    from .aws_config import AWSConfig, get_aws_config
    from .aws_credentials import configure_aws_credentials
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    # Define dummy classes/functions to prevent ImportErrors
    class BedrockClient:
        def __init__(self, *args, **kwargs): pass
    class MAIFBedrockIntegration:
        def __init__(self, *args, **kwargs): pass
        def generate_text_block(self, *args, **kwargs): return {"text": "dummy", "metadata": {"model": "dummy"}}
        def embed_text(self, *args, **kwargs): return [0.1, 0.2, 0.3]
        def analyze_image(self, *args, **kwargs): return "dummy description"
        def generate_image_block(self, *args, **kwargs): return {"image_data": b"dummy"}
    def create_kms_verifier(*args, **kwargs): raise ImportError("AWS dependencies not installed")
    def sign_block_data_with_kms(*args, **kwargs): raise ImportError("AWS dependencies not installed")
    class MAIFXRayIntegration: 
        def trace_agent_operation(self, *args): return lambda x: x
        def trace_aws_call(self, *args): return lambda x: x
    def xray_trace(*args, **kwargs): return lambda x: x
    def xray_subsegment(*args, **kwargs): return lambda x: x
    class AWSConfig:
        def __init__(self, *args, **kwargs): 
            self.s3_bucket = "dummy-bucket"
            self.region_name = "us-east-1"
            self.region = "us-east-1"
            self.profile = None
            self.s3_prefix = "maif/"
            self.s3_encryption = "AES256"
            self.kms_key_alias = None
            self.kms_key_id = None
            self.dynamodb_table = None
            self.kinesis_stream = None
            self.cloudwatch_log_group = "maif-compliance"
            self.cloudwatch_retention_days = 90
            self.secrets_prefix = "maif/security/"
            self.enable_macie = False
        def get_client(self, *args, **kwargs): 
            mock = MagicMock()
            mock.invoke.return_value = {'Payload': MagicMock(read=lambda: b'{"result": "success"}')}
            return mock
        def get_resource(self, *args, **kwargs): return MagicMock()
    def get_aws_config(*args, **kwargs): return AWSConfig()
    def configure_aws_credentials(*args, **kwargs): raise ImportError("AWS dependencies not installed")
    from unittest.mock import MagicMock



# ===== Enhanced AWS System Implementations =====

class AWSEnhancedPerceptionSystem(PerceptionSystem):
    """Perception system enhanced with AWS Bedrock capabilities."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        super().__init__(agent)
        
        # Initialize Bedrock integration
        bedrock_client = BedrockClient(region_name=region_name)
        self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
    
    @xray_subsegment("perception_process_text")
    async def _process_text(self, text: str, artifact: MAIFArtifact):
        """Process text using AWS Bedrock."""
        # Generate embeddings using Bedrock
        embedding = self.bedrock_integration.embed_text(text)
        
        # Add to artifact
        artifact.add_text(text, title="Text Perception", language="en")
        
        if embedding:
            artifact.custom_metadata.update({
                "embedding": embedding,
                "perception_type": "text",
                "source": "aws_bedrock"
            })
    
    async def _process_image(self, image_data: bytes, artifact: MAIFArtifact):
        """Process image using AWS Bedrock."""
        # Analyze image using Bedrock
        description = self.bedrock_integration.analyze_image(image_data)
        
        # Add to artifact
        artifact.add_image(image_data, title="Image Perception", format="unknown")
        artifact.add_text(description, title="Image Description", language="en")
        
        artifact.custom_metadata["perception_type"] = "image"
        artifact.custom_metadata["source"] = "aws_bedrock"


class AWSEnhancedReasoningSystem(ReasoningSystem):
    """Reasoning system enhanced with AWS Bedrock capabilities."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        super().__init__(agent)
        
        # Initialize Bedrock integration
        bedrock_client = BedrockClient(region_name=region_name)
        self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
    
    async def process(self, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Apply reasoning using AWS Bedrock."""
        # Extract content from context
        texts = []
        for artifact in context:
            for content in artifact.get_content():
                if content['content_type'] == 'text':
                    texts.append(content['data'].decode('utf-8'))
        
        # Generate reasoning using Bedrock
        combined_text = "\n\n".join(texts)
        prompt = f"Analyze the following information and provide insights:\n\n{combined_text}"
        
        reasoning_block = self.bedrock_integration.generate_text_block(prompt)
        
        # Create reasoning artifact
        artifact = MAIFArtifact(
            name=f"reasoning_{int(time.time() * 1000000)}",
            client=self.agent.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        artifact.add_text(
            reasoning_block["text"],
            title="Reasoning Result",
            language="en"
        )
        
        # Add metadata
        artifact.custom_metadata.update({
            "source": "aws_bedrock",
            "model": reasoning_block["metadata"]["model"],
            "timestamp": time.time()
        })
        
        # Save to knowledge base
        artifact.save(self.agent.knowledge_path)
        
        return artifact


class AWSExecutionSystem(ExecutionSystem):
    """Execution system enhanced with AWS service capabilities."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        super().__init__(agent)
        
        # Add AWS-specific executors
        self.executors.update({
            "invoke_lambda": self._invoke_lambda,
            "store_in_s3": self._store_in_s3,
            "query_dynamodb": self._query_dynamodb,
            "generate_image": self._generate_image_with_bedrock
        })
        
        # Initialize AWS clients using centralized config
        self.aws_config = get_aws_config()
        self.lambda_client = self.aws_config.get_client('lambda')
        self.s3_client = self.aws_config.get_client('s3')
        self.dynamodb = self.aws_config.get_resource('dynamodb')
        
        # Initialize Bedrock
        bedrock_client = BedrockClient(region_name=region_name)
        self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
    
    async def _invoke_lambda(self, parameters: Dict) -> Dict:
        """Invoke AWS Lambda function."""
        function_name = parameters.get('function_name')
        payload = parameters.get('payload', {})
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps(payload)
            )
            
            return {
                "status": "success",
                "response": json.loads(response['Payload'].read())
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _store_in_s3(self, parameters: Dict) -> Dict:
        """Store data in S3."""
        bucket = parameters.get('bucket')
        key = parameters.get('key')
        data = parameters.get('data')
        content_type = parameters.get('content_type', 'application/octet-stream')
        
        try:
            import io
            data_bytes = data.encode('utf-8') if isinstance(data, str) else data
            self.s3_client.upload_fileobj(
                io.BytesIO(data_bytes),
                bucket,
                key,
                ExtraArgs={'ContentType': content_type}
            )
            return {
                "status": "success",
                "location": f"s3://{bucket}/{key}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_dynamodb(self, parameters: Dict) -> Dict:
        """Query DynamoDB table."""
        table_name = parameters.get('table_name')
        query_type = parameters.get('query_type', 'scan')
        query_params = parameters.get('query_params', {})
        
        try:
            table = self.dynamodb.Table(table_name)
            
            if query_type == 'get_item':
                response = table.get_item(**query_params)
                result = response.get('Item')
            elif query_type == 'query':
                response = table.query(**query_params)
                result = response.get('Items', [])
            else:  # scan
                response = table.scan(**query_params)
                result = response.get('Items', [])
                
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _generate_image_with_bedrock(self, parameters: Dict) -> Dict:
        """Generate image using Bedrock."""
        prompt = parameters.get('prompt', '')
        width = parameters.get('width', 1024)
        height = parameters.get('height', 1024)
        
        image_block = self.bedrock_integration.generate_image_block(
            prompt, width=width, height=height
        )
        
        return {
            "status": "success",
            "image_data": image_block["image_data"]
        }


# ===== Agent Creation Decorators =====

def maif_agent(agent_class: Type = None, use_aws: bool = False, aws_config: Optional[AWSConfig] = None,
               enable_xray: bool = False, xray_service_name: Optional[str] = None, **config):
    """
    Decorator to easily create a MAIF agent class with optional AWS backend and X-Ray tracing.
    
    @maif_agent(workspace="./agent_data", use_aws=True, enable_xray=True)
    class MyAgent:
        def process(self, data):
            # Your logic here - automatically uses AWS backends and X-Ray tracing
    
    Args:
        use_aws: Enable AWS backends for artifact storage
        aws_config: Optional AWS configuration
        enable_xray: Enable X-Ray tracing for agent operations
        xray_service_name: Custom service name for X-Ray (default: agent class name)
        **config: Additional configuration for the agent
    """
    def decorator(cls):
        # Create a new class that inherits from both the original class and MAIFAgent
        class MixedAgent(cls, MAIFAgent):
            def __init__(self, agent_id="default_agent", workspace_path="./workspace", **kwargs):
                # Initialize MAIFAgent with AWS backend support
                MAIFAgent.__init__(self, agent_id, workspace_path, config)
                
                # Replace the MAIF client with AWS-enabled version if requested
                if use_aws:
                    self.maif_client = MAIFClient(
                        agent_id=agent_id,
                        use_aws=True,
                        aws_config=aws_config
                    )
                
                # Initialize the original class
                cls.__init__(self, **kwargs)
                
            async def run(self):
                # Default implementation that can be overridden
                while self.state != AgentState.TERMINATED:
                    try:
                        # Call the process method from the original class if it exists
                        if hasattr(self, 'process'):
                            await self.process()
                        await asyncio.sleep(5.0)
                    except Exception as e:
                        print(f"Agent error: {e}")
                        await asyncio.sleep(10.0)
        
        return MixedAgent
    
    # Handle case where decorator is used without parentheses
    if agent_class is not None:
        return decorator(agent_class)
    return decorator


# ===== AWS Integration Decorators =====

def aws_bedrock(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS Bedrock capabilities to a method.
    
    @aws_bedrock()
    def generate_text(self, prompt):
        # 'bedrock' is injected into the method
        return bedrock.invoke_text_model("anthropic.claude-v2", prompt)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create Bedrock client if not already available
            if not hasattr(self, '_bedrock_client'):
                self._bedrock_client = BedrockClient(region_name=region_name, profile_name=profile_name)
                self._bedrock_integration = MAIFBedrockIntegration(self._bedrock_client)
            
            # Inject bedrock client into the function call
            kwargs['bedrock'] = self._bedrock_integration
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_kms(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS KMS capabilities to a method.
    
    @aws_kms()
    def sign_data(self, data):
        # 'key_store' and 'verifier' are injected into the method
        return verifier.sign_data(data, key_id)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create KMS verifier if not already available
            if not hasattr(self, '_kms_key_store') or not hasattr(self, '_kms_verifier'):
                self._kms_key_store, self._kms_verifier = create_kms_verifier(
                    region_name=region_name, 
                    profile_name=profile_name
                )
            
            # Inject KMS components into the function call
            kwargs['key_store'] = self._kms_key_store
            kwargs['verifier'] = self._kms_verifier
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_s3(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS S3 capabilities to a method.
    
    @aws_s3()
    def store_artifact(self, artifact, bucket_name):
        # 's3_client' is injected into the method
        s3_client.upload_fileobj(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create S3 client if not already available
            if not hasattr(self, '_s3_client'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._s3_client = session.client('s3')
            
            # Inject S3 client into the function call
            kwargs['s3_client'] = self._s3_client
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_lambda(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS Lambda capabilities to a method.
    
    @aws_lambda()
    def invoke_function(self, function_name, payload):
        # 'lambda_client' is injected into the method
        lambda_client.invoke(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create Lambda client if not already available
            if not hasattr(self, '_lambda_client'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._lambda_client = session.client('lambda')
            
            # Inject Lambda client into the function call
            kwargs['lambda_client'] = self._lambda_client
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_dynamodb(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS DynamoDB capabilities to a method.
    
    @aws_dynamodb()
    def query_table(self, table_name, key):
        # 'dynamodb' is injected into the method
        table = dynamodb.Table(table_name)
        return table.get_item(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create DynamoDB resource if not already available
            if not hasattr(self, '_dynamodb'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._dynamodb = session.resource('dynamodb')
            
            # Inject DynamoDB resource into the function call
            kwargs['dynamodb'] = self._dynamodb
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_step_functions(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS Step Functions capabilities to a method.
    
    @aws_step_functions()
    def execute_workflow(self, state_machine_arn, input_data):
        # 'sfn_client' is injected into the method
        execution = sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps(input_data)
        )
        return execution['executionArn']
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create Step Functions client if not already available
            if not hasattr(self, '_sfn_client'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._sfn_client = session.client('stepfunctions')
            
            # Inject Step Functions client into the function call
            kwargs['sfn_client'] = self._sfn_client
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ===== Agent System Enhancement Decorators =====

def enhance_perception_with_bedrock(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with Bedrock-powered perception.
    
    @enhance_perception_with_bedrock()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Replace perception system with enhanced version
            self.perception = AWSEnhancedPerceptionSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def enhance_reasoning_with_bedrock(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with Bedrock-powered reasoning.
    
    @enhance_reasoning_with_bedrock()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Replace reasoning system with enhanced version
            self.reasoning = AWSEnhancedReasoningSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def enhance_execution_with_aws(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with AWS-powered execution.
    
    @enhance_execution_with_aws()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Replace execution system with enhanced version
            self.execution = AWSExecutionSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# ===== Step Functions Workflow System =====

class StepFunctionsWorkflowSystem:
    """Workflow system that uses AWS Step Functions for orchestration."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        """Initialize Step Functions workflow system."""
        self.agent = agent
        
        # Initialize AWS Step Functions client using centralized config
        self.aws_config = get_aws_config()
        self.sfn_client = self.aws_config.get_client('stepfunctions')
        
        # Store workflow definitions and executions
        self.workflows = {}
        self.executions = {}
    
    def register_workflow(self, name: str, state_machine_arn: str, description: str = ""):
        """Register a Step Functions workflow."""
        self.workflows[name] = {
            "state_machine_arn": state_machine_arn,
            "description": description,
            "registered_at": time.time()
        }
    
    async def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> str:
        """Execute a Step Functions workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not registered")
        
        state_machine_arn = self.workflows[workflow_name]["state_machine_arn"]
        
        # Start execution
        response = self.sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps(input_data)
        )
        
        execution_arn = response["executionArn"]
        
        # Store execution details
        self.executions[execution_arn] = {
            "workflow_name": workflow_name,
            "started_at": time.time(),
            "status": "RUNNING",
            "input": input_data
        }
        
        return execution_arn
    
    async def check_execution_status(self, execution_arn: str) -> Dict[str, Any]:
        """Check the status of a workflow execution."""
        response = self.sfn_client.describe_execution(
            executionArn=execution_arn
        )
        
        # Update execution status
        if execution_arn in self.executions:
            self.executions[execution_arn]["status"] = response["status"]
            
            if "output" in response and response["status"] == "SUCCEEDED":
                self.executions[execution_arn]["output"] = json.loads(response["output"])
        
        return {
            "execution_arn": execution_arn,
            "status": response["status"],
            "started_at": response["startDate"].timestamp(),
            "output": json.loads(response["output"]) if "output" in response else None
        }
    
    async def wait_for_execution(self, execution_arn: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Wait for a workflow execution to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.check_execution_status(execution_arn)
            
            if status["status"] in ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]:
                return status
            
            # Wait before checking again
            await asyncio.sleep(2.0)
        
        # Timeout reached
        return {
            "execution_arn": execution_arn,
            "status": "TIMEOUT",
            "error": f"Execution did not complete within {timeout} seconds"
        }


# ===== Step Functions Enhancement Decorator =====

def enhance_with_step_functions(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with Step Functions workflow capabilities.
    
    @enhance_with_step_functions()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Add workflow system
            self.workflow = StepFunctionsWorkflowSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# ===== Full AWS Agent Decorator =====

def aws_agent(region_name: str = "us-east-1", profile_name: Optional[str] = None,
              use_aws: bool = True, aws_config: Optional[AWSConfig] = None,
              enable_xray: bool = True, xray_service_name: Optional[str] = None, **config):
    """
    Comprehensive decorator to create a fully AWS-integrated MAIF agent.
    
    @aws_agent(workspace="./agent_data")
    class MyAgent:
        def process(self):
            # Your logic here - automatically uses AWS backends and X-Ray tracing
    
    Args:
        region_name: AWS region name (default: us-east-1)
        profile_name: AWS profile name (optional)
        use_aws: Enable AWS backends for artifact storage (default: True)
        aws_config: Optional AWS configuration object
        enable_xray: Enable X-Ray tracing (default: True)
        xray_service_name: Custom service name for X-Ray
        **config: Additional configuration passed to maif_agent
    """
    def decorator(cls):
        # Configure AWS settings if aws_config not provided
        config_to_use = aws_config
        if use_aws and config_to_use is None:
            # Create credential manager with the specified profile
            from .aws_credentials import configure_aws_credentials
            credential_manager = configure_aws_credentials(
                profile_name=profile_name,
                region_name=region_name
            )
            config_to_use = AWSConfig(
                credential_manager=credential_manager,
                default_region=region_name
            )
        
        # First apply the maif_agent decorator with AWS backend support and X-Ray
        agent_cls = maif_agent(
            use_aws=use_aws,
            aws_config=config_to_use,
            enable_xray=enable_xray,
            xray_service_name=xray_service_name,
            **config
        )(cls)
        
        # Then apply all AWS enhancement decorators
        agent_cls = enhance_perception_with_bedrock(region_name)(agent_cls)
        agent_cls = enhance_reasoning_with_bedrock(region_name)(agent_cls)
        agent_cls = enhance_execution_with_aws(region_name)(agent_cls)
        agent_cls = enhance_with_step_functions(region_name)(agent_cls)
        
        return agent_cls
    
    return decorator


# ===== X-Ray Tracing Decorator =====

def aws_xray(service_name: Optional[str] = None, sampling_rate: float = 0.1):
    """
    Decorator to add AWS X-Ray tracing to a method or class.
    
    @aws_xray(service_name="MyService")
    def process_data(data):
        # Your code here - automatically traced
        pass
    
    Args:
        service_name: Optional service name for X-Ray
        sampling_rate: Percentage of requests to trace (0.0-1.0)
    """
    def decorator(target):
        # Initialize X-Ray integration
        xray_integration = MAIFXRayIntegration(
            service_name=service_name or target.__name__,
            sampling_rate=sampling_rate
        )
        
        # If decorating a class
        if isinstance(target, type):
            # Trace all public methods
            for attr_name in dir(target):
                attr = getattr(target, attr_name)
                if callable(attr) and not attr_name.startswith('_'):
                    traced_method = xray_integration.trace_agent_operation(attr_name)(attr)
                    setattr(target, attr_name, traced_method)
            
            # Store X-Ray integration
            target._xray_integration = xray_integration
            return target
        
        # If decorating a function/method
        else:
            return xray_integration.trace_agent_operation(target.__name__)(target)
    
    return decorator


# Convenience decorator for tracing AWS service calls
def trace_aws_call(service_name: str):
    """
    Decorator to trace AWS service calls with X-Ray.
    
    @trace_aws_call("s3")
    async def upload_to_s3(bucket, key, data):
        # S3 upload code
        pass
    """
    xray_integration = MAIFXRayIntegration()
    return xray_integration.trace_aws_call(service_name)