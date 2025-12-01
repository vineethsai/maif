"""
AWS Backend configuration for MAIF SDK.

This module provides the AWS backend implementations that can be enabled
with use_aws=True in the MAIFClient.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AWSConfig:
    """Configuration for AWS backends."""
    region: str = "us-east-1"
    profile: Optional[str] = None
    
    # S3 Configuration
    s3_bucket: Optional[str] = None
    s3_prefix: str = "maif/"
    s3_encryption: str = "AES256"
    
    # KMS Configuration  
    kms_key_alias: Optional[str] = None
    kms_key_id: Optional[str] = None
    
    # DynamoDB Configuration
    dynamodb_table: Optional[str] = None
    
    # Kinesis Configuration
    kinesis_stream: Optional[str] = None
    
    # CloudWatch Configuration
    cloudwatch_log_group: str = "maif-compliance"
    cloudwatch_retention_days: int = 90
    
    # Secrets Manager Configuration
    secrets_prefix: str = "maif/security/"
    
    # Macie Configuration
    enable_macie: bool = True
    macie_job_prefix: str = "maif-scan-"
    
    @classmethod
    def from_environment(cls) -> 'AWSConfig':
        """Create configuration from environment variables."""
        return cls(
            region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            profile=os.getenv("AWS_PROFILE"),
            s3_bucket=os.getenv("MAIF_AWS_S3_BUCKET"),
            s3_prefix=os.getenv("MAIF_AWS_S3_PREFIX", "maif/"),
            kms_key_alias=os.getenv("MAIF_AWS_KMS_KEY_ALIAS"),
            kms_key_id=os.getenv("MAIF_AWS_KMS_KEY_ID"),
            dynamodb_table=os.getenv("MAIF_AWS_DYNAMODB_TABLE"),
            kinesis_stream=os.getenv("MAIF_AWS_KINESIS_STREAM"),
            cloudwatch_log_group=os.getenv("MAIF_AWS_LOG_GROUP", "maif-compliance"),
            secrets_prefix=os.getenv("MAIF_AWS_SECRETS_PREFIX", "maif/security/")
        )


def create_aws_backends(config: Optional[AWSConfig] = None) -> Dict[str, Any]:
    """
    Create AWS backend instances based on configuration.
    
    Returns a dictionary of backend instances that can be used by MAIFClient.
    """
    if config is None:
        config = AWSConfig.from_environment()
    
    backends = {}
    
    # Try to import AWS backends - these are optional dependencies
    try:
        from maif.aws_s3_integration import MAIFS3Integration, create_s3_integration
    except ImportError:
        MAIFS3Integration = None
        create_s3_integration = None
        
    try:
        from maif.aws_kms_integration import AWSKMSIntegration
    except ImportError:
        AWSKMSIntegration = None
        
    try:
        from maif.aws_secrets_manager_security import AWSSecretsManagerSecurity
    except ImportError:
        AWSSecretsManagerSecurity = None
        
    try:
        from maif.aws_macie_privacy import AWSMaciePrivacy
    except ImportError:
        AWSMaciePrivacy = None
        
    try:
        from maif.aws_cloudwatch_compliance import AWSCloudWatchComplianceLogger
    except ImportError:
        AWSCloudWatchComplianceLogger = None
        
    try:
        from maif.aws_s3_block_storage import AWSS3BlockStorage
    except ImportError:
        AWSS3BlockStorage = None
        
    try:
        from maif.aws_kinesis_streaming import AWSKinesisStreaming
    except ImportError:
        AWSKinesisStreaming = None
    
    # Storage backend
    if config.s3_bucket and create_s3_integration:
        backends['storage'] = create_s3_integration(
            region_name=config.region,
            profile_name=config.profile,
            default_bucket=config.s3_bucket
        )
        
        # Block storage backend
        if AWSS3BlockStorage:
            backends['block_storage'] = AWSS3BlockStorage(
                bucket_name=config.s3_bucket,
                region_name=config.region,
                profile_name=config.profile,
                prefix=config.s3_prefix + "blocks/"
            )
    
    # Security/Encryption backend
    if (config.kms_key_alias or config.kms_key_id) and AWSKMSIntegration:
        backends['encryption'] = AWSKMSIntegration(
            region_name=config.region,
            profile_name=config.profile,
            key_alias=config.kms_key_alias,
            key_id=config.kms_key_id
        )
    
    # Secrets Manager for security
    if AWSSecretsManagerSecurity:
        backends['security'] = AWSSecretsManagerSecurity(
            region_name=config.region,
            secret_prefix=config.secrets_prefix
        )
    
    # Privacy backend with Macie
    if config.enable_macie and AWSMaciePrivacy:
        backends['privacy'] = AWSMaciePrivacy(
            region_name=config.region,
            enable_auto_classification=True
        )
    
    # Compliance logging with CloudWatch
    if AWSCloudWatchComplianceLogger:
        backends['compliance'] = AWSCloudWatchComplianceLogger(
            region_name=config.region,
            log_group_name=config.cloudwatch_log_group,
            retention_days=config.cloudwatch_retention_days
        )
    
    # Streaming with Kinesis
    if config.kinesis_stream and AWSKinesisStreaming:
        backends['streaming'] = AWSKinesisStreaming(
            stream_name=config.kinesis_stream,
            region_name=config.region,
            profile_name=config.profile
        )
    
    return backends