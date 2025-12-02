"""
AWS S3 Integration for MAIF
==========================

Provides integration with AWS S3 for secure and efficient object storage operations.
Uses centralized credential management for consistent authentication across services.
"""

import json
import time
import logging
import datetime
import io
import hashlib
import os
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO
import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError
import concurrent.futures

# Import centralized credential manager
from .aws_credentials import get_credential_manager, AWSCredentialManager

# Configure logger
logger = logging.getLogger(__name__)


class S3Error(Exception):
    """Base exception for S3 integration errors."""
    pass


class S3ConnectionError(S3Error):
    """Exception for S3 connection errors."""
    pass


class S3ThrottlingError(S3Error):
    """Exception for S3 throttling errors."""
    pass


class S3ValidationError(S3Error):
    """Exception for S3 validation errors."""
    pass


class S3PermissionError(S3Error):
    """Exception for S3 permission errors."""
    pass


class S3Client:
    """Client for AWS S3 service with production-ready features."""
    
    # Set of known transient errors that can be retried
    RETRYABLE_ERRORS = {
        'ThrottlingException', 
        'Throttling', 
        'RequestLimitExceeded',
        'TooManyRequestsException',
        'ServiceUnavailable',
        'InternalServerError',
        'InternalFailure',
        'ServiceFailure',
        'SlowDown',
        'RequestTimeout'
    }
    
    def __init__(self,
                 credential_manager: Optional[AWSCredentialManager] = None,
                 region_name: Optional[str] = None,
                 max_retries: int = 3,
                 base_delay: float = 0.5,
                 max_delay: float = 5.0,
                 max_pool_connections: int = 10):
        """
        Initialize S3 client with centralized credential management.
        
        Args:
            credential_manager: Optional credential manager instance (uses global if not provided)
            region_name: AWS region name (overrides credential manager's region if provided)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            max_pool_connections: Maximum number of connections in the connection pool
            
        Raises:
            S3ConnectionError: If unable to initialize the S3 client
        """
        # Use provided credential manager or get global instance
        self.credential_manager = credential_manager or get_credential_manager()
        
        # Use provided region or get from credential manager
        self.region_name = region_name or self.credential_manager.region_name
        
        # Validate inputs
        if max_retries < 0:
            raise S3ValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise S3ValidationError("base_delay and max_delay must be positive")
        
        if max_pool_connections <= 0:
            raise S3ValidationError("max_pool_connections must be positive")
        
        # Initialize AWS clients using credential manager
        try:
            logger.info(f"Initializing S3 client in region {self.region_name}")
            
            # Get credentials info for logging
            cred_info = self.credential_manager.get_credential_info()
            logger.info(f"Using credentials from source: {cred_info['source']}")
            
            # Configure S3 client with connection pooling
            self.s3_client = self.credential_manager.get_client(
                's3',
                config=boto3.session.Config(
                    max_pool_connections=max_pool_connections,
                    retries={'max_attempts': 0}  # We'll handle retries ourselves
                )
            )
            
            # For resource-based operations
            self.s3_resource = self.credential_manager.get_resource('s3')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Thread pool for parallel operations
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_pool_connections)
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "bytes_uploaded": 0,
                "bytes_downloaded": 0,
                "multipart_uploads": 0
            }
            
            # Thread safety
            self._lock = threading.RLock()
            
            logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}", exc_info=True)
            raise S3ConnectionError(f"Failed to initialize S3 client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            S3Error: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        with self._lock:
            self.metrics["total_operations"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing S3 {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                with self._lock:
                    self.metrics["successful_operations"] += 1
                    latency = time.time() - start_time
                    self.metrics["operation_latencies"].append(latency)
                    
                    # Trim latencies list if it gets too large
                    if len(self.metrics["operation_latencies"]) > 1000:
                        self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"S3 {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    with self._lock:
                        self.metrics["retried_operations"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"S3 {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    with self._lock:
                        self.metrics["failed_operations"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"S3 {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise S3ThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDenied', 'NoSuchBucketPolicy'):
                        logger.error(
                            f"S3 {operation_name} failed due to permission error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise S3PermissionError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"S3 {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise S3Error(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    with self._lock:
                        self.metrics["retried_operations"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"S3 {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    with self._lock:
                        self.metrics["failed_operations"] += 1
                    logger.error(f"S3 {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise S3ConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                with self._lock:
                    self.metrics["failed_operations"] += 1
                logger.error(f"Unexpected error in S3 {operation_name}: {str(e)}", exc_info=True)
                raise S3Error(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise S3Error(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise S3Error(f"Operation failed after {retries} retries")
    
    def list_buckets(self) -> List[Dict[str, Any]]:
        """
        List all S3 buckets.
        
        Returns:
            List of bucket information dictionaries
            
        Raises:
            S3Error: If an error occurs while listing buckets
        """
        logger.info("Listing S3 buckets")
        
        def list_buckets_operation():
            response = self.s3_client.list_buckets()
            return response.get('Buckets', [])
        
        buckets = self._execute_with_retry("list_buckets", list_buckets_operation)
        logger.info(f"Found {len(buckets)} S3 buckets")
        return buckets
    
    def list_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in an S3 bucket.
        
        Args:
            bucket: S3 bucket name
            prefix: Object key prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object information dictionaries
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while listing objects
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if max_keys <= 0:
            raise S3ValidationError("max_keys must be positive")
        
        logger.info(f"Listing objects in bucket {bucket} with prefix '{prefix}'")
        
        def list_objects_operation():
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            objects = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys):
                objects.extend(page.get('Contents', []))
                
                # Stop if we've reached max_keys
                if len(objects) >= max_keys:
                    objects = objects[:max_keys]
                    break
            
            return objects
        
        objects = self._execute_with_retry("list_objects", list_objects_operation)
        logger.info(f"Found {len(objects)} objects in bucket {bucket} with prefix '{prefix}'")
        return objects
    
    def upload_file(self, file_path: str, bucket: str, key: str, 
                   extra_args: Optional[Dict[str, Any]] = None,
                   use_multipart: bool = True,
                   part_size: int = 8 * 1024 * 1024) -> bool:
        """
        Upload a file to S3.
        
        Args:
            file_path: Path to the file to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Extra arguments to pass to S3 (e.g., ContentType, Metadata)
            use_multipart: Whether to use multipart upload for large files
            part_size: Size of each part in bytes for multipart uploads
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs during upload
        """
        # Validate inputs
        if not file_path:
            raise S3ValidationError("file_path cannot be empty")
        
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        if part_size <= 0:
            raise S3ValidationError("part_size must be positive")
        
        logger.info(f"Uploading file {file_path} to s3://{bucket}/{key}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise S3ValidationError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        
        # Use multipart upload for large files
        if use_multipart and file_size > part_size:
            logger.info(f"Using multipart upload for file {file_path} ({file_size} bytes)")
            with self._lock:
                self.metrics["multipart_uploads"] += 1
            
            def multipart_upload_operation():
                transfer_config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=part_size,
                    max_concurrency=10,
                    multipart_chunksize=part_size,
                    use_threads=True
                )
                
                self.s3_client.upload_file(
                    file_path, 
                    bucket, 
                    key, 
                    ExtraArgs=extra_args,
                    Config=transfer_config
                )
                return True
            
            result = self._execute_with_retry("multipart_upload", multipart_upload_operation)
            with self._lock:
                self.metrics["bytes_uploaded"] += file_size
            return result
        else:
            # Use standard upload for smaller files
            def upload_operation():
                self.s3_client.upload_file(
                    file_path, 
                    bucket, 
                    key, 
                    ExtraArgs=extra_args
                )
                return True
            
            result = self._execute_with_retry("upload_file", upload_operation)
            with self._lock:
                self.metrics["bytes_uploaded"] += file_size
            return result
    
    def upload_fileobj(self, fileobj: BinaryIO, bucket: str, key: str, 
                      extra_args: Optional[Dict[str, Any]] = None,
                      use_multipart: bool = True,
                      part_size: int = 8 * 1024 * 1024) -> bool:
        """
        Upload a file-like object to S3.
        
        Args:
            fileobj: File-like object to upload
            bucket: S3 bucket name
            key: S3 object key
            extra_args: Extra arguments to pass to S3 (e.g., ContentType, Metadata)
            use_multipart: Whether to use multipart upload for large files
            part_size: Size of each part in bytes for multipart uploads
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs during upload
        """
        # Validate inputs
        if fileobj is None:
            raise S3ValidationError("fileobj cannot be None")
        
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        if part_size <= 0:
            raise S3ValidationError("part_size must be positive")
        
        logger.info(f"Uploading file object to s3://{bucket}/{key}")
        
        # Determine if we should use multipart upload
        if use_multipart:
            logger.info(f"Using multipart upload for file object to s3://{bucket}/{key}")
            with self._lock:
                self.metrics["multipart_uploads"] += 1
            
            def multipart_upload_operation():
                transfer_config = boto3.s3.transfer.TransferConfig(
                    multipart_threshold=part_size,
                    max_concurrency=10,
                    multipart_chunksize=part_size,
                    use_threads=True
                )
                
                self.s3_client.upload_fileobj(
                    fileobj, 
                    bucket, 
                    key, 
                    ExtraArgs=extra_args,
                    Config=transfer_config
                )
                return True
            
            return self._execute_with_retry("multipart_upload_fileobj", multipart_upload_operation)
        else:
            # Use standard upload
            def upload_operation():
                self.s3_client.upload_fileobj(
                    fileobj, 
                    bucket, 
                    key, 
                    ExtraArgs=extra_args
                )
                return True
            
            return self._execute_with_retry("upload_fileobj", upload_operation)
    
    def download_file(self, bucket: str, key: str, file_path: str) -> bool:
        """
        Download a file from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            file_path: Path to save the downloaded file
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs during download
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        if not file_path:
            raise S3ValidationError("file_path cannot be empty")
        
        logger.info(f"Downloading s3://{bucket}/{key} to {file_path}")
        
        def download_operation():
            self.s3_client.download_file(bucket, key, file_path)
            return True
        
        result = self._execute_with_retry("download_file", download_operation)
        
        # Update metrics
        if result:
            try:
                file_size = os.path.getsize(file_path)
                with self._lock:
                    self.metrics["bytes_downloaded"] += file_size
            except Exception:
                pass
        
        return result
    
    def download_fileobj(self, bucket: str, key: str, fileobj: BinaryIO) -> bool:
        """
        Download a file from S3 to a file-like object.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            fileobj: File-like object to write to
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs during download
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        if fileobj is None:
            raise S3ValidationError("fileobj cannot be None")
        
        logger.info(f"Downloading s3://{bucket}/{key} to file object")
        
        def download_operation():
            self.s3_client.download_fileobj(bucket, key, fileobj)
            return True
        
        return self._execute_with_retry("download_fileobj", download_operation)
    
    def get_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Get an object from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Object data and metadata
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while getting the object
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        logger.info(f"Getting object s3://{bucket}/{key}")
        
        def get_object_operation():
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Read the body
            body = response['Body'].read()
            
            # Update metrics
            with self._lock:
                self.metrics["bytes_downloaded"] += len(body)
            
            # Create result with metadata and body
            result = {k: v for k, v in response.items() if k != 'Body'}
            result['Body'] = body
            
            return result
        
        return self._execute_with_retry("get_object", get_object_operation)
    
    def put_object(self, bucket: str, key: str, body: Union[bytes, str, BinaryIO],
                  metadata: Optional[Dict[str, str]] = None,
                  content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Put an object in S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            body: Object data
            metadata: Object metadata
            content_type: Content type of the object
            
        Returns:
            Response from S3
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while putting the object
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        if body is None:
            raise S3ValidationError("body cannot be None")
        
        logger.info(f"Putting object s3://{bucket}/{key}")
        
        # Convert string to bytes if necessary
        if isinstance(body, str):
            body = body.encode('utf-8')
        
        # Prepare arguments
        put_args = {
            'Bucket': bucket,
            'Key': key,
            'Body': body
        }
        
        if metadata:
            put_args['Metadata'] = metadata
        
        if content_type:
            put_args['ContentType'] = content_type
        
        def put_object_operation():
            response = self.s3_client.put_object(**put_args)
            
            # Update metrics
            if isinstance(body, bytes):
                with self._lock:
                    self.metrics["bytes_uploaded"] += len(body)
            
            return response
        
        return self._execute_with_retry("put_object", put_object_operation)
    
    def delete_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Delete an object from S3.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Response from S3
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while deleting the object
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        logger.info(f"Deleting object s3://{bucket}/{key}")
        
        def delete_object_operation():
            return self.s3_client.delete_object(Bucket=bucket, Key=key)
        
        return self._execute_with_retry("delete_object", delete_object_operation)
    
    def copy_object(self, source_bucket: str, source_key: str, 
                   dest_bucket: str, dest_key: str,
                   metadata: Optional[Dict[str, str]] = None,
                   metadata_directive: str = 'COPY') -> Dict[str, Any]:
        """
        Copy an object in S3.
        
        Args:
            source_bucket: Source S3 bucket name
            source_key: Source S3 object key
            dest_bucket: Destination S3 bucket name
            dest_key: Destination S3 object key
            metadata: New metadata to apply (if metadata_directive is 'REPLACE')
            metadata_directive: Whether to copy or replace metadata ('COPY' or 'REPLACE')
            
        Returns:
            Response from S3
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while copying the object
        """
        # Validate inputs
        if not source_bucket:
            raise S3ValidationError("source_bucket cannot be empty")
        
        if not source_key:
            raise S3ValidationError("source_key cannot be empty")
        
        if not dest_bucket:
            raise S3ValidationError("dest_bucket cannot be empty")
        
        if not dest_key:
            raise S3ValidationError("dest_key cannot be empty")
        
        if metadata_directive not in ('COPY', 'REPLACE'):
            raise S3ValidationError("metadata_directive must be 'COPY' or 'REPLACE'")
        
        logger.info(f"Copying object s3://{source_bucket}/{source_key} to s3://{dest_bucket}/{dest_key}")
        
        # Prepare arguments
        copy_args = {
            'Bucket': dest_bucket,
            'Key': dest_key,
            'CopySource': f'{source_bucket}/{source_key}',
            'MetadataDirective': metadata_directive
        }
        
        if metadata and metadata_directive == 'REPLACE':
            copy_args['Metadata'] = metadata
        
        def copy_object_operation():
            return self.s3_client.copy_object(**copy_args)
        
        return self._execute_with_retry("copy_object", copy_object_operation)
    
    def generate_presigned_url(self, bucket: str, key: str, 
                              expiration: int = 3600,
                              http_method: str = 'GET') -> str:
        """
        Generate a presigned URL for an S3 object.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            expiration: Time in seconds until the URL expires
            http_method: HTTP method to allow ('GET', 'PUT', etc.)
            
        Returns:
            Presigned URL
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while generating the URL
        """
        # Validate inputs
        if not bucket:
            raise S3ValidationError("bucket cannot be empty")
        
        if not key:
            raise S3ValidationError("key cannot be empty")
        
        if expiration <= 0:
            raise S3ValidationError("expiration must be positive")
        
        valid_methods = ('GET', 'PUT', 'HEAD', 'DELETE', 'POST')
        if http_method not in valid_methods:
            raise S3ValidationError(f"http_method must be one of {valid_methods}")
        
        logger.info(f"Generating presigned URL for s3://{bucket}/{key} (method: {http_method}, expiration: {expiration}s)")
        
        def generate_url_operation():
            # Map HTTP methods to S3 operations
            method_map = {
                'GET': 'get_object',
                'PUT': 'put_object',
                'HEAD': 'head_object',
                'DELETE': 'delete_object',
                'POST': 'post_object'
            }
            
            client_method = method_map.get(http_method, http_method.lower() + '_object')
            
            return self.s3_client.generate_presigned_url(
                client_method,
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
        
        return self._execute_with_retry("generate_presigned_url", generate_url_operation)
    
    def close(self):
        """Close the client and release resources."""
        logger.info("Closing S3 client")
        try:
            self.thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error closing thread pool: {e}")


class MAIFS3Integration:
    """Integration between MAIF and AWS S3 for artifact storage."""
    
    def __init__(self, s3_client: S3Client, default_bucket: Optional[str] = None):
        """
        Initialize MAIF S3 integration.
        
        Args:
            s3_client: S3 client
            default_bucket: Default S3 bucket for operations
            
        Raises:
            S3ValidationError: If s3_client is None
        """
        if s3_client is None:
            raise S3ValidationError("s3_client cannot be None")
        
        logger.info("Initializing MAIF S3 integration")
        self.s3_client = s3_client
        self.default_bucket = default_bucket
        
        # Metrics
        self.metrics = {
            "artifacts_stored": 0,
            "artifacts_retrieved": 0,
            "bytes_stored": 0,
            "bytes_retrieved": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("MAIF S3 integration initialized successfully")
    
    def store_artifact(self, artifact, bucket: Optional[str] = None, 
                      prefix: str = "artifacts/",
                      include_metadata: bool = True) -> str:
        """
        Store a MAIF artifact in S3.
        
        Args:
            artifact: MAIF artifact to store
            bucket: S3 bucket name (uses default_bucket if not provided)
            prefix: Key prefix for the artifact
            include_metadata: Whether to include artifact metadata
            
        Returns:
            S3 URI of the stored artifact
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while storing the artifact
        """
        # Validate inputs
        if artifact is None:
            raise S3ValidationError("artifact cannot be None")
        
        bucket_name = bucket or self.default_bucket
        if not bucket_name:
            raise S3ValidationError("bucket must be provided or default_bucket must be set")
        
        logger.info(f"Storing artifact {artifact.name} in S3 bucket {bucket_name}")
        
        # Save artifact to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.maif') as temp_file:
            artifact_path = artifact.save(temp_file.name)
            
            # Generate S3 key
            key = f"{prefix.rstrip('/')}/{artifact.name}.maif"
            
            # Prepare metadata
            metadata = {}
            if include_metadata:
                # Add basic metadata
                metadata = {
                    "artifact_name": artifact.name,
                    "artifact_type": "maif",
                    "content_count": str(len(artifact.get_content())),
                    "timestamp": str(int(time.time())),
                    "security_level": str(artifact.security_level.value)
                }
                
                # Add custom metadata (limited to what S3 allows)
                for k, v in artifact.custom_metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[f"custom_{k}"] = str(v)
            
            # Upload to S3
            extra_args = {
                'Metadata': metadata,
                'ContentType': 'application/octet-stream'
            }
            
            self.s3_client.upload_file(
                artifact_path,
                bucket_name,
                key,
                extra_args=extra_args
            )
            
            # Update metrics
            file_size = os.path.getsize(artifact_path)
            with self._lock:
                self.metrics["artifacts_stored"] += 1
                self.metrics["bytes_stored"] += file_size
            
            # Return S3 URI
            return f"s3://{bucket_name}/{key}"
    
    def retrieve_artifact(self, s3_uri: str, client=None) -> Any:
        """
        Retrieve a MAIF artifact from S3.
        
        Args:
            s3_uri: S3 URI of the artifact (s3://bucket/key)
            client: MAIF client to use for creating the artifact
            
        Returns:
            Retrieved MAIF artifact
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while retrieving the artifact
        """
        # Validate inputs
        if not s3_uri:
            raise S3ValidationError("s3_uri cannot be empty")
        
        # Parse S3 URI
        if not s3_uri.startswith("s3://"):
            raise S3ValidationError("s3_uri must start with 's3://'")
        
        path_parts = s3_uri[5:].split('/', 1)
        if len(path_parts) != 2:
            raise S3ValidationError("s3_uri must be in the format 's3://bucket/key'")
        
        bucket = path_parts[0]
        key = path_parts[1]
        
        logger.info(f"Retrieving artifact from {s3_uri}")
        
        # Download to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.maif') as temp_file:
            self.s3_client.download_file(bucket, key, temp_file.name)
            
            # Update metrics
            file_size = os.path.getsize(temp_file.name)
            with self._lock:
                self.metrics["artifacts_retrieved"] += 1
                self.metrics["bytes_retrieved"] += file_size
            
            # Load artifact
            from maif_sdk.artifact import Artifact
            artifact = Artifact.load(temp_file.name, client=client)
            
            return artifact
    
    def list_artifacts(self, bucket: Optional[str] = None,
                      prefix: str = "artifacts/",
                      max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List MAIF artifacts in S3.
        
        Args:
            bucket: S3 bucket name (uses default_bucket if not provided)
            prefix: Key prefix for artifacts
            max_keys: Maximum number of keys to return
            
        Returns:
            List of artifact information dictionaries
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while listing artifacts
        """
        bucket_name = bucket or self.default_bucket
        if not bucket_name:
            raise S3ValidationError("bucket must be provided or default_bucket must be set")
        
        logger.info(f"Listing artifacts in bucket {bucket_name} with prefix '{prefix}'")
        
        # List objects with the given prefix
        objects = self.s3_client.list_objects(bucket_name, prefix, max_keys)
        
        # Filter for MAIF artifacts
        artifacts = []
        for obj in objects:
            if obj['Key'].endswith('.maif'):
                # Get object metadata
                try:
                    # Get object metadata using the S3 client's internal boto3 client
                    # Note: self.s3_client is our S3Client wrapper, self.s3_client.s3_client is the boto3 client
                    response = self.s3_client.s3_client.head_object(
                        Bucket=bucket_name,
                        Key=obj['Key']
                    )
                    metadata = response.get('Metadata', {})
                    
                    # Create artifact info
                    artifact_info = {
                        "s3_uri": f"s3://{bucket_name}/{obj['Key']}",
                        "key": obj['Key'],
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat(),
                        "metadata": metadata
                    }
                    
                    artifacts.append(artifact_info)
                except ClientError as e:
                    if e.response.get('Error', {}).get('Code') == 'NotFound':
                        logger.warning(f"Object not found when getting metadata for {obj['Key']}")
                    else:
                        logger.error(f"Error getting metadata for {obj['Key']}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error getting metadata for {obj['Key']}: {e}")
        
        logger.info(f"Found {len(artifacts)} MAIF artifacts")
        return artifacts
    
    def delete_artifact(self, s3_uri: str) -> bool:
        """
        Delete a MAIF artifact from S3.
        
        Args:
            s3_uri: S3 URI of the artifact (s3://bucket/key)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            S3ValidationError: If input validation fails
            S3Error: If an error occurs while deleting the artifact
        """
        # Validate inputs
        if not s3_uri:
            raise S3ValidationError("s3_uri cannot be empty")
        
        # Parse S3 URI
        if not s3_uri.startswith("s3://"):
            raise S3ValidationError("s3_uri must start with 's3://'")
        
        path_parts = s3_uri[5:].split('/', 1)
        if len(path_parts) != 2:
            raise S3ValidationError("s3_uri must be in the format 's3://bucket/key'")
        
        bucket = path_parts[0]
        key = path_parts[1]
        
        logger.info(f"Deleting artifact from {s3_uri}")
        
        # Delete object
        self.s3_client.delete_object(bucket, key)
        return True


# Helper functions for easy integration
def create_s3_integration(region_name: str = "us-east-1",
                         profile_name: Optional[str] = None,
                         default_bucket: Optional[str] = None,
                         max_retries: int = 3) -> MAIFS3Integration:
    """
    Create MAIF S3 integration.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name (optional)
        default_bucket: Default S3 bucket for operations
        max_retries: Maximum number of retries for transient errors
        
    Returns:
        MAIFS3Integration
        
    Raises:
        S3ConnectionError: If unable to initialize the S3 client
    """
    logger.info(f"Creating S3 integration in region {region_name}")
    
    try:
        s3_client = S3Client(
            region_name=region_name,
            max_retries=max_retries
        )
        
        integration = MAIFS3Integration(
            s3_client=s3_client,
            default_bucket=default_bucket
        )
        
        logger.info("S3 integration created successfully")
        return integration
        
    except S3Error as e:
        logger.error(f"Failed to create S3 integration: {e}", exc_info=True)
        raise