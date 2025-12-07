"""
Shared utilities for MAIF framework integrations.

This module provides common utility functions used across all framework
integrations, including:

- Safe JSON serialization with type handling
- Timestamp formatting
- Data truncation for large payloads
- Error handling utilities
"""

import json
import time
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID


# Maximum size for serialized data (to prevent huge artifacts)
MAX_DATA_SIZE = 100_000  # 100KB
MAX_STRING_LENGTH = 10_000  # 10K characters for individual strings


class SerializationError(Exception):
    """Raised when data cannot be serialized safely."""
    pass


def safe_serialize(data: Any, max_size: int = MAX_DATA_SIZE) -> str:
    """Safely serialize data to JSON string.
    
    Handles common non-JSON-serializable types:
    - UUID -> str
    - datetime -> ISO format string
    - bytes -> base64 or truncated hex
    - sets -> lists
    - custom objects -> str representation
    
    Args:
        data: The data to serialize
        max_size: Maximum size of the output string
        
    Returns:
        JSON string representation of the data
        
    Raises:
        SerializationError: If data cannot be serialized
    """
    try:
        result = json.dumps(data, cls=MAIFJSONEncoder, ensure_ascii=False)
        
        if len(result) > max_size:
            # Truncate with indicator
            truncated_data = _truncate_data(data, max_size)
            result = json.dumps(truncated_data, cls=MAIFJSONEncoder, ensure_ascii=False)
            
        return result
    except Exception as e:
        raise SerializationError(f"Failed to serialize data: {e}") from e


class MAIFJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles common non-serializable types."""
    
    def default(self, obj: Any) -> Any:
        # Handle UUID
        if isinstance(obj, UUID):
            return str(obj)
        
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle bytes
        if isinstance(obj, bytes):
            if len(obj) > 1000:
                # Large binary - just store hash and size
                return {
                    "_type": "bytes",
                    "size": len(obj),
                    "hash": hashlib.sha256(obj).hexdigest()[:16],
                }
            # Small binary - store as hex
            return {
                "_type": "bytes",
                "hex": obj.hex(),
                "size": len(obj),
            }
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Handle objects with __dict__
        if hasattr(obj, "__dict__"):
            return {
                "_type": type(obj).__name__,
                "_data": str(obj)[:MAX_STRING_LENGTH],
            }
        
        # Fallback to string representation
        try:
            return str(obj)[:MAX_STRING_LENGTH]
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


def _truncate_data(data: Any, max_size: int) -> Any:
    """Recursively truncate data to fit within size limit.
    
    Args:
        data: The data to truncate
        max_size: Target maximum size
        
    Returns:
        Truncated version of the data
    """
    if isinstance(data, str):
        if len(data) > MAX_STRING_LENGTH:
            return data[:MAX_STRING_LENGTH] + f"... [truncated, total {len(data)} chars]"
        return data
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            result[key] = _truncate_data(value, max_size // len(data) if data else max_size)
        return result
    
    if isinstance(data, list):
        if len(data) > 100:
            # Keep first and last items
            return (
                [_truncate_data(item, max_size // 10) for item in data[:50]]
                + [f"... {len(data) - 100} items omitted ..."]
                + [_truncate_data(item, max_size // 10) for item in data[-50:]]
            )
        return [_truncate_data(item, max_size // len(data) if data else max_size) for item in data]
    
    return data


def format_timestamp(ts: Optional[float] = None) -> str:
    """Format a Unix timestamp as ISO 8601 string.
    
    Args:
        ts: Unix timestamp (defaults to current time)
        
    Returns:
        ISO 8601 formatted datetime string
    """
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts).isoformat()


def generate_run_id() -> str:
    """Generate a unique run ID.
    
    Returns:
        A unique string identifier
    """
    import uuid
    return str(uuid.uuid4())


def truncate_string(s: str, max_length: int = MAX_STRING_LENGTH) -> str:
    """Truncate a string to a maximum length.
    
    Args:
        s: The string to truncate
        max_length: Maximum length
        
    Returns:
        Truncated string with indicator if truncated
    """
    if len(s) <= max_length:
        return s
    return s[:max_length] + f"... [truncated, {len(s)} total]"


def extract_error_info(error: Exception) -> Dict[str, Any]:
    """Extract structured information from an exception.
    
    Args:
        error: The exception to extract info from
        
    Returns:
        Dictionary with error type, message, and traceback info
    """
    import traceback
    
    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc()[:5000],  # Limit traceback size
    }


def safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from an object.
    
    Args:
        obj: The object to get the attribute from
        attr: The attribute name
        default: Default value if attribute doesn't exist
        
    Returns:
        The attribute value or default
    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def calculate_token_estimate(text: str) -> int:
    """Estimate token count for a text string.
    
    This is a rough estimate based on the common rule that
    1 token is approximately 4 characters for English text.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Rough estimate: ~4 characters per token
    return len(text) // 4


def merge_metadata(*dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple metadata dictionaries.
    
    Later dictionaries override earlier ones for duplicate keys.
    None values are skipped.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result: Dict[str, Any] = {}
    for d in dicts:
        if d is not None:
            result.update(d)
    return result

