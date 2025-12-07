"""
Migration utilities for LangGraph checkpointers.

Provides tools to migrate from other checkpointers (SqliteSaver, etc.)
to MAIF-backed checkpointing with full provenance.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, Union, Iterator, Dict, Any, Tuple

from .checkpointer import MAIFCheckpointer


def migrate_from_sqlite(
    sqlite_path: Union[str, Path],
    maif_path: Union[str, Path],
    *,
    thread_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Migrate checkpoints from SqliteSaver to MAIFCheckpointer.
    
    This function reads all checkpoints from a SqliteSaver database and
    stores them in a new MAIF artifact with full cryptographic provenance.
    
    Args:
        sqlite_path: Path to the SQLite database file
        maif_path: Path for the new MAIF artifact
        thread_id: Optional thread_id to filter (migrates all if None)
        verbose: Print progress messages
        
    Returns:
        Dictionary with migration statistics:
        - checkpoints_migrated: Number of checkpoints migrated
        - threads_migrated: List of thread IDs migrated
        - artifact_path: Path to the created MAIF artifact
        
    Example:
        >>> from maif.integrations.langgraph import migrate_from_sqlite
        >>> stats = migrate_from_sqlite("checkpoints.db", "checkpoints.maif")
        >>> print(f"Migrated {stats['checkpoints_migrated']} checkpoints")
        
    Note:
        The original SQLite database is not modified.
    """
    sqlite_path = Path(sqlite_path)
    maif_path = Path(maif_path)
    
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    
    if verbose:
        print(f"Migrating from {sqlite_path} to {maif_path}...")
    
    # Read checkpoints from SQLite
    checkpoints = list(_read_sqlite_checkpoints(sqlite_path, thread_id))
    
    if not checkpoints:
        if verbose:
            print("No checkpoints found to migrate.")
        return {
            "checkpoints_migrated": 0,
            "threads_migrated": [],
            "artifact_path": str(maif_path),
        }
    
    # Create MAIF checkpointer and migrate
    maif_checkpointer = MAIFCheckpointer(maif_path, agent_id="migration")
    
    threads_seen = set()
    for i, (config, checkpoint, metadata) in enumerate(checkpoints):
        thread = config.get("configurable", {}).get("thread_id", "unknown")
        threads_seen.add(thread)
        
        # Store in MAIF
        maif_checkpointer.put(config, checkpoint, metadata)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Migrated {i + 1} checkpoints...")
    
    # Finalize
    maif_checkpointer.finalize()
    
    if verbose:
        print(f"Migration complete: {len(checkpoints)} checkpoints from {len(threads_seen)} threads")
        print(f"MAIF artifact: {maif_path}")
    
    return {
        "checkpoints_migrated": len(checkpoints),
        "threads_migrated": list(threads_seen),
        "artifact_path": str(maif_path),
    }


def _read_sqlite_checkpoints(
    sqlite_path: Path,
    thread_id: Optional[str] = None,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """Read checkpoints from a SqliteSaver database.
    
    Yields:
        Tuples of (config, checkpoint, metadata)
    """
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    
    try:
        # Check table structure - SqliteSaver uses 'checkpoints' table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        )
        if not cursor.fetchone():
            # Try alternate table name
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            raise ValueError(
                f"No 'checkpoints' table found. Available tables: {tables}"
            )
        
        # Query checkpoints
        query = "SELECT * FROM checkpoints"
        params = []
        
        if thread_id:
            query += " WHERE thread_id = ?"
            params.append(thread_id)
        
        query += " ORDER BY thread_id, checkpoint_id"
        
        cursor = conn.execute(query, params)
        
        for row in cursor:
            row_dict = dict(row)
            
            # Extract fields
            cp_thread_id = row_dict.get("thread_id", "")
            cp_checkpoint_ns = row_dict.get("checkpoint_ns", "")
            cp_checkpoint_id = row_dict.get("checkpoint_id", "")
            
            # Parse checkpoint data
            checkpoint_blob = row_dict.get("checkpoint")
            if checkpoint_blob:
                if isinstance(checkpoint_blob, bytes):
                    checkpoint = json.loads(checkpoint_blob.decode("utf-8"))
                else:
                    checkpoint = json.loads(checkpoint_blob)
            else:
                checkpoint = {}
            
            # Parse metadata
            metadata_blob = row_dict.get("metadata")
            if metadata_blob:
                if isinstance(metadata_blob, bytes):
                    metadata = json.loads(metadata_blob.decode("utf-8"))
                else:
                    metadata = json.loads(metadata_blob)
            else:
                metadata = {}
            
            # Build config
            config = {
                "configurable": {
                    "thread_id": cp_thread_id,
                    "checkpoint_ns": cp_checkpoint_ns,
                    "checkpoint_id": cp_checkpoint_id,
                }
            }
            
            # Handle parent reference
            parent_id = row_dict.get("parent_checkpoint_id")
            if parent_id:
                config["configurable"]["parent_checkpoint_id"] = parent_id
            
            yield config, checkpoint, metadata
            
    finally:
        conn.close()


def compare_checkpointers(
    sqlite_path: Union[str, Path],
    maif_path: Union[str, Path],
    thread_id: str,
) -> Dict[str, Any]:
    """Compare checkpoints between SqliteSaver and MAIFCheckpointer.
    
    Useful for verifying migration was successful.
    
    Args:
        sqlite_path: Path to SQLite database
        maif_path: Path to MAIF artifact
        thread_id: Thread ID to compare
        
    Returns:
        Comparison results with match status
    """
    sqlite_path = Path(sqlite_path)
    maif_path = Path(maif_path)
    
    # Read from SQLite
    sqlite_checkpoints = list(_read_sqlite_checkpoints(sqlite_path, thread_id))
    
    # Read from MAIF
    maif_checkpointer = MAIFCheckpointer(maif_path)
    maif_checkpoints = list(maif_checkpointer.list(
        {"configurable": {"thread_id": thread_id}}
    ))
    
    return {
        "sqlite_count": len(sqlite_checkpoints),
        "maif_count": len(maif_checkpoints),
        "match": len(sqlite_checkpoints) == len(maif_checkpoints),
        "thread_id": thread_id,
    }

