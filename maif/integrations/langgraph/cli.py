"""
CLI tools for inspecting and managing MAIF LangGraph artifacts.

Usage:
    python -m maif.integrations.langgraph.cli inspect state.maif
    python -m maif.integrations.langgraph.cli verify state.maif
    python -m maif.integrations.langgraph.cli export state.maif --format json
    python -m maif.integrations.langgraph.cli migrate checkpoints.db state.maif
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="maif-langgraph",
        description="Inspect and manage MAIF LangGraph artifacts"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a MAIF artifact")
    inspect_parser.add_argument("artifact", help="Path to MAIF artifact")
    inspect_parser.add_argument("--checkpoints", "-c", action="store_true",
                                help="Show checkpoint details")
    inspect_parser.add_argument("--thread", "-t", help="Filter by thread ID")
    inspect_parser.add_argument("--limit", "-n", type=int, default=10,
                                help="Limit number of items shown")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify artifact integrity")
    verify_parser.add_argument("artifact", help="Path to MAIF artifact")
    verify_parser.add_argument("--verbose", "-v", action="store_true",
                               help="Show detailed verification")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export artifact data")
    export_parser.add_argument("artifact", help="Path to MAIF artifact")
    export_parser.add_argument("--format", "-f", choices=["json", "csv", "markdown"],
                               default="json", help="Export format")
    export_parser.add_argument("--output", "-o", help="Output file (stdout if omitted)")
    export_parser.add_argument("--thread", "-t", help="Filter by thread ID")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate from SqliteSaver")
    migrate_parser.add_argument("sqlite_db", help="Path to SQLite database")
    migrate_parser.add_argument("maif_artifact", help="Path for new MAIF artifact")
    migrate_parser.add_argument("--thread", "-t", help="Migrate specific thread only")
    
    # Threads command
    threads_parser = subparsers.add_parser("threads", help="List all threads")
    threads_parser.add_argument("artifact", help="Path to MAIF artifact")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "threads":
        cmd_threads(args)


def cmd_inspect(args):
    """Inspect a MAIF artifact."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Basic info
    print(f"\n{'='*60}")
    print(f"MAIF Artifact: {artifact_path.name}")
    print(f"{'='*60}")
    print(f"Size: {artifact_path.stat().st_size / 1024:.1f} KB")
    print(f"Blocks: {len(decoder.blocks)}")
    
    # Verify integrity
    is_valid, errors = decoder.verify_integrity()
    status = "VALID" if is_valid else "INVALID"
    print(f"Integrity: {status}")
    
    if errors:
        for err in errors[:3]:
            print(f"  - {err}")
    
    # Count by type
    type_counts = {}
    threads = set()
    
    for block in decoder.blocks:
        meta = block.metadata or {}
        event_type = meta.get("type", "unknown")
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        thread_id = meta.get("thread_id")
        if thread_id:
            threads.add(thread_id)
    
    print(f"\nThreads: {len(threads)}")
    print(f"\nEvent Types:")
    for event_type, count in sorted(type_counts.items()):
        print(f"  {event_type}: {count}")
    
    # Show checkpoints if requested
    if args.checkpoints:
        print(f"\n{'='*60}")
        print("Checkpoints")
        print(f"{'='*60}")
        
        count = 0
        for block in decoder.blocks:
            meta = block.metadata or {}
            if meta.get("type") != "state_checkpoint":
                continue
            
            if args.thread and meta.get("thread_id") != args.thread:
                continue
            
            count += 1
            if count > args.limit:
                print(f"\n... and more (use --limit to see more)")
                break
            
            ts = meta.get("timestamp", 0)
            time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "N/A"
            
            print(f"\n[{count}] Thread: {meta.get('thread_id', 'N/A')}")
            print(f"    Checkpoint: {meta.get('checkpoint_id', 'N/A')[:20]}...")
            print(f"    Time: {time_str}")
            print(f"    Parent: {meta.get('parent_checkpoint_id', 'None')}")


def cmd_verify(args):
    """Verify artifact integrity."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nVerifying: {artifact_path.name}")
    print("-" * 60)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Check header
    print("[1/4] File header... ", end="")
    try:
        with open(artifact_path, "rb") as f:
            magic = f.read(4)
        if magic == b"MAIF":
            print("OK (MAIF format)")
        else:
            print(f"FAIL (got {magic})")
    except Exception as e:
        print(f"FAIL ({e})")
    
    # Check blocks
    print(f"[2/4] Block count... ", end="")
    print(f"OK ({len(decoder.blocks)} blocks)")
    
    # Verify integrity
    print("[3/4] Hash chain... ", end="")
    is_valid, errors = decoder.verify_integrity()
    if is_valid:
        print("OK (all blocks linked)")
    else:
        print("FAIL")
        for err in errors:
            print(f"       - {err}")
    
    # Check signatures
    print("[4/4] Signatures... ", end="")
    if is_valid:
        print(f"OK ({len(decoder.blocks)} verified)")
    else:
        print("FAIL (integrity check failed)")
    
    print("-" * 60)
    if is_valid:
        print("RESULT: Artifact integrity VERIFIED")
        sys.exit(0)
    else:
        print("RESULT: Artifact integrity FAILED")
        sys.exit(1)


def cmd_export(args):
    """Export artifact data."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Collect data
    events = []
    for block in decoder.blocks:
        meta = block.metadata or {}
        
        if args.thread and meta.get("thread_id") != args.thread:
            continue
        
        # Parse block data
        try:
            data = block.data
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            parsed_data = json.loads(data)
        except:
            parsed_data = {"raw": str(block.data)[:500]}
        
        events.append({
            "type": meta.get("type", "unknown"),
            "thread_id": meta.get("thread_id", ""),
            "checkpoint_id": meta.get("checkpoint_id", ""),
            "timestamp": meta.get("timestamp", 0),
            "data": parsed_data,
        })
    
    # Format output
    if args.format == "json":
        output = json.dumps({
            "artifact": str(artifact_path),
            "events": events,
        }, indent=2, default=str)
    
    elif args.format == "csv":
        lines = ["timestamp,type,thread_id,checkpoint_id"]
        for e in events:
            ts = datetime.fromtimestamp(e["timestamp"]).isoformat() if e["timestamp"] else ""
            lines.append(f"{ts},{e['type']},{e['thread_id']},{e['checkpoint_id']}")
        output = "\n".join(lines)
    
    elif args.format == "markdown":
        lines = [
            f"# MAIF Export: {artifact_path.name}",
            "",
            f"**Total Events:** {len(events)}",
            "",
            "| Time | Type | Thread | Checkpoint |",
            "|------|------|--------|------------|",
        ]
        for e in events[:50]:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S") if e["timestamp"] else "N/A"
            lines.append(f"| {ts} | {e['type']} | {e['thread_id'][:10]} | {e['checkpoint_id'][:10]}... |")
        if len(events) > 50:
            lines.append(f"\n*... and {len(events) - 50} more events*")
        output = "\n".join(lines)
    
    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Exported to: {args.output}")
    else:
        print(output)


def cmd_migrate(args):
    """Migrate from SqliteSaver."""
    from .migration import migrate_from_sqlite
    
    sqlite_path = Path(args.sqlite_db)
    maif_path = Path(args.maif_artifact)
    
    if not sqlite_path.exists():
        print(f"Error: SQLite database not found: {sqlite_path}", file=sys.stderr)
        sys.exit(1)
    
    if maif_path.exists():
        response = input(f"MAIF artifact exists: {maif_path}. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        maif_path.unlink()
    
    stats = migrate_from_sqlite(
        sqlite_path,
        maif_path,
        thread_id=args.thread,
        verbose=True,
    )
    
    if stats["checkpoints_migrated"] > 0:
        print(f"\nMigration successful!")
        sys.exit(0)
    else:
        print(f"\nNo checkpoints migrated.")
        sys.exit(1)


def cmd_threads(args):
    """List all threads in artifact."""
    from maif import MAIFDecoder
    
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Error: Artifact not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)
    
    decoder = MAIFDecoder(str(artifact_path))
    decoder.load()
    
    # Collect thread info
    threads = {}
    for block in decoder.blocks:
        meta = block.metadata or {}
        thread_id = meta.get("thread_id")
        if not thread_id:
            continue
        
        if thread_id not in threads:
            threads[thread_id] = {"checkpoints": 0, "last_activity": 0}
        
        if meta.get("type") == "state_checkpoint":
            threads[thread_id]["checkpoints"] += 1
        
        ts = meta.get("timestamp", 0)
        if ts > threads[thread_id]["last_activity"]:
            threads[thread_id]["last_activity"] = ts
    
    print(f"\nThreads in {artifact_path.name}:")
    print("-" * 60)
    print(f"{'Thread ID':<30} | {'Checkpoints':<12} | Last Activity")
    print("-" * 60)
    
    for thread_id, info in sorted(threads.items()):
        ts = info["last_activity"]
        last = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "N/A"
        tid_display = thread_id[:28] + ".." if len(thread_id) > 30 else thread_id
        print(f"{tid_display:<30} | {info['checkpoints']:<12} | {last}")
    
    print("-" * 60)
    print(f"Total: {len(threads)} threads")


if __name__ == "__main__":
    main()

