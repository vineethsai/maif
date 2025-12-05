"""
MAIF Transactions Module

Contains ACID transaction support:
- Transaction management
- Event sourcing
- Optimistic/pessimistic locking
"""

try:
    from .acid_transactions import (
        AcidMAIFEncoder,
        TransactionManager,
        Transaction,
        IsolationLevel,
    )
except ImportError:
    AcidMAIFEncoder = None
    TransactionManager = None
    Transaction = None
    IsolationLevel = None

try:
    from .acid_truly_optimized import TrulyOptimizedAcidMAIF
except ImportError:
    TrulyOptimizedAcidMAIF = None

try:
    from .event_sourcing import EventStore, Event, EventSourcedMAIF
except ImportError:
    EventStore = None
    Event = None
    EventSourcedMAIF = None

__all__ = [
    # ACID transactions
    "AcidMAIFEncoder",
    "TransactionManager",
    "Transaction",
    "IsolationLevel",
    "TrulyOptimizedAcidMAIF",
    # Event sourcing
    "EventStore",
    "Event",
    "EventSourcedMAIF",
]

