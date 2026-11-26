"""
Repository Interfaces for Conversation Management

These abstract base classes define the contract for data access,
enabling different storage backends (SQLite, MongoDB, Redis, etc.)
to be used interchangeably.

Benefits:
- Loose coupling between business logic and storage
- Easy to test with mock implementations
- Swap storage backends without changing business code
- Per-tenant storage configuration possible
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, AsyncIterator

from app.domain.models import (
    Message, Session, Conversation, ConversationEvent,
    MessageRole, SessionStatus, Visibility,
    SessionId, ClientId, TenantId
)


class ConversationRepository(ABC):
    """
    Repository for conversation aggregate operations.

    This is the primary repository for managing conversations,
    treating them as aggregate roots.
    """

    @abstractmethod
    async def get(self, session_id: SessionId) -> Optional[Conversation]:
        """
        Retrieve a conversation by session ID.

        Args:
            session_id: The unique session identifier

        Returns:
            The conversation if found, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        """
        Save a conversation (create or update).

        Args:
            conversation: The conversation to save
        """
        pass

    @abstractmethod
    async def delete(self, session_id: SessionId) -> bool:
        """
        Delete a conversation by session ID.

        Args:
            session_id: The session to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, session_id: SessionId) -> bool:
        """Check if a conversation exists."""
        pass

    @abstractmethod
    async def find_by_client(
        self,
        client_id: ClientId,
        limit: int = 100,
        offset: int = 0
    ) -> List[Conversation]:
        """
        Find all conversations for a specific client.

        Args:
            client_id: The client identifier
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of conversations for the client
        """
        pass

    @abstractmethod
    async def find_active_sessions(
        self,
        older_than: Optional[datetime] = None
    ) -> List[Session]:
        """
        Find all active sessions, optionally filtered by age.

        Args:
            older_than: If provided, only return sessions idle longer than this

        Returns:
            List of active sessions
        """
        pass

    @abstractmethod
    async def archive_expired_sessions(self) -> int:
        """
        Archive all expired sessions.

        Returns:
            Number of sessions archived
        """
        pass


class MessageRepository(ABC):
    """
    Repository for individual message operations.

    Used when you need fine-grained control over messages
    without loading the entire conversation.
    """

    @abstractmethod
    async def add(self, message: Message) -> Message:
        """
        Add a new message.

        Args:
            message: The message to add

        Returns:
            The saved message (with any generated fields populated)
        """
        pass

    @abstractmethod
    async def get_by_session(
        self,
        session_id: SessionId,
        limit: int = 50,
        offset: int = 0,
        order_desc: bool = True
    ) -> List[Message]:
        """
        Get messages for a session.

        Args:
            session_id: The session to get messages for
            limit: Maximum number of messages
            offset: Number of messages to skip
            order_desc: If True, newest first

        Returns:
            List of messages
        """
        pass

    @abstractmethod
    async def get_recent(
        self,
        session_id: SessionId,
        limit: int = 10
    ) -> List[Message]:
        """
        Get the most recent messages for a session.

        Args:
            session_id: The session ID
            limit: Number of recent messages to return

        Returns:
            List of recent messages, oldest first
        """
        pass

    @abstractmethod
    async def count_by_session(self, session_id: SessionId) -> int:
        """Count messages in a session."""
        pass

    @abstractmethod
    async def delete_by_session(self, session_id: SessionId) -> int:
        """
        Delete all messages for a session.

        Returns:
            Number of messages deleted
        """
        pass

    @abstractmethod
    async def delete_old_messages(
        self,
        session_id: SessionId,
        keep_recent: int = 50
    ) -> int:
        """
        Delete old messages, keeping only the most recent.

        Args:
            session_id: The session ID
            keep_recent: Number of recent messages to keep

        Returns:
            Number of messages deleted
        """
        pass


class SessionRepository(ABC):
    """
    Repository for session lifecycle management.

    Handles session creation, validation, expiration, and cleanup.
    """

    @abstractmethod
    async def create(
        self,
        client_id: Optional[ClientId] = None,
        tenant_id: Optional[TenantId] = None,
        visibility: Visibility = Visibility.PUBLIC,
        ttl: timedelta = timedelta(hours=24),
        metadata: Dict[str, Any] = None,
        session_id: Optional[str] = None
    ) -> Session:
        """
        Create a new session.

        Args:
            client_id: Optional client identifier for multi-tenancy
            tenant_id: Optional tenant identifier
            visibility: Public or private visibility
            ttl: Time-to-live for the session
            metadata: Optional session metadata

        Returns:
            The created session
        """
        pass

    @abstractmethod
    async def get(self, session_id: SessionId) -> Optional[Session]:
        """Get a session by ID."""
        pass

    @abstractmethod
    async def update(self, session: Session) -> None:
        """Update a session."""
        pass

    @abstractmethod
    async def touch(self, session_id: SessionId) -> Optional[Session]:
        """
        Update session's last activity timestamp.

        Returns:
            The updated session, or None if not found
        """
        pass

    @abstractmethod
    async def expire(self, session_id: SessionId) -> bool:
        """
        Expire a session immediately.

        Returns:
            True if expired, False if not found
        """
        pass

    @abstractmethod
    async def validate(self, session_id: SessionId) -> bool:
        """
        Validate that a session exists and is active.

        Returns:
            True if valid and active, False otherwise
        """
        pass

    @abstractmethod
    async def find_expired(self, batch_size: int = 100) -> List[Session]:
        """
        Find expired sessions for cleanup.

        Args:
            batch_size: Maximum number to return

        Returns:
            List of expired sessions
        """
        pass

    @abstractmethod
    async def cleanup_expired(self, older_than: timedelta = timedelta(days=7)) -> int:
        """
        Cleanup expired sessions older than the specified duration.

        Args:
            older_than: Delete sessions expired longer than this

        Returns:
            Number of sessions cleaned up
        """
        pass

    @abstractmethod
    async def get_by_client(
        self,
        client_id: ClientId,
        include_expired: bool = False
    ) -> List[Session]:
        """Get all sessions for a client."""
        pass


class EventRepository(ABC):
    """
    Repository for conversation events (event sourcing).

    Used for audit trails, replaying state, and temporal queries.
    """

    @abstractmethod
    async def append(self, event: ConversationEvent) -> None:
        """
        Append an event to the event stream.

        Args:
            event: The event to append
        """
        pass

    @abstractmethod
    async def get_events(
        self,
        session_id: SessionId,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[ConversationEvent]:
        """
        Get events for a session within a version range.

        Args:
            session_id: The session ID
            from_version: Starting version (inclusive)
            to_version: Ending version (inclusive), None for all

        Returns:
            List of events in order
        """
        pass

    @abstractmethod
    async def get_events_by_type(
        self,
        session_id: SessionId,
        event_type: str
    ) -> List[ConversationEvent]:
        """Get events of a specific type."""
        pass

    @abstractmethod
    async def get_latest_version(self, session_id: SessionId) -> int:
        """Get the latest event version for a session."""
        pass

    @abstractmethod
    async def stream_events(
        self,
        session_id: SessionId,
        from_version: int = 0
    ) -> AsyncIterator[ConversationEvent]:
        """
        Stream events for real-time processing.

        Yields:
            Events as they become available
        """
        pass


class CacheRepository(ABC):
    """
    Repository for caching conversation data.

    Provides multi-level caching support for performance optimization.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a cached value."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        pass

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            Number of keys deleted
        """
        pass

    @abstractmethod
    async def get_session_cache(self, session_id: SessionId) -> Optional[Dict[str, Any]]:
        """Get cached data for a session."""
        pass

    @abstractmethod
    async def set_session_cache(
        self,
        session_id: SessionId,
        data: Dict[str, Any],
        ttl: int = 3600
    ) -> None:
        """Cache data for a session."""
        pass

    @abstractmethod
    async def invalidate_session(self, session_id: SessionId) -> None:
        """Invalidate all cached data for a session."""
        pass


# ============================================================================
# Unit of Work Pattern
# ============================================================================

class UnitOfWork(ABC):
    """
    Unit of Work pattern for managing transactions across repositories.

    Ensures atomic operations when working with multiple repositories.
    """

    conversations: ConversationRepository
    messages: MessageRepository
    sessions: SessionRepository
    events: EventRepository
    cache: CacheRepository

    @abstractmethod
    async def __aenter__(self) -> "UnitOfWork":
        """Enter the unit of work context."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the unit of work context, committing or rolling back."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes."""
        pass


# ============================================================================
# Factory for Repository Creation
# ============================================================================

class RepositoryFactory(ABC):
    """
    Factory for creating repository instances.

    Allows different storage backends to be configured at runtime.
    """

    @abstractmethod
    def create_conversation_repository(self) -> ConversationRepository:
        """Create a conversation repository."""
        pass

    @abstractmethod
    def create_message_repository(self) -> MessageRepository:
        """Create a message repository."""
        pass

    @abstractmethod
    def create_session_repository(self) -> SessionRepository:
        """Create a session repository."""
        pass

    @abstractmethod
    def create_event_repository(self) -> EventRepository:
        """Create an event repository."""
        pass

    @abstractmethod
    def create_cache_repository(self) -> CacheRepository:
        """Create a cache repository."""
        pass

    @abstractmethod
    def create_unit_of_work(self) -> UnitOfWork:
        """Create a unit of work."""
        pass
