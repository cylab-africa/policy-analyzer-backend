"""
Domain Models for Conversation Management

These models represent the core business entities in the system,
independent of any specific storage or transport mechanism.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import uuid4


class MessageRole(str, Enum):
    """Role of a message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(str, Enum):
    """Lifecycle status of a conversation session."""
    ACTIVE = "active"
    IDLE = "idle"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class Visibility(str, Enum):
    """Data visibility/access level."""
    PUBLIC = "public"
    PRIVATE = "private"


@dataclass
class Message:
    """
    Represents a single message in a conversation.

    Immutable after creation - modifications create new instances.
    """
    content: str
    role: MessageRole
    session_id: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **kwargs) -> "Message":
        """Create a new message with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return Message(
            id=self.id,
            content=self.content,
            role=self.role,
            session_id=self.session_id,
            timestamp=self.timestamp,
            metadata=new_metadata
        )


@dataclass
class Session:
    """
    Represents a conversation session with lifecycle management.

    Sessions have explicit creation, expiration, and archival states.
    """
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    status: SessionStatus = SessionStatus.ACTIVE

    # Multi-tenant isolation
    client_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Session context
    visibility: Visibility = Visibility.PUBLIC
    current_tab: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Configuration overrides (per-client customization)
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() > self.expires_at or self.status == SessionStatus.EXPIRED

    def is_active(self) -> bool:
        """Check if the session is still active."""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()

    def renew(self, ttl: timedelta) -> "Session":
        """Return an active copy of this session that expires ttl from now."""
        now = datetime.utcnow()
        return replace(self, status=SessionStatus.ACTIVE, last_activity=now, expires_at=now + ttl)

    def touch(self, extend_by: timedelta = timedelta(hours=1)) -> "Session":
        """
        Update last activity and optionally extend expiration.
        Returns a new Session instance (immutability).
        """
        now = datetime.utcnow()
        new_expires = max(self.expires_at, now + extend_by)

        return Session(
            session_id=self.session_id,
            created_at=self.created_at,
            last_activity=now,
            expires_at=new_expires,
            status=SessionStatus.ACTIVE,
            client_id=self.client_id,
            tenant_id=self.tenant_id,
            visibility=self.visibility,
            current_tab=self.current_tab,
            metadata=self.metadata,
            config_overrides=self.config_overrides
        )

    def archive(self) -> "Session":
        """Mark session as archived."""
        return Session(
            session_id=self.session_id,
            created_at=self.created_at,
            last_activity=self.last_activity,
            expires_at=self.expires_at,
            status=SessionStatus.ARCHIVED,
            client_id=self.client_id,
            tenant_id=self.tenant_id,
            visibility=self.visibility,
            current_tab=self.current_tab,
            metadata=self.metadata,
            config_overrides=self.config_overrides
        )

    def expire(self) -> "Session":
        """Mark session as expired."""
        return Session(
            session_id=self.session_id,
            created_at=self.created_at,
            last_activity=self.last_activity,
            expires_at=datetime.utcnow(),
            status=SessionStatus.EXPIRED,
            client_id=self.client_id,
            tenant_id=self.tenant_id,
            visibility=self.visibility,
            current_tab=self.current_tab,
            metadata=self.metadata,
            config_overrides=self.config_overrides
        )


@dataclass
class Conversation:
    """
    Aggregate root for a complete conversation.

    Contains the session and all associated messages.
    This is the primary entity for conversation management.
    """
    session: Session
    messages: List[Message] = field(default_factory=list)

    # Computed conversation state
    summary: Optional[str] = None
    detected_entities: Dict[str, Any] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def add_message(self, content: str, role: MessageRole, metadata: Dict[str, Any] = None) -> "Conversation":
        """
        Add a message to the conversation.
        Returns a new Conversation instance (immutability).
        """
        new_message = Message(
            content=content,
            role=role,
            session_id=self.session.session_id,
            metadata=metadata or {}
        )

        new_messages = self.messages + [new_message]
        new_session = self.session.touch()

        return Conversation(
            session=new_session,
            messages=new_messages,
            summary=self.summary,
            detected_entities=self.detected_entities
        )

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-limit:] if self.messages else []

    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Get all messages with a specific role."""
        return [m for m in self.messages if m.role == role]

    def to_context_string(self, max_messages: int = 6) -> str:
        """
        Format recent messages as context string for LLM.
        """
        recent = self.get_recent_messages(max_messages)
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == MessageRole.USER else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        return "\n\n".join(lines)


@dataclass
class ConversationEvent:
    """
    Event representing a change in conversation state.
    Used for event sourcing and audit trails.
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For ordering and deduplication
    version: int = 0


@dataclass
class QuerySubmittedEvent(ConversationEvent):
    """Event emitted when a user submits a query."""
    event_type: str = "query_submitted"


@dataclass
class ResponseGeneratedEvent(ConversationEvent):
    """Event emitted when a response is generated."""
    event_type: str = "response_generated"


@dataclass
class SessionCreatedEvent(ConversationEvent):
    """Event emitted when a session is created."""
    event_type: str = "session_created"


@dataclass
class SessionExpiredEvent(ConversationEvent):
    """Event emitted when a session expires."""
    event_type: str = "session_expired"


# Type aliases for clarity
SessionId = str
ClientId = str
TenantId = str
MessageId = str
