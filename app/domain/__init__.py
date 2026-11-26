"""
Domain Layer - Core Business Logic and Abstractions

This package contains:
- Domain models (entities, value objects)
- Repository interfaces (abstract base classes)
- Domain events for event sourcing

The domain layer is independent of infrastructure concerns
like databases, web frameworks, or external services.
"""

from app.domain.models import (
    Message,
    MessageRole,
    Session,
    SessionStatus,
    Visibility,
    Conversation,
    ConversationEvent,
    QuerySubmittedEvent,
    ResponseGeneratedEvent,
    SessionCreatedEvent,
    SessionExpiredEvent,
)

from app.domain.repositories import (
    ConversationRepository,
    MessageRepository,
    SessionRepository,
    EventRepository,
    CacheRepository,
    UnitOfWork,
    RepositoryFactory,
)

__all__ = [
    # Models
    "Message",
    "MessageRole",
    "Session",
    "SessionStatus",
    "Visibility",
    "Conversation",
    "ConversationEvent",
    "QuerySubmittedEvent",
    "ResponseGeneratedEvent",
    "SessionCreatedEvent",
    "SessionExpiredEvent",
    # Repositories
    "ConversationRepository",
    "MessageRepository",
    "SessionRepository",
    "EventRepository",
    "CacheRepository",
    "UnitOfWork",
    "RepositoryFactory",
]
