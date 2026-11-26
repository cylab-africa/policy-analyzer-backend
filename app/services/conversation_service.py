"""
Conversation Service - Business Logic Layer

This service orchestrates conversation management, providing a clean API
for the rest of the application while encapsulating business rules.

Key responsibilities:
- Session lifecycle management
- Message handling with validation
- Multi-tenant isolation enforcement
- Context building for LLM prompts
- Conversation state management
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple

from app.core.config import settings
from app.domain.models import (
    Message, MessageRole, Session, SessionStatus,
    Visibility, Conversation, SessionId, ClientId
)
from app.domain.repositories import (
    ConversationRepository, MessageRepository, SessionRepository
)

logger = logging.getLogger(__name__)


class ConversationServiceError(Exception):
    """Base exception for conversation service errors."""
    pass


class SessionNotFoundError(ConversationServiceError):
    """Raised when a session is not found."""
    pass


class SessionExpiredError(ConversationServiceError):
    """Raised when attempting to use an expired session."""
    pass


class UnauthorizedAccessError(ConversationServiceError):
    """Raised when attempting unauthorized access to a conversation."""
    pass


class ConversationService:
    """
    Service layer for conversation management.

    This class provides the main API for managing conversations,
    enforcing business rules and coordinating repository operations.
    """

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        session_repo: SessionRepository,
        default_session_ttl: timedelta = timedelta(hours=24),
        max_messages_in_context: int = 10,
        max_context_chars: int = 8000
    ):
        self.conversations = conversation_repo
        self.messages = message_repo
        self.sessions = session_repo
        self.default_session_ttl = default_session_ttl
        self.max_messages_in_context = max_messages_in_context
        self.max_context_chars = max_context_chars

    # =========================================================================
    # Session Management
    # =========================================================================

    async def create_session(
        self,
        client_id: Optional[ClientId] = None,
        visibility: Visibility = Visibility.PUBLIC,
        metadata: Dict[str, Any] = None,
        config_overrides: Dict[str, Any] = None,
        session_id: Optional[SessionId] = None
    ) -> Session:
        """
        Create a new conversation session.

        Args:
            client_id: Optional client identifier for multi-tenancy
            visibility: Public or private visibility
            metadata: Optional session metadata
            config_overrides: Per-session configuration overrides
            session_id: ID to use; a new one is made when omitted

        Returns:
            The created session
        """
        session = await self.sessions.create(
            client_id=client_id,
            visibility=visibility,
            ttl=self.default_session_ttl,
            metadata=metadata or {},
            session_id=session_id
        )

        if config_overrides:
            session = Session(
                session_id=session.session_id,
                created_at=session.created_at,
                last_activity=session.last_activity,
                expires_at=session.expires_at,
                status=session.status,
                client_id=session.client_id,
                tenant_id=session.tenant_id,
                visibility=session.visibility,
                current_tab=session.current_tab,
                metadata=session.metadata,
                config_overrides=config_overrides
            )
            await self.sessions.update(session)

        logger.info(
            f"Created session {session.session_id} | "
            f"client={client_id} | visibility={visibility.value}"
        )
        return session

    async def get_or_create_session(
        self,
        session_id: Optional[SessionId] = None,
        client_id: Optional[ClientId] = None,
        visibility: Visibility = Visibility.PUBLIC
    ) -> Tuple[Session, bool]:
        """
        Get an existing session, or create one under the given ID. An expired session is
        renewed, so that it keeps its ID and messages. Only its owner can use or renew it.

        Args:
            session_id: Optional existing session ID
            client_id: Client ID for new session
            visibility: Visibility for new session

        Returns:
            Tuple of (session, is_new)
        """
        if session_id:
            session = await self.sessions.get(session_id)
            if session:
                # Checked first, so that an expired session cannot be taken over
                if client_id and session.client_id != client_id:
                    raise UnauthorizedAccessError(
                        f"Client {client_id} cannot access session {session_id}"
                    )
                if session.is_expired():
                    await self.sessions.update(session.renew(self.default_session_ttl))
                # Touch the session to update activity
                session = await self.sessions.touch(session_id)
                return session, False

        # Create new session
        new_session = await self.create_session(
            client_id=client_id,
            visibility=visibility,
            session_id=session_id
        )
        return new_session, True

    async def validate_session(
        self,
        session_id: SessionId,
        client_id: Optional[ClientId] = None
    ) -> Session:
        """
        Validate and return a session.

        Args:
            session_id: The session to validate
            client_id: Optional client ID for access control

        Returns:
            The validated session

        Raises:
            SessionNotFoundError: If session doesn't exist
            SessionExpiredError: If session is expired
            UnauthorizedAccessError: If client doesn't have access
        """
        session = await self.sessions.get(session_id)

        if not session:
            raise SessionNotFoundError(f"Session {session_id} not found")

        if session.is_expired():
            raise SessionExpiredError(f"Session {session_id} has expired")

        if client_id and session.client_id and session.client_id != client_id:
            raise UnauthorizedAccessError(
                f"Client {client_id} cannot access session {session_id}"
            )

        return session

    async def end_session(self, session_id: SessionId) -> bool:
        """
        End a session, marking it as archived.

        Args:
            session_id: The session to end

        Returns:
            True if session was ended, False if not found
        """
        session = await self.sessions.get(session_id)
        if not session:
            return False

        archived = session.archive()
        await self.sessions.update(archived)

        logger.info(f"Ended session {session_id}")
        return True

    # =========================================================================
    # Message Management
    # =========================================================================

    async def add_user_message(
        self,
        session_id: SessionId,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """
        Add a user message to a conversation.

        Args:
            session_id: The session to add the message to
            content: The message content
            metadata: Optional message metadata

        Returns:
            The created message
        """
        # Validate session exists and is active
        await self.validate_session(session_id)

        message = Message(
            content=content,
            role=MessageRole.USER,
            session_id=session_id,
            metadata=metadata or {}
        )

        saved = await self.messages.add(message)

        # Touch session to update activity
        await self.sessions.touch(session_id)

        return saved

    async def add_assistant_message(
        self,
        session_id: SessionId,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """
        Add an assistant message to a conversation.

        Args:
            session_id: The session to add the message to
            content: The message content
            metadata: Optional message metadata (e.g., sources, timing)

        Returns:
            The created message
        """
        message = Message(
            content=content,
            role=MessageRole.ASSISTANT,
            session_id=session_id,
            metadata=metadata or {}
        )

        saved = await self.messages.add(message)

        # Touch session to update activity
        await self.sessions.touch(session_id)

        return saved

    async def add_message_pair(
        self,
        session_id: SessionId,
        user_content: str,
        assistant_content: str,
        user_metadata: Dict[str, Any] = None,
        assistant_metadata: Dict[str, Any] = None
    ) -> Tuple[Message, Message]:
        """
        Add a user-assistant message pair atomically.

        Args:
            session_id: The session ID
            user_content: User message content
            assistant_content: Assistant message content
            user_metadata: User message metadata
            assistant_metadata: Assistant message metadata

        Returns:
            Tuple of (user_message, assistant_message)
        """
        user_msg = await self.add_user_message(
            session_id, user_content, user_metadata
        )
        assistant_msg = await self.add_assistant_message(
            session_id, assistant_content, assistant_metadata
        )
        return user_msg, assistant_msg

    # =========================================================================
    # Conversation Retrieval
    # =========================================================================

    async def get_conversation(
        self,
        session_id: SessionId,
        client_id: Optional[ClientId] = None
    ) -> Optional[Conversation]:
        """
        Get a conversation by session ID.

        Args:
            session_id: The session ID
            client_id: Optional client ID for access control

        Returns:
            The conversation if found and accessible
        """
        conversation = await self.conversations.get(session_id)

        if not conversation:
            return None

        # Enforce client access
        if client_id and conversation.session.client_id:
            if conversation.session.client_id != client_id:
                raise UnauthorizedAccessError(
                    f"Client {client_id} cannot access conversation {session_id}"
                )

        return conversation

    async def get_conversation_history(
        self,
        session_id: SessionId,
        limit: int = 10,
        client_id: Optional[ClientId] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history in a format suitable for the existing API.

        This method provides backward compatibility with the existing
        chat history interface.

        Args:
            session_id: The session ID
            limit: Maximum number of messages
            client_id: Optional client ID for access control

        Returns:
            List of message dictionaries
        """
        # Validate access if client_id provided
        if client_id:
            session = await self.sessions.get(session_id)
            if session and session.client_id and session.client_id != client_id:
                raise UnauthorizedAccessError(
                    f"Client {client_id} cannot access session {session_id}"
                )

        messages = await self.messages.get_recent(session_id, limit=limit)

        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]

    async def get_context_for_llm(
        self,
        session_id: SessionId,
        max_messages: Optional[int] = None
    ) -> str:
        """
        Get conversation context formatted for LLM prompts.

        Args:
            session_id: The session ID
            max_messages: Maximum messages to include

        Returns:
            Formatted context string
        """
        limit = max_messages or self.max_messages_in_context
        messages = await self.messages.get_recent(session_id, limit=limit)

        if not messages:
            return ""

        lines = []
        total_chars = 0

        for msg in messages:
            role_label = "User" if msg.role == MessageRole.USER else "Assistant"
            line = f"{role_label}: {msg.content}"

            # Check character limit
            if total_chars + len(line) > self.max_context_chars:
                break

            lines.append(line)
            total_chars += len(line)

        return "\n\n".join(lines)

    # =========================================================================
    # Cleanup and Maintenance
    # =========================================================================

    async def cleanup_old_messages(
        self,
        session_id: SessionId,
        keep_recent: int = 50
    ) -> int:
        """
        Clean up old messages from a session.

        Args:
            session_id: The session to clean up
            keep_recent: Number of recent messages to keep

        Returns:
            Number of messages deleted
        """
        deleted = await self.messages.delete_old_messages(session_id, keep_recent)

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old messages from session {session_id}")

        return deleted

    async def cleanup_expired_sessions(
        self,
        older_than: timedelta = timedelta(days=7)
    ) -> int:
        """
        Clean up expired sessions and their messages.

        Args:
            older_than: Delete sessions expired longer than this

        Returns:
            Number of sessions cleaned up
        """
        return await self.sessions.cleanup_expired(older_than)

    async def archive_idle_sessions(
        self,
        idle_threshold: timedelta = timedelta(hours=2)
    ) -> int:
        """
        Archive sessions that have been idle beyond the threshold.

        Args:
            idle_threshold: Archive sessions idle longer than this

        Returns:
            Number of sessions archived
        """
        cutoff = datetime.now(timezone.utc) - idle_threshold
        idle_sessions = await self.conversations.find_active_sessions(older_than=cutoff)

        count = 0
        for session in idle_sessions:
            archived = session.archive()
            await self.sessions.update(archived)
            count += 1

        if count > 0:
            logger.info(f"Archived {count} idle sessions")

        return count

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    async def get_session_stats(self, session_id: SessionId) -> Dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: The session ID

        Returns:
            Dictionary with session statistics
        """
        session = await self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        message_count = await self.messages.count_by_session(session_id)

        return {
            "session_id": session_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "is_expired": session.is_expired(),
            "message_count": message_count,
            "visibility": session.visibility.value,
            "client_id": session.client_id
        }


# ============================================================================
# Factory function for creating the service
# ============================================================================

_conversation_service: Optional[ConversationService] = None


def get_conversation_service(
    db_path: Optional[str] = None,
    force_new: bool = False
) -> ConversationService:
    """
    Get or create the conversation service singleton.

    Args:
        db_path: Path to the SQLite database (default: conversations.db in DB_DATA_DIR)
        force_new: If True, create a new instance

    Returns:
        The conversation service instance
    """
    global _conversation_service

    if _conversation_service is None or force_new:
        from app.infrastructure.sqlite_repositories import (
            SQLiteConversationRepository,
            SQLiteMessageRepository,
            SQLiteSessionRepository
        )

        if db_path is None:
            os.makedirs(settings.db_data_dir, exist_ok=True)
            db_path = os.path.join(settings.db_data_dir, "conversations.db")

        message_repo = SQLiteMessageRepository(db_path)
        session_repo = SQLiteSessionRepository(db_path)
        conversation_repo = SQLiteConversationRepository(
            db_path, message_repo, session_repo
        )

        _conversation_service = ConversationService(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            session_repo=session_repo
        )

    return _conversation_service
