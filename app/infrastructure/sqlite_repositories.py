"""
SQLite Repository Implementations

Concrete implementations of repository interfaces using SQLite.
These can be swapped with MongoDB, PostgreSQL, or any other storage.
"""

import aiosqlite
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.domain.models import (
    Message, MessageRole, Session, SessionStatus,
    Visibility, Conversation
)
from app.domain.repositories import (
    MessageRepository, SessionRepository, ConversationRepository
)

logger = logging.getLogger(__name__)


class SQLiteMessageRepository(MessageRepository):
    """
    SQLite implementation of MessageRepository.

    Thread-safe async implementation using aiosqlite.
    """

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the database schema exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp DESC)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_role
                ON messages(session_id, role)
            """)
            await db.commit()

        self._initialized = True

    async def add(self, message: Message) -> Message:
        """Add a new message."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role.value,
                    message.content,
                    message.timestamp.isoformat(),
                    json.dumps(message.metadata) if message.metadata else None
                )
            )
            await db.commit()

        logger.debug(f"Added message {message.id} to session {message.session_id}")
        return message

    async def get_by_session(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
        order_desc: bool = True
    ) -> List[Message]:
        """Get messages for a session."""
        await self._ensure_initialized()

        order = "DESC" if order_desc else "ASC"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""
                SELECT id, session_id, role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp {order}
                LIMIT ? OFFSET ?
                """,
                (session_id, limit, offset)
            )
            rows = await cursor.fetchall()

        messages = []
        for row in rows:
            messages.append(Message(
                id=row["id"],
                session_id=row["session_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            ))

        # If descending order, reverse to return oldest first
        if order_desc:
            messages.reverse()

        return messages

    async def get_recent(self, session_id: str, limit: int = 10) -> List[Message]:
        """Get the most recent messages for a session."""
        return await self.get_by_session(session_id, limit=limit, order_desc=True)

    async def count_by_session(self, session_id: str) -> int:
        """Count messages in a session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def delete_by_session(self, session_id: str) -> int:
        """Delete all messages for a session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            return cursor.rowcount

    async def delete_old_messages(
        self,
        session_id: str,
        keep_recent: int = 50
    ) -> int:
        """Delete old messages, keeping only the most recent."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Get IDs of messages to keep
            cursor = await db.execute(
                """
                SELECT id FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, keep_recent)
            )
            keep_ids = [row[0] for row in await cursor.fetchall()]

            if not keep_ids:
                return 0

            # Delete messages not in keep list
            placeholders = ",".join("?" * len(keep_ids))
            cursor = await db.execute(
                f"""
                DELETE FROM messages
                WHERE session_id = ? AND id NOT IN ({placeholders})
                """,
                [session_id] + keep_ids
            )
            await db.commit()
            return cursor.rowcount


class SQLiteSessionRepository(SessionRepository):
    """
    SQLite implementation of SessionRepository.

    Manages session lifecycle with proper expiration handling.
    """

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the database schema exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    client_id TEXT,
                    tenant_id TEXT,
                    visibility TEXT NOT NULL DEFAULT 'public',
                    current_tab TEXT,
                    metadata TEXT,
                    config_overrides TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_client
                ON sessions(client_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status_expires
                ON sessions(status, expires_at)
            """)
            await db.commit()

        self._initialized = True

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
        """Convert a database row to a Session object."""
        return Session(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            status=SessionStatus(row["status"]),
            client_id=row["client_id"],
            tenant_id=row["tenant_id"],
            visibility=Visibility(row["visibility"]),
            current_tab=row["current_tab"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            config_overrides=json.loads(row["config_overrides"]) if row["config_overrides"] else {}
        )

    async def create(
        self,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        visibility: Visibility = Visibility.PUBLIC,
        ttl: timedelta = timedelta(hours=24),
        metadata: Dict[str, Any] = None,
        session_id: Optional[str] = None
    ) -> Session:
        """Create a new session, under session_id when one is given."""
        await self._ensure_initialized()

        now = datetime.utcnow()
        session = Session(
            **({"session_id": session_id} if session_id else {}),
            created_at=now,
            last_activity=now,
            expires_at=now + ttl,
            status=SessionStatus.ACTIVE,
            client_id=client_id,
            tenant_id=tenant_id,
            visibility=visibility,
            metadata=metadata or {}
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO sessions
                (session_id, created_at, last_activity, expires_at, status,
                 client_id, tenant_id, visibility, current_tab, metadata, config_overrides)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.expires_at.isoformat(),
                    session.status.value,
                    session.client_id,
                    session.tenant_id,
                    session.visibility.value,
                    session.current_tab,
                    json.dumps(session.metadata) if session.metadata else None,
                    json.dumps(session.config_overrides) if session.config_overrides else None
                )
            )
            await db.commit()

        logger.info(f"Created session {session.session_id} for client {client_id}")
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_session(row)

    async def update(self, session: Session) -> None:
        """Update a session."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE sessions SET
                    last_activity = ?,
                    expires_at = ?,
                    status = ?,
                    visibility = ?,
                    current_tab = ?,
                    metadata = ?,
                    config_overrides = ?
                WHERE session_id = ?
                """,
                (
                    session.last_activity.isoformat(),
                    session.expires_at.isoformat(),
                    session.status.value,
                    session.visibility.value,
                    session.current_tab,
                    json.dumps(session.metadata) if session.metadata else None,
                    json.dumps(session.config_overrides) if session.config_overrides else None,
                    session.session_id
                )
            )
            await db.commit()

    async def touch(self, session_id: str) -> Optional[Session]:
        """Update session's last activity timestamp."""
        session = await self.get(session_id)
        if not session:
            return None

        updated = session.touch()
        await self.update(updated)
        return updated

    async def expire(self, session_id: str) -> bool:
        """Expire a session immediately."""
        session = await self.get(session_id)
        if not session:
            return False

        expired = session.expire()
        await self.update(expired)
        logger.info(f"Expired session {session_id}")
        return True

    async def validate(self, session_id: str) -> bool:
        """Validate that a session exists and is active."""
        session = await self.get(session_id)
        if not session:
            return False
        return session.is_active()

    async def find_expired(self, batch_size: int = 100) -> List[Session]:
        """Find expired sessions for cleanup."""
        await self._ensure_initialized()

        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM sessions
                WHERE status = 'active' AND expires_at < ?
                LIMIT ?
                """,
                (now, batch_size)
            )
            rows = await cursor.fetchall()

        return [self._row_to_session(row) for row in rows]

    async def cleanup_expired(self, older_than: timedelta = timedelta(days=7)) -> int:
        """Cleanup expired sessions older than the specified duration."""
        await self._ensure_initialized()

        cutoff = (datetime.utcnow() - older_than).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM sessions
                WHERE status IN ('expired', 'archived')
                AND expires_at < ?
                """,
                (cutoff,)
            )
            await db.commit()
            deleted = cursor.rowcount

        logger.info(f"Cleaned up {deleted} old sessions")
        return deleted

    async def get_by_client(
        self,
        client_id: str,
        include_expired: bool = False
    ) -> List[Session]:
        """Get all sessions for a client."""
        await self._ensure_initialized()

        query = "SELECT * FROM sessions WHERE client_id = ?"
        params = [client_id]

        if not include_expired:
            query += " AND status = 'active'"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [self._row_to_session(row) for row in rows]


class SQLiteConversationRepository(ConversationRepository):
    """
    SQLite implementation of ConversationRepository.

    Manages conversations as aggregate roots, coordinating
    session and message repositories.
    """

    def __init__(
        self,
        db_path: str = "conversations.db",
        message_repo: Optional[SQLiteMessageRepository] = None,
        session_repo: Optional[SQLiteSessionRepository] = None
    ):
        self.db_path = db_path
        self.messages = message_repo or SQLiteMessageRepository(db_path)
        self.sessions = session_repo or SQLiteSessionRepository(db_path)

    async def get(self, session_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by session ID."""
        session = await self.sessions.get(session_id)
        if not session:
            return None

        messages = await self.messages.get_by_session(session_id, limit=100)

        return Conversation(
            session=session,
            messages=messages
        )

    async def save(self, conversation: Conversation) -> None:
        """Save a conversation (create or update)."""
        # Check if session exists
        existing = await self.sessions.get(conversation.session_id)

        if existing:
            await self.sessions.update(conversation.session)
        else:
            # Create new session via direct insert
            async with aiosqlite.connect(self.db_path) as db:
                session = conversation.session
                await db.execute(
                    """
                    INSERT OR REPLACE INTO sessions
                    (session_id, created_at, last_activity, expires_at, status,
                     client_id, tenant_id, visibility, current_tab, metadata, config_overrides)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        session.created_at.isoformat(),
                        session.last_activity.isoformat(),
                        session.expires_at.isoformat(),
                        session.status.value,
                        session.client_id,
                        session.tenant_id,
                        session.visibility.value,
                        session.current_tab,
                        json.dumps(session.metadata) if session.metadata else None,
                        json.dumps(session.config_overrides) if session.config_overrides else None
                    )
                )
                await db.commit()

        # Save new messages (that don't exist yet)
        existing_ids = set()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id FROM messages WHERE session_id = ?",
                (conversation.session_id,)
            )
            existing_ids = {row[0] for row in await cursor.fetchall()}

        for message in conversation.messages:
            if message.id not in existing_ids:
                await self.messages.add(message)

    async def delete(self, session_id: str) -> bool:
        """Delete a conversation by session ID."""
        # Delete messages first
        await self.messages.delete_by_session(session_id)

        # Delete session
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def exists(self, session_id: str) -> bool:
        """Check if a conversation exists."""
        session = await self.sessions.get(session_id)
        return session is not None

    async def find_by_client(
        self,
        client_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Conversation]:
        """Find all conversations for a specific client."""
        sessions = await self.sessions.get_by_client(client_id)

        # Apply limit and offset
        sessions = sessions[offset:offset + limit]

        conversations = []
        for session in sessions:
            messages = await self.messages.get_by_session(session.session_id, limit=50)
            conversations.append(Conversation(session=session, messages=messages))

        return conversations

    async def find_active_sessions(
        self,
        older_than: Optional[datetime] = None
    ) -> List[Session]:
        """Find all active sessions, optionally filtered by age."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if older_than:
                cursor = await db.execute(
                    """
                    SELECT * FROM sessions
                    WHERE status = 'active' AND last_activity < ?
                    """,
                    (older_than.isoformat(),)
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM sessions WHERE status = 'active'"
                )

            rows = await cursor.fetchall()

        return [self.sessions._row_to_session(row) for row in rows]

    async def archive_expired_sessions(self) -> int:
        """Archive all expired sessions."""
        expired = await self.sessions.find_expired(batch_size=1000)

        count = 0
        for session in expired:
            archived = session.archive()
            await self.sessions.update(archived)
            count += 1

        logger.info(f"Archived {count} expired sessions")
        return count
