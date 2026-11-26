import sqlite3
import json
import os
from typing import List, Dict, Optional

from app.core.config import settings


def _get_default_db_path() -> str:
    """Get default database path using settings.db_data_dir."""
    db_dir = settings.db_data_dir
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "chat_history.db")


class SQLiteHistoryStore:
    def __init__(self, db_path: str = None):
        """Initialize SQLite database for conversation history using only session_id"""
        self.db_path = db_path or _get_default_db_path()
        self.init_database()
    
    def init_database(self):
        """Create the database table if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create index for faster session queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON messages(session_id, timestamp)
            """)
            
            conn.commit()
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to the session history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute(
                """INSERT INTO messages (session_id, role, content, metadata) 
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, content, metadata_json)
            )
            conn.commit()
    
    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """A session's messages, oldest first. With a limit, only the latest ones."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Newest first, so that a limit keeps the latest messages. The ID breaks ties:
            # timestamps are only precise to the second. A limit of -1 means no limit in SQLite.
            cursor.execute(
                """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (session_id, limit or -1),
            )
            rows = cursor.fetchall()[::-1]
            
            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "metadata": json.loads(row[3]) if row[3] else None
                }
                for row in rows
            ]
            
    def save_session_metadata(self, session_id: str, key: str, value):
        """Save metadata for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, metadata FROM messages 
                   WHERE session_id = ? AND role = 'system' AND content = 'session_metadata'
                   ORDER BY timestamp DESC LIMIT 1""",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                metadata_id = row[0]
                metadata = json.loads(row[1]) if row[1] else {}
                metadata[key] = value
                
                cursor.execute(
                    """UPDATE messages SET metadata = ? WHERE id = ?""",
                    (json.dumps(metadata), metadata_id)
                )
            else:
                metadata = {key: value}
                cursor.execute(
                    """INSERT INTO messages (session_id, role, content, metadata) 
                       VALUES (?, 'system', 'session_metadata', ?)""",
                    (session_id, json.dumps(metadata))
                )
            
            conn.commit()
    
    def get_session_metadata(self, session_id: str, key: str = None):
        """Retrieve metadata for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT metadata FROM messages 
                   WHERE session_id = ? AND role = 'system' AND content = 'session_metadata' 
                   ORDER BY timestamp DESC LIMIT 1""",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if not row or not row[0]:
                return None if key else {}
                
            metadata = json.loads(row[0])
            return metadata.get(key) if key else metadata
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, 
                       MIN(timestamp) as first_message,
                       MAX(timestamp) as last_message,
                       COUNT(*) as message_count
                FROM messages 
                GROUP BY session_id
                ORDER BY last_message DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [
                {
                    "session_id": row[0],
                    "first_message": row[1],
                    "last_message": row[2],
                    "message_count": row[3]
                }
                for row in rows
            ]
    
    def search_sessions(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search for sessions containing specific terms"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, content, timestamp
                FROM messages
                WHERE content LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{search_term}%", limit))
            
            rows = cursor.fetchall()
            return [
                {
                    "session_id": row[0],
                    "preview": row[1][:100] + "..." if len(row[1]) > 100 else row[1],
                    "timestamp": row[2]
                }
                for row in rows
            ]
    
    def clear_session(self, session_id: str):
        """Clear all messages from a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()

database = SQLiteHistoryStore()