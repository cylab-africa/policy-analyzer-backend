"""
Dependency Injection Container

This module provides a simple dependency injection container for managing
application dependencies. It enables:

1. Loose coupling between components
2. Easy testing with mock implementations
3. Per-tenant/client configuration
4. Lazy initialization of expensive resources

Usage:
    from app.core.container import get_container

    container = get_container()
    conversation_service = container.conversation_service()
    query_handler = container.query_handler()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, TypeVar, Generic
from functools import cached_property
from threading import Lock

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Singleton(Generic[T]):
    """
    Thread-safe singleton wrapper with lazy initialization.
    """

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._instance: Optional[T] = None
        self._lock = Lock()

    def get(self) -> T:
        """Get the singleton instance, creating it if necessary."""
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def reset(self) -> None:
        """Reset the singleton, forcing recreation on next access."""
        with self._lock:
            self._instance = None


def _get_db_path(filename: str) -> str:
    """Get database path using settings.db_data_dir."""
    import os
    db_dir = settings.db_data_dir
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, filename)


@dataclass
class ContainerConfig:
    """
    Configuration for the dependency container.

    Allows customization of database paths, cache settings, etc.
    """
    # Database paths
    sqlite_path: str = field(default_factory=lambda: _get_db_path("conversations.db"))
    legacy_sqlite_path: str = field(default_factory=lambda: _get_db_path("chat_history.db"))

    # Session settings
    session_ttl_hours: int = 24
    max_messages_in_context: int = 10
    max_context_chars: int = 8000

    # Cleanup settings
    cleanup_expired_after_days: int = 7
    archive_idle_after_hours: int = 2

    # Feature flags
    use_legacy_storage: bool = True  # Backward compatibility with old SQLite
    enable_event_sourcing: bool = False  # Future feature
    enable_caching: bool = True

    # Per-client configuration overrides
    client_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class Container:
    """
    Dependency Injection Container.

    Manages the creation and lifecycle of application components.
    Supports both singleton and factory patterns.
    """

    def __init__(self, config: Optional[ContainerConfig] = None):
        self.config = config or ContainerConfig()
        self._singletons: Dict[str, Singleton] = {}
        self._lock = Lock()

    # =========================================================================
    # Repository Layer
    # =========================================================================

    def message_repository(self):
        """Get the message repository."""
        if "message_repo" not in self._singletons:
            from app.infrastructure.sqlite_repositories import SQLiteMessageRepository
            self._singletons["message_repo"] = Singleton(
                lambda: SQLiteMessageRepository(self.config.sqlite_path)
            )
        return self._singletons["message_repo"].get()

    def session_repository(self):
        """Get the session repository."""
        if "session_repo" not in self._singletons:
            from app.infrastructure.sqlite_repositories import SQLiteSessionRepository
            self._singletons["session_repo"] = Singleton(
                lambda: SQLiteSessionRepository(self.config.sqlite_path)
            )
        return self._singletons["session_repo"].get()

    def conversation_repository(self):
        """Get the conversation repository."""
        if "conversation_repo" not in self._singletons:
            from app.infrastructure.sqlite_repositories import SQLiteConversationRepository
            self._singletons["conversation_repo"] = Singleton(
                lambda: SQLiteConversationRepository(
                    self.config.sqlite_path,
                    self.message_repository(),
                    self.session_repository()
                )
            )
        return self._singletons["conversation_repo"].get()

    # =========================================================================
    # Service Layer
    # =========================================================================

    def conversation_service(self):
        """Get the conversation service."""
        if "conversation_service" not in self._singletons:
            from app.services.conversation_service import ConversationService
            from datetime import timedelta

            self._singletons["conversation_service"] = Singleton(
                lambda: ConversationService(
                    conversation_repo=self.conversation_repository(),
                    message_repo=self.message_repository(),
                    session_repo=self.session_repository(),
                    default_session_ttl=timedelta(hours=self.config.session_ttl_hours),
                    max_messages_in_context=self.config.max_messages_in_context,
                    max_context_chars=self.config.max_context_chars
                )
            )
        return self._singletons["conversation_service"].get()

    def llm_factory(self):
        """Get the LLM factory."""
        if "llm_factory" not in self._singletons:
            from app.services.llm_factory import LLMFactory
            self._singletons["llm_factory"] = Singleton(lambda: LLMFactory)
        return self._singletons["llm_factory"].get()

    def prompt_manager(self):
        """Get the prompt manager."""
        if "prompt_manager" not in self._singletons:
            from app.services.prompt_manager import get_prompt_manager
            self._singletons["prompt_manager"] = Singleton(
                lambda: get_prompt_manager(self.mongo_client())
            )
        return self._singletons["prompt_manager"].get()

    def entity_detector(self):
        """Get the entity detector."""
        if "entity_detector" not in self._singletons:
            from app.services.entity_detector import get_entity_detector
            self._singletons["entity_detector"] = Singleton(get_entity_detector)
        return self._singletons["entity_detector"].get()

    def filter_builder(self):
        """Get the filter builder."""
        if "filter_builder" not in self._singletons:
            from app.services.filter_builder import get_filter_builder
            self._singletons["filter_builder"] = Singleton(
                lambda: get_filter_builder(self.entity_detector())
            )
        return self._singletons["filter_builder"].get()

    def vector_stores(self):
        """Get the vector stores."""
        if "vector_stores" not in self._singletons:
            from app.services.vector_store import get_vector_stores_v2
            self._singletons["vector_stores"] = Singleton(get_vector_stores_v2)
        return self._singletons["vector_stores"].get()

    # =========================================================================
    # Infrastructure Layer
    # =========================================================================

    def mongo_client(self):
        """Get the MongoDB client."""
        if "mongo_client" not in self._singletons:
            from app.db.base_db import mongo_client
            self._singletons["mongo_client"] = Singleton(lambda: mongo_client)
        return self._singletons["mongo_client"].get()

    def legacy_chat_history(self):
        """
        Get the legacy chat history store for backward compatibility.

        This provides access to the old SQLiteHistoryStore during migration.
        """
        if "legacy_chat_history" not in self._singletons:
            from app.db.chat_history import database
            self._singletons["legacy_chat_history"] = Singleton(lambda: database)
        return self._singletons["legacy_chat_history"].get()

    # =========================================================================
    # Query Handler (Main Entry Point)
    # =========================================================================

    def query_handler(self):
        """
        Get the query handler.

        This is the main orchestrator for query processing.
        """
        if "query_handler" not in self._singletons:
            from app.services.query_handler import DynamicQueryHandler
            self._singletons["query_handler"] = Singleton(
                lambda: DynamicQueryHandler(db_client=self.mongo_client())
            )
        return self._singletons["query_handler"].get()

    # =========================================================================
    # Monitoring & Observability
    # =========================================================================

    def circuit_breaker(self):
        """Get the circuit breaker."""
        if "circuit_breaker" not in self._singletons:
            from app.services.llm_factory import get_circuit_breaker
            self._singletons["circuit_breaker"] = Singleton(get_circuit_breaker)
        return self._singletons["circuit_breaker"].get()

    def llm_metrics(self):
        """Get the LLM metrics tracker."""
        if "llm_metrics" not in self._singletons:
            from app.services.llm_factory import get_llm_metrics
            self._singletons["llm_metrics"] = Singleton(get_llm_metrics)
        return self._singletons["llm_metrics"].get()

    # =========================================================================
    # Container Management
    # =========================================================================

    def reset(self, component: Optional[str] = None) -> None:
        """
        Reset singleton(s), forcing recreation on next access.

        Args:
            component: Specific component to reset, or None for all
        """
        with self._lock:
            if component:
                if component in self._singletons:
                    self._singletons[component].reset()
            else:
                for singleton in self._singletons.values():
                    singleton.reset()
                self._singletons.clear()

        logger.info(f"Reset container components: {component or 'all'}")

    def get_client_config(self, client_id: str) -> Dict[str, Any]:
        """
        Get configuration overrides for a specific client.

        Args:
            client_id: The client identifier

        Returns:
            Client-specific configuration or empty dict
        """
        return self.config.client_configs.get(client_id, {})

    def register_client_config(
        self,
        client_id: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Register configuration overrides for a client.

        Args:
            client_id: The client identifier
            config: Configuration overrides
        """
        self.config.client_configs[client_id] = config
        logger.info(f"Registered config for client {client_id}")


# ============================================================================
# Global Container Instance
# ============================================================================

_container: Optional[Container] = None
_container_lock = Lock()


def get_container(config: Optional[ContainerConfig] = None) -> Container:
    """
    Get the global container instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The container instance
    """
    global _container

    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container(config)
                logger.info("Initialized dependency injection container")

    return _container


def reset_container() -> None:
    """Reset the global container, useful for testing."""
    global _container

    with _container_lock:
        if _container:
            _container.reset()
        _container = None

    logger.info("Reset global container")
