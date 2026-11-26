import copy
import threading
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from functools import lru_cache

from app.core.config import settings


class TabManager:
    def __init__(self, db_client=None, collection_name: str = "tab_configs"):
        self.db_client = db_client
        self.collection_name = collection_name
        self._cache = {}
        self._cache_timestamp = None
        # The tab list is read several times for every question (tab detection, the relevance
        # check, the answer prompt) and each read was a database query on the request's thread.
        # Keep it for a minute; saving or deleting an override clears it at once.
        self._all_tabs: Optional[List[Dict[str, Any]]] = None
        self._all_tabs_at = 0.0
        self._all_tabs_seconds = 60.0
        self._all_tabs_lock = threading.Lock()

    def _get_collection(self):
        if not self.db_client:
            return None
        return self.db_client.get_collection(self.collection_name)

    def get_tab_config(self, tab_name: str) -> Dict[str, Any]:
        # Try cache first
        cache_key = f"tab_{tab_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try database override
        db_config = self._load_from_database(tab_name)
        if db_config:
            self._cache[cache_key] = db_config
            return db_config

        # Load from JSON config
        json_config = self._load_from_json(tab_name)
        if json_config:
            self._cache[cache_key] = json_config
            return json_config

        return None

    def _load_from_json(self, tab_name: str) -> Optional[Dict[str, Any]]:
        """
        Load tab config from JSON file.

        Args:
            tab_name: Tab identifier

        Returns:
            Tab configuration or None
        """
        try:
            app_config = settings.application_config
            tabs = app_config.get("tabs", {})
            return tabs.get(tab_name)
        except Exception as e:
            print(f"Error loading tab from JSON: {e}")
            return None

    def _load_from_database(self, tab_name: str) -> Optional[Dict[str, Any]]:
        """
        Load tab config from database.

        Args:
            tab_name: Tab identifier

        Returns:
            Tab configuration or None
        """
        collection = self._get_collection()
        if collection is None:
            return None

        try:
            doc = collection.find_one({"tab_name": tab_name})
            if doc:
                return doc.get("config", {})
        except Exception as e:
            print(f"Error loading tab from database: {e}")

        return None

    def list_all_tabs(self) -> List[Dict[str, Any]]:
        """
        List all available tabs (JSON + database), cached for a minute.

        Returns:
            List of tab configurations
        """
        with self._all_tabs_lock:
            if self._all_tabs is not None and time.monotonic() - self._all_tabs_at < self._all_tabs_seconds:
                return copy.deepcopy(self._all_tabs)
        tabs = self._load_all_tabs()
        with self._all_tabs_lock:
            self._all_tabs, self._all_tabs_at = copy.deepcopy(tabs), time.monotonic()
        return tabs

    def _load_all_tabs(self) -> List[Dict[str, Any]]:
        tabs = {}

        # Load from JSON
        try:
            app_config = settings.application_config
            json_tabs = app_config.get("tabs", {})
            for tab_name, config in json_tabs.items():
                tabs[tab_name] = {
                    "tab_name": tab_name,
                    "config": config,
                    "source": "json"
                }
        except Exception as e:
            print(f"Error loading tabs from JSON: {e}")

        # Overlay database overrides
        collection = self._get_collection()
        if collection is not None:
            try:
                db_tabs = collection.find()
                for doc in db_tabs:
                    tab_name = doc.get("tab_name")
                    tabs[tab_name] = {
                        "tab_name": tab_name,
                        "config": doc.get("config", {}),
                        "source": "database",
                        "updated_at": doc.get("updated_at"),
                        "updated_by": doc.get("updated_by")
                    }
            except Exception as e:
                print(f"Error loading tabs from database: {e}")

        return list(tabs.values())

    def get_enabled_tabs(self) -> List[str]:
        """
        Get list of enabled tab names.

        Returns:
            List of enabled tab names
        """
        all_tabs = self.list_all_tabs()
        enabled = []

        for tab in all_tabs:
            config = tab.get("config", {})
            if config.get("enabled", True):
                enabled.append(tab.get("tab_name"))

        # Sort by priority
        enabled.sort(key=lambda t: self.get_tab_config(t).get("priority", 999))

        return enabled

    def save_tab_override(
        self,
        tab_name: str,
        config: Dict[str, Any],
        updated_by: str = "admin"
    ) -> bool:
        """
        Save tab configuration override to database.

        Args:
            tab_name: Tab identifier
            config: Tab configuration
            updated_by: User making the change

        Returns:
            True if successful
        """
        collection = self._get_collection()
        if collection is None:
            print("Warning: No database client available")
            return False

        # Validate config
        validation_error = self._validate_tab_config(config)
        if validation_error:
            raise ValueError(f"Invalid tab config: {validation_error}")

        try:
            result = collection.update_one(
                {"tab_name": tab_name},
                {
                    "$set": {
                        "tab_name": tab_name,
                        "config": config,
                        "updated_at": datetime.now(timezone.utc),
                        "updated_by": updated_by
                    }
                },
                upsert=True
            )

            # Invalidate cache
            self._invalidate(tab_name)

            return True
        except Exception as e:
            print(f"Error saving tab override: {e}")
            return False

    def delete_tab_override(self, tab_name: str) -> bool:
        """
        Delete tab override from database (reverts to JSON default).

        Args:
            tab_name: Tab identifier

        Returns:
            True if deleted
        """
        collection = self._get_collection()
        if collection is None:
            return False

        try:
            result = collection.delete_one({"tab_name": tab_name})

            # Invalidate cache
            self._invalidate(tab_name)

            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting tab override: {e}")
            return False

    def toggle_tab(self, tab_name: str, enabled: bool, updated_by: str = "admin") -> bool:
        """
        Enable or disable a tab.

        Args:
            tab_name: Tab identifier
            enabled: Whether to enable the tab
            updated_by: User making the change

        Returns:
            True if successful
        """
        # Get current config
        config = self.get_tab_config(tab_name)
        if not config:
            return False

        # Update enabled status
        config["enabled"] = enabled

        # Save override
        return self.save_tab_override(tab_name, config, updated_by)

    def _validate_tab_config(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Validate tab configuration.

        Args:
            config: Tab configuration to validate

        Returns:
            Error message if invalid, None if valid
        """
        required_fields = ["display_name", "description"]

        for field in required_fields:
            if field not in config:
                return f"Missing required field: {field}"

        # Validate enabled flag
        if "enabled" in config and not isinstance(config["enabled"], bool):
            return "enabled must be a boolean"

        # Validate priority
        if "priority" in config and not isinstance(config["priority"], int):
            return "priority must be an integer"

        # Validate metadata_filters
        if "metadata_filters" in config:
            if not isinstance(config["metadata_filters"], list):
                return "metadata_filters must be a list"

        return None

    def _invalidate(self, tab_name: str) -> None:
        self._cache.pop(f"tab_{tab_name}", None)
        with self._all_tabs_lock:
            self._all_tabs = None

    def clear_cache(self) -> None:
        """Clear the tab configuration cache."""
        self._cache.clear()
        self._cache_timestamp = None
        with self._all_tabs_lock:
            self._all_tabs = None
        print("Tab cache cleared")

    def validate_all_tabs(self) -> Dict[str, List[str]]:
        """
        Validate all tab configurations.

        Returns:
            Dictionary with 'valid' and 'invalid' tab lists
        """
        all_tabs = self.list_all_tabs()
        valid = []
        invalid = []

        for tab in all_tabs:
            tab_name = tab.get("tab_name")
            config = tab.get("config", {})

            error = self._validate_tab_config(config)
            if error:
                invalid.append({"tab_name": tab_name, "error": error})
            else:
                valid.append(tab_name)

        return {
            "valid": valid,
            "invalid": invalid
        }

    def get_tab_metadata_filters(self, tab_name: str) -> List[str]:
        """
        Get metadata filters configured for a tab.

        Args:
            tab_name: Tab identifier

        Returns:
            List of metadata filter keys
        """
        config = self.get_tab_config(tab_name)
        if not config:
            return []

        return config.get("metadata_filters", [])


# Global instance
_tab_manager: Optional[TabManager] = None


def get_tab_manager(db_client=None) -> TabManager:
    """
    Get or create the global TabManager instance.

    Args:
        db_client: Optional MongoDB client

    Returns:
        TabManager instance
    """
    global _tab_manager

    if _tab_manager is None:
        if db_client is None:
            from app.db.base_db import mongo_client
            db_client = mongo_client

        _tab_manager = TabManager(db_client=db_client)

    return _tab_manager
