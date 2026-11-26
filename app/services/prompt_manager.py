"""
Prompt Manager - Hybrid prompt configuration system

This module provides centralized prompt management with a three-tier hierarchy:
1. JSON defaults (config/prompts.json)
2. Database overrides (MongoDB prompt_overrides collection)
3. Environment variable overrides

Load order: JSON → DB → ENV (later overrides earlier)
"""

import os
import re
from typing import Dict, Any, Optional
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path

from app.core.config import settings, PROJECT_ROOT


class PromptManager:
    """
    Manages system prompts with hybrid configuration support.

    Features:
    - Load defaults from JSON configuration
    - Override from database (runtime configuration)
    - Override from environment variables (deployment-specific)
    - Caching with TTL for performance
    - Template variable substitution
    """

    def __init__(self, db_client=None, cache_ttl_seconds: int = 3600):
        """
        Initialize the PromptManager.

        Args:
            db_client: MongoDB client for database overrides (optional)
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.db_client = db_client
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._prompts_config = None

    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration from JSON (cached)."""
        if self._prompts_config is None:
            self._prompts_config = settings.prompts_config
        return self._prompts_config

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                return value
            else:
                # Expired, remove from cache
                del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp."""
        self._cache[key] = (value, datetime.now())

    def clear_cache(self) -> None:
        """Clear all cached prompts."""
        self._cache.clear()

    def _load_from_json(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """Load prompt from JSON configuration."""
        prompts_config = self._load_prompts_config()
        system_prompts = prompts_config.get("system_prompts", {})
        return system_prompts.get(prompt_key)

    def _load_from_db(self, prompt_key: str, use_case: Optional[str] = None) -> Optional[str]:
        """
        Load prompt override from database.

        Args:
            prompt_key: The prompt identifier
            use_case: Optional use case filter

        Returns:
            Prompt text if found, None otherwise
        """
        if not self.db_client:
            return None

        try:
            collection = self.db_client.get_collection("prompt_overrides")
            query = {"prompt_key": prompt_key}
            if use_case:
                query["use_case"] = use_case

            result = collection.find_one(query, sort=[("updated_at", -1)])
            if result:
                return result.get("prompt_text")
        except Exception as e:
            print(f"Warning: Failed to load prompt from database: {e}")

        return None

    def _load_from_env(self, prompt_key: str) -> Optional[str]:
        """
        Load prompt override from environment variable.

        Environment variable format: PROMPT_{PROMPT_KEY_UPPER}
        Example: PROMPT_RAG_RESPONSE
        """
        env_key = f"PROMPT_{prompt_key.upper()}"
        return os.getenv(env_key)

    def get_prompt(
        self,
        prompt_key: str,
        use_case: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> str:
        """
        Get a prompt with the hybrid loading strategy.

        Load order: JSON defaults → DB overrides → ENV overrides

        Args:
            prompt_key: The prompt identifier (e.g., "rag_response")
            use_case: Optional use case for specialized overrides
            variables: Template variables to substitute in the prompt
            use_cache: Whether to use caching

        Returns:
            The final prompt text with variables substituted

        Raises:
            ValueError: If prompt_key not found in any source
        """
        cache_key = f"{prompt_key}:{use_case or 'default'}"

        # Check cache first
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                return self._substitute_variables(cached, variables)

        # 1. Load from JSON (default)
        prompt_config = self._load_from_json(prompt_key)
        if not prompt_config:
            raise ValueError(f"Prompt '{prompt_key}' not found in configuration")

        prompt_text = prompt_config.get("prompt", "")

        # 2. Override from database if available
        db_override = self._load_from_db(prompt_key, use_case)
        if db_override:
            prompt_text = db_override

        # 3. Override from environment variable if set
        env_override = self._load_from_env(prompt_key)
        if env_override:
            prompt_text = env_override

        # Cache the result
        if use_cache:
            self._set_cache(cache_key, prompt_text)

        # Substitute variables
        return self._substitute_variables(prompt_text, variables)

    def get_prompt_config(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """
        Get full prompt configuration including metadata.

        Args:
            prompt_key: The prompt identifier

        Returns:
            Prompt configuration dict or None if not found
        """
        return self._load_from_json(prompt_key)

    def _substitute_variables(self, prompt: str, variables: Optional[Dict[str, Any]]) -> str:
        """
        Substitute template variables in prompt text.

        Supports Python format strings: {variable_name}

        Args:
            prompt: Prompt text with template variables
            variables: Dictionary of variable values

        Returns:
            Prompt with variables substituted
        """
        if not variables:
            return prompt

        # A placeholder with no value is left as written, and every other one is still filled.
        # (Before, one missing variable meant the whole template was returned unfilled, so the
        # model got "{context}" and "{question}" instead of the documents and the question.)
        missing = []

        class _KeepMissing(dict):
            def __missing__(self, key):
                missing.append(key)
                return "{" + key + "}"

        try:
            result = prompt.format_map(_KeepMissing(variables))
        except (ValueError, IndexError, AttributeError):
            # Braces that are not placeholders (an example JSON object, say) break str.format.
            # Fill only the {name} placeholders that have a value and leave the rest alone.
            result = re.sub(
                r"\{(\w+)\}",
                lambda m: str(variables[m.group(1)]) if m.group(1) in variables else m.group(0),
                prompt,
            )
        if missing:
            print(f"Warning: no value for template variable(s) {sorted(set(missing))} in prompt")
        return result

    def get_template(self, template_key: str) -> Optional[str]:
        """
        Get a prompt template.

        Args:
            template_key: The template identifier

        Returns:
            Template text or None if not found
        """
        prompts_config = self._load_prompts_config()
        templates = prompts_config.get("templates", {})
        template_value = templates.get(template_key)

        # Check if it's a file path
        if template_value and template_value.endswith(".txt"):
            template_path = Path(template_value)
            if not template_path.is_absolute():
                # Relative to project root
                template_path = PROJECT_ROOT / template_value

            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()

        return template_value

    def save_override(
        self,
        prompt_key: str,
        prompt_text: str,
        use_case: Optional[str] = None,
        updated_by: str = "system"
    ) -> bool:
        """
        Save a prompt override to the database.

        Args:
            prompt_key: The prompt identifier
            prompt_text: New prompt text
            use_case: Optional use case scope
            updated_by: User or system that made the change

        Returns:
            True if successful, False otherwise
        """
        if not self.db_client:
            print("Warning: No database client available for saving prompt override")
            return False

        try:
            collection = self.db_client.get_collection("prompt_overrides")

            override_doc = {
                "prompt_key": prompt_key,
                "prompt_text": prompt_text,
                "use_case": use_case,
                "updated_at": datetime.utcnow(),
                "updated_by": updated_by
            }

            # Upsert: update if exists, insert if not
            query = {"prompt_key": prompt_key}
            if use_case:
                query["use_case"] = use_case

            collection.update_one(
                query,
                {"$set": override_doc},
                upsert=True
            )

            # Clear cache for this prompt
            cache_key = f"{prompt_key}:{use_case or 'default'}"
            if cache_key in self._cache:
                del self._cache[cache_key]

            return True

        except Exception as e:
            print(f"Error saving prompt override: {e}")
            return False

    def delete_override(self, prompt_key: str, use_case: Optional[str] = None) -> bool:
        """
        Delete a prompt override from the database (revert to JSON default).

        Args:
            prompt_key: The prompt identifier
            use_case: Optional use case scope

        Returns:
            True if successful, False otherwise
        """
        if not self.db_client:
            return False

        try:
            collection = self.db_client.get_collection("prompt_overrides")

            query = {"prompt_key": prompt_key}
            if use_case:
                query["use_case"] = use_case

            result = collection.delete_many(query)

            # Clear cache
            cache_key = f"{prompt_key}:{use_case or 'default'}"
            if cache_key in self._cache:
                del self._cache[cache_key]

            return result.deleted_count > 0

        except Exception as e:
            print(f"Error deleting prompt override: {e}")
            return False

    def list_overrides(self, use_case: Optional[str] = None) -> list:
        """
        List all prompt overrides in the database.

        Args:
            use_case: Optional filter by use case

        Returns:
            List of override documents
        """
        if not self.db_client:
            return []

        try:
            collection = self.db_client.get_collection("prompt_overrides")

            query = {}
            if use_case:
                query["use_case"] = use_case

            overrides = list(collection.find(query).sort("updated_at", -1))
            return overrides

        except Exception as e:
            print(f"Error listing prompt overrides: {e}")
            return []


# Global instance (lazy initialization)
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(db_client=None) -> PromptManager:
    """
    Get or create the global PromptManager instance.

    Args:
        db_client: Optional MongoDB client for database overrides

    Returns:
        PromptManager instance
    """
    global _prompt_manager

    if _prompt_manager is None:
        from app.db.base_db import mongo_client
        _prompt_manager = PromptManager(db_client=db_client or mongo_client)

    return _prompt_manager
