import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from functools import lru_cache

from pydantic import field_validator
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    app_name: str = Field(default="Legal Document RAG API", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="us.anthropic.claude-sonnet-5", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    mongodb_connection_string: str = Field(..., env="MONGODB_CONNECTION_STRING")
    mongodb_database_name: str = Field(default="policy_analyzer_db", env="MONGODB_DATABASE_NAME")
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    serpapi_api_key: Optional[str] = Field(default=None, env="SERPAPI_API_KEY")
    api_key: str = Field(..., env="API_KEY")
    # Operator-only key, sent in the X-Admin-Key header. Unlike API_KEY it is never given to the
    # website, so it can guard listing every batch job and deleting any job. Unset disables both.
    admin_api_key: Optional[str] = Field(default=None, env="ADMIN_API_KEY")

    model_provider: str = Field(default="openai", env="MODEL_PROVIDER")
    embedding_provider: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    embedding_model_id: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL_ID")
    supported_tabs_raw: str = Field(default="data-protection,digital-id", env="supported_tabs")
    vector_store: str = Field(default="qdrant", env="VECTOR_STORE")
    qdrant_host: str = Field(default="http://localhost:6333", env="QDRANT_HOST")
    qdrant_collection_name_public: str = Field(default="public_vectors", env="QDRANT_COLLECTION_NAME_PUBLIC")
    qdrant_collection_name_private: str = Field(default="private_vectors", env="QDRANT_COLLECTION_NAME_PRIVATE")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    # mongodb_collection_name_public: str = Field(
    #     default="public_vectors", 
    #     env="MONGODB_COLLECTION_NAME_PUBLIC"
    # )
    # mongodb_index_name_public: str = Field(
    #     default="public_index", 
    #     env="MONGODB_INDEX_NAME_PUBLIC"
    # )

    # # Private vector store config  
    # mongodb_collection_name_private: str = Field(
    #     default="private_vectors", 
    #     env="MONGODB_COLLECTION_NAME_PRIVATE"
    # )
    # mongodb_index_name_private: str = Field(
    #     default="private_index", 
    #     env="MONGODB_INDEX_NAME_PRIVATE"
    # )

    
    db_data_dir: str = Field(default=".", env="DB_DATA_DIR")
    upload_directory: str = Field(default="uploads", env="UPLOAD_DIRECTORY")
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB default
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    use_s3: bool = Field(default=True, env="USE_S3")
    max_tokens: int = Field(default=1024, env="MAX_TOKENS")
    temperature: float = Field(default=0.0, env="TEMPERATURE")
    AWS_REGION: Optional[str] = Field(default=None, env="AWS_REGION")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    S3_BUCKET_NAME: Optional[str] = Field(default=None, env="S3_BUCKET_NAME")
    S3_ENDPOINT: Optional[str] = Field(default=None, env="S3_ENDPOINT")
    UPLOAD_EXPIRATION: Optional[int] = Field(default=3600, env="UPLOAD_EXPIRATION")  # in seconds
    DOWNLOAD_EXPIRATION: Optional[int] = Field(default=86400, env="DOWNLOAD_EXPIRATION")  # in seconds
    CDN_ENDPOINT: Optional[str] = Field(default=None, env="CDN_ENDPOINT")
    MAX_FILE_SIZE: Optional[int] = Field(default=10485760, env="MAX_FILE_SIZE")
    ALLOWED_FILE_TYPES: Optional[str] = Field(default=None, env="ALLOWED_FILE_TYPES")

    # Email configuration for batch query notifications
    SMTP_SERVER: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USERNAME: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    # Address batch emails are sent from. Unset means SMTP_USERNAME, the account that logs in to
    # the mail server: most providers (Gmail included) reject or rewrite mail sent "from" any other
    # address unless that address is set up as an alias of the account.
    SENDER_EMAIL: Optional[str] = Field(default=None, env="SENDER_EMAIL")
    SENDER_NAME: str = Field(default="PDP Africa Analyzer", env="SENDER_NAME")

    # Batch query job configuration
    BATCH_JOBS_COLLECTION: str = Field(default="batch_query_jobs", env="BATCH_JOBS_COLLECTION")
    BATCH_PDF_FOLDER: str = Field(default="batch_reports", env="BATCH_PDF_FOLDER")

    # Interactive API docs (/api/docs) and the route list (/openapi.json) are public when enabled,
    # so they are off by default. Turn them on for local development only.
    enable_api_docs: bool = Field(default=False, env="ENABLE_API_DOCS")

    # Browser access. The website calls this API from its own server, so by default no other
    # origin is allowed. Set a comma-separated list only if a browser must call this API directly.
    cors_allowed_origins: str = Field(default="", env="CORS_ALLOWED_ORIGINS")

    # /api/documents/folder-process reads files from the server's disk. It is switched off unless
    # this names the one directory it may read from.
    ingest_folder_root: Optional[str] = Field(default=None, env="INGEST_FOLDER_ROOT")

    # Rate limits per client (0 switches a limit off). See app/middleware/rate_limit.py.
    query_rate_limit_per_minute: int = Field(default=20, env="QUERY_RATE_LIMIT_PER_MINUTE")
    batch_upload_rate_limit_per_hour: int = Field(default=10, env="BATCH_UPLOAD_RATE_LIMIT_PER_HOUR")
    upload_rate_limit_per_hour: int = Field(default=10, env="UPLOAD_RATE_LIMIT_PER_HOUR")
    # Where request counts are kept: "redis" (shared by all workers and replicas; falls back to
    # memory while Redis is down) or "memory" (per process).
    rate_limit_store: str = Field(default="redis", env="RATE_LIMIT_STORE")
    # Comma-separated exact addresses or hostnames of the website's server (as this API sees it),
    # e.g. its container name on a shared Docker network. Requests from them may name the visitor
    # in X-Client-IP, so each visitor gets their own limit. Ranges are refused (a network range
    # includes its gateway, through which outside callers could forge the header). Leave empty
    # and every request through the website shares one limit.
    trusted_proxy_ips: str = Field(default="", env="TRUSTED_PROXY_IPS")

    # Redis configuration for caching
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")


    @field_validator("supported_tabs_raw")
    def validate_supported_tabs(cls, v):
        if not v:
            raise ValueError("supported_tabs cannot be empty")
        return v
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse allowed CORS origins from a comma-separated string. '*' is never accepted."""
        origins = [o.strip().rstrip("/") for o in self.cors_allowed_origins.split(",")]
        return [o for o in origins if o and o != "*"]

    @property
    def supported_tabs(self) -> List[str]:
        """Parse supported tabs from comma-separated string."""
        return [tab.strip() for tab in self.supported_tabs_raw.split(",") if tab.strip()]
    
    def get_mongodb_collection_name(self, tab: str) -> str:
        """Get MongoDB collection name for a specific tab."""
        tab_clean = tab.upper().replace('-', '_')
        return getattr(self, f"mongodb_collection_name_{tab.replace('-', '_')}", f"{tab}_vectors")
    
    def get_mongodb_index_name(self, tab: str) -> str:
        """Get MongoDB index name for a specific tab."""
        tab_clean = tab.upper().replace('-', '_')
        return getattr(self, f"mongodb_index_name_{tab.replace('-', '_')}", f"{tab}_index")

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_json_config(config_name: str) -> Dict[str, Any]:
        """Load a JSON configuration file with caching."""
        config_path = CONFIG_DIR / f"{config_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")

    @property
    def application_config(self) -> Dict[str, Any]:
        """Load application configuration from JSON."""
        return self._load_json_config("application")

    @property
    def prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration from JSON."""
        return self._load_json_config("prompts")

    @property
    def llm_providers_config(self) -> Dict[str, Any]:
        """Load LLM providers configuration from JSON."""
        return self._load_json_config("llm_providers")

    @property
    def tabs_config(self) -> Dict[str, Any]:
        """
        Get tabs configuration (supports dynamic loading).

        Uses TabManager for dynamic tab loading from database overrides.
        Falls back to JSON config if TabManager not available.
        """
        try:
            from app.services.tab_manager import get_tab_manager
            tab_manager = get_tab_manager()

            # Build tabs dict from all tabs
            all_tabs = tab_manager.list_all_tabs()
            tabs_dict = {}
            for tab_item in all_tabs:
                tab_name = tab_item.get("tab_name")
                tabs_dict[tab_name] = tab_item.get("config", {})

            return tabs_dict
        except Exception as e:
            # Fallback to JSON config if TabManager fails
            print(f"Warning: TabManager unavailable, using JSON config: {e}")
            return self.application_config.get("tabs", {})

    @property
    def metadata_config(self) -> Dict[str, Any]:
        """Get metadata configuration from application config."""
        return self.application_config.get("metadata", {})

    @property
    def retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration from application config."""
        return self.application_config.get("retrieval", {})

    def get_tab_config(self, tab: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific tab (supports dynamic loading).

        Uses TabManager for dynamic tab loading from database overrides.
        Falls back to JSON config if TabManager not available.
        """
        try:
            from app.services.tab_manager import get_tab_manager
            tab_manager = get_tab_manager()
            return tab_manager.get_tab_config(tab)
        except Exception as e:
            # Fallback to JSON config
            print(f"Warning: TabManager unavailable, using JSON config: {e}")
            return self.tabs_config.get(tab)

    def get_enabled_tabs(self) -> List[str]:
        """
        Get list of enabled tabs (supports dynamic loading).

        Uses TabManager for dynamic tab loading from database overrides.
        Falls back to JSON config if TabManager not available.
        """
        try:
            from app.services.tab_manager import get_tab_manager
            tab_manager = get_tab_manager()
            return tab_manager.get_enabled_tabs()
        except Exception as e:
            # Fallback to JSON config
            print(f"Warning: TabManager unavailable, using JSON config: {e}")
            return [tab for tab, config in self.tabs_config.items() if config.get("enabled", True)]

    def get_llm_use_case_config(self, use_case: str) -> Optional[Dict[str, Any]]:
        """Get LLM configuration for a specific use case."""
        return self.llm_providers_config.get("use_cases", {}).get(use_case)

    def get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific LLM provider."""
        return self.llm_providers_config.get("providers", {}).get(provider)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()



def get_settings() -> Settings:
    """Get the application settings instance."""
    return settings


def get_openai_api_key() -> str:
    """Get OpenAI API key."""
    return settings.openai_api_key


def get_mongodb_connection_string() -> str:
    """Get MongoDB connection string."""
    return settings.mongodb_connection_string


def get_supported_tabs() -> List[str]:
    """Get list of supported tabs."""
    return settings.supported_tabs

def get_env_var(key: str, default=None):
    """
    Get environment variable with optional default.
    This function is provided for backward compatibility during migration.
    """
    return os.getenv(key, default)

