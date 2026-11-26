import os
import time
import logging
from typing import Optional, Dict, Any, Union, Callable
from functools import lru_cache, wraps
from collections import defaultdict
from threading import Lock
from datetime import datetime, timedelta

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.core.config import settings

# Configure logging for LLM operations
logger = logging.getLogger("llm_factory")
logger.setLevel(logging.INFO)


# ============================================================================
# Custom Exceptions for LLM Operations
# ============================================================================

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Error from a specific LLM provider."""
    def __init__(self, provider: str, message: str, retryable: bool = True):
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


class LLMRateLimitError(LLMProviderError):
    """Rate limit error - should trigger retry with backoff."""
    def __init__(self, provider: str, message: str = "Rate limit exceeded"):
        super().__init__(provider, message, retryable=True)


class LLMTimeoutError(LLMProviderError):
    """Timeout error - may be retryable."""
    def __init__(self, provider: str, message: str = "Request timed out"):
        super().__init__(provider, message, retryable=True)


class LLMAuthenticationError(LLMProviderError):
    """Authentication error - not retryable."""
    def __init__(self, provider: str, message: str = "Authentication failed"):
        super().__init__(provider, message, retryable=False)


class LLMCircuitOpenError(LLMError):
    """Circuit breaker is open - provider temporarily unavailable."""
    def __init__(self, provider: str, reset_time: datetime):
        self.provider = provider
        self.reset_time = reset_time
        super().__init__(
            f"Circuit breaker open for provider '{provider}'. "
            f"Will reset at {reset_time.isoformat()}"
        )


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for LLM providers.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing if service is recovered
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout  # seconds
        self.half_open_max_calls = half_open_max_calls

        self._states: Dict[str, str] = defaultdict(lambda: self.CLOSED)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._last_failure_times: Dict[str, datetime] = {}
        self._half_open_calls: Dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def can_execute(self, provider: str) -> bool:
        """Check if a request can be executed for the given provider."""
        with self._lock:
            state = self._states[provider]

            if state == self.CLOSED:
                return True

            if state == self.OPEN:
                # Check if recovery timeout has passed
                last_failure = self._last_failure_times.get(provider)
                if last_failure:
                    elapsed = (datetime.now() - last_failure).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        # Transition to HALF_OPEN
                        self._states[provider] = self.HALF_OPEN
                        self._half_open_calls[provider] = 0
                        logger.info(f"Circuit breaker for '{provider}' transitioning to HALF_OPEN")
                        return True
                return False

            if state == self.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls[provider] < self.half_open_max_calls:
                    return True
                return False

            return False

    def record_success(self, provider: str) -> None:
        """Record a successful call."""
        with self._lock:
            state = self._states[provider]

            if state == self.HALF_OPEN:
                self._half_open_calls[provider] += 1
                if self._half_open_calls[provider] >= self.half_open_max_calls:
                    # All test calls succeeded, close the circuit
                    self._states[provider] = self.CLOSED
                    self._failure_counts[provider] = 0
                    logger.info(f"Circuit breaker for '{provider}' is now CLOSED")
            elif state == self.CLOSED:
                # Reset failure count on success
                self._failure_counts[provider] = 0

    def record_failure(self, provider: str) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_counts[provider] += 1
            self._last_failure_times[provider] = datetime.now()

            state = self._states[provider]

            if state == self.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._states[provider] = self.OPEN
                logger.warning(f"Circuit breaker for '{provider}' is now OPEN (failure in HALF_OPEN)")
            elif state == self.CLOSED:
                if self._failure_counts[provider] >= self.failure_threshold:
                    self._states[provider] = self.OPEN
                    logger.warning(
                        f"Circuit breaker for '{provider}' is now OPEN "
                        f"(threshold {self.failure_threshold} reached)"
                    )

    def get_state(self, provider: str) -> str:
        """Get the current state for a provider."""
        with self._lock:
            return self._states[provider]

    def get_reset_time(self, provider: str) -> Optional[datetime]:
        """Get the expected reset time for an open circuit."""
        with self._lock:
            if self._states[provider] != self.OPEN:
                return None
            last_failure = self._last_failure_times.get(provider)
            if last_failure:
                return last_failure + timedelta(seconds=self.recovery_timeout)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for all providers."""
        with self._lock:
            return {
                provider: {
                    "state": self._states[provider],
                    "failure_count": self._failure_counts[provider],
                    "last_failure": self._last_failure_times.get(provider),
                }
                for provider in set(list(self._states.keys()) + list(self._failure_counts.keys()))
            }


# Global circuit breaker instance
_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=3
)


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    return _circuit_breaker


# ============================================================================
# LLM Call Metrics Tracking
# ============================================================================

class LLMCallMetrics:
    """Track metrics for LLM calls."""

    def __init__(self):
        self._lock = Lock()
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._total_latency: Dict[str, float] = defaultdict(float)
        self._retry_counts: Dict[str, int] = defaultdict(int)

    def record_call(self, provider: str, use_case: str, latency_ms: float, success: bool, retries: int = 0):
        """Record a completed LLM call."""
        key = f"{provider}:{use_case}"
        with self._lock:
            self._call_counts[key] += 1
            self._total_latency[key] += latency_ms
            self._retry_counts[key] += retries
            if not success:
                self._error_counts[key] += 1

        # Log the call
        log_data = {
            "provider": provider,
            "use_case": use_case,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "retries": retries
        }
        if success:
            logger.info(f"LLM call completed: {log_data}")
        else:
            logger.warning(f"LLM call failed: {log_data}")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        with self._lock:
            stats = {}
            for key in self._call_counts:
                count = self._call_counts[key]
                stats[key] = {
                    "total_calls": count,
                    "error_count": self._error_counts[key],
                    "error_rate": self._error_counts[key] / count if count > 0 else 0,
                    "avg_latency_ms": self._total_latency[key] / count if count > 0 else 0,
                    "total_retries": self._retry_counts[key]
                }
            return stats


# Global metrics instance
_llm_metrics = LLMCallMetrics()


def get_llm_metrics() -> LLMCallMetrics:
    """Get the global LLM metrics instance."""
    return _llm_metrics


# ============================================================================
# Provider health from real calls
# ============================================================================

# Every request has a time limit and is retried with backoff by the provider's own client (the
# OpenAI and Anthropic SDKs both do this). Without a limit, a provider that stops answering would
# hold each request until the connection drops. Generous, because a non-streamed answer (batch
# jobs) arrives all at once when it is finished; set "request_timeout" on a use case in
# config/llm_providers.json to change it.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 2
EMBEDDING_REQUEST_TIMEOUT_SECONDS = 30
_PROVIDERS_WITH_CLIENT_RETRIES = {"openai", "azure", "anthropic"}
# HTTP statuses that mean this particular request was not acceptable
_REQUEST_REFUSED_STATUSES = {400, 413, 422}


class ProviderHealthCallback(BaseCallbackHandler):
    """
    Reports each model call's outcome to the circuit breaker and the call metrics.

    Both used to be updated only when a model object was created, which never fails when the
    provider is down, so the breaker could not open and the metrics counted nothing real. This
    is attached to every model the factory creates and sees the calls themselves (after the
    client's own retries).
    """

    run_inline = True  # just counters: no need for a thread in async code
    raise_error = False

    def __init__(self, provider: str, use_case: str):
        self.provider = provider
        self.use_case = use_case
        self._started: Dict[Any, float] = {}
        self._lock = Lock()

    def _start(self, run_id) -> None:
        with self._lock:
            self._started[run_id] = time.time()

    def _finish(self, run_id, success: bool, provider_at_fault: bool = True) -> None:
        with self._lock:
            started = self._started.pop(run_id, None)
        latency_ms = (time.time() - started) * 1000 if started else 0.0
        breaker = get_circuit_breaker()
        if success:
            breaker.record_success(self.provider)
        elif provider_at_fault:
            breaker.record_failure(self.provider)
        get_llm_metrics().record_call(
            provider=self.provider, use_case=self.use_case, latency_ms=latency_ms, success=success
        )

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        self._finish(run_id, success=True)

    def on_llm_error(self, error, *, run_id, **kwargs):
        # A refused request (such as an image the model cannot take) is counted as a failed call,
        # but must not open the circuit: that would block every other request to the provider
        refused = getattr(error, "status_code", None) in _REQUEST_REFUSED_STATUSES
        self._finish(run_id, success=False, provider_at_fault=not refused)


class LLMFactory:
    PROVIDER_MODULES = {
        "openai": "langchain_openai",
        "anthropic": "langchain_anthropic",
        "azure": "langchain_openai",
        "google": "langchain_google_genai",
        "ollama": "langchain_community.chat_models",
        "huggingface": "langchain_community.llms"
    }

    @staticmethod
    def _import_provider_class(provider: str, class_name: str):
        module_name = LLMFactory.PROVIDER_MODULES.get(provider)
        if not module_name:
            raise ValueError(f"Unknown provider: {provider}")

        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Provider '{provider}' requires package '{module_name}'. "
                f"Install it with: pip install {module_name}"
            ) from e

    @staticmethod
    def _get_api_key(provider: str, provider_config: Dict[str, Any]) -> Optional[str]:
        """
        Get API key for the provider from environment.

        Args:
            provider: Provider name
            provider_config: Provider configuration dict

        Returns:
            API key if found, None otherwise
        """
        api_key_env = provider_config.get("api_key_env")
        if not api_key_env:
            return None

        api_key = os.getenv(api_key_env)
        if not api_key and provider == "openai":
            # Fallback to settings for backward compatibility
            api_key = settings.openai_api_key

        return api_key

    @staticmethod
    def _build_llm_kwargs(
        provider: str,
        model: str,
        provider_config: Dict[str, Any],
        use_case_config: Optional[Dict[str, Any]] = None,
        **overrides
    ) -> Dict[str, Any]:
        """
        Build kwargs for LLM instantiation.

        Args:
            provider: Provider name
            model: Model name
            provider_config: Provider configuration
            use_case_config: Use case configuration
            **overrides: Manual overrides

        Returns:
            Dictionary of kwargs for LLM constructor
        """
        kwargs = {}

        # Get API key
        api_key = LLMFactory._get_api_key(provider, provider_config)
        if api_key:
            if provider == "azure":
                kwargs["api_key"] = api_key
            elif provider in ["openai", "anthropic", "google", "huggingface"]:
                kwargs["api_key"] = api_key if provider != "huggingface" else api_key
                if provider == "huggingface":
                    kwargs["huggingfacehub_api_token"] = api_key

        if provider == "anthropic":
            kwargs["model"] = model or kwargs.get("model_name", model)
        elif provider == "azure":
            kwargs["deployment_name"] = model
            additional_params = provider_config.get("additional_params", {})
            azure_endpoint = os.getenv(additional_params.get("azure_endpoint", ""))
            if azure_endpoint:
                kwargs["azure_endpoint"] = azure_endpoint
                kwargs["base_url"] = azure_endpoint
            api_version = additional_params.get("api_version")
            if api_version:
                kwargs["api_version"] = api_version
        elif provider == "ollama":
            kwargs["model"] = model
            base_url = provider_config.get("base_url")
            if base_url:
                kwargs["base_url"] = base_url
        elif provider == "openai":
            kwargs["model"] = model or kwargs.get("model_name", model)
            # Support custom base URL from environment
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
        elif provider == "huggingface":
            kwargs["repo_id"] = model
        else:
            kwargs["model"] = model or kwargs.get("model_name", model)

        # Use case configuration
        if use_case_config:
            if "temperature" in use_case_config:
                kwargs["temperature"] = use_case_config["temperature"]
            if "max_tokens" in use_case_config:
                kwargs["max_tokens"] = use_case_config["max_tokens"]
            if "request_timeout" in use_case_config:
                kwargs["request_timeout"] = use_case_config["request_timeout"]
            if "streaming" in use_case_config and provider not in ["huggingface"]:
                kwargs["streaming"] = use_case_config["streaming"]

        if provider in _PROVIDERS_WITH_CLIENT_RETRIES:
            if "request_timeout" not in kwargs and "timeout" not in overrides:
                kwargs["timeout"] = DEFAULT_REQUEST_TIMEOUT_SECONDS
            kwargs.setdefault("max_retries", DEFAULT_MAX_RETRIES)

        # Apply overrides
        kwargs.update(overrides)

        return kwargs

    @staticmethod
    def create_llm(
        use_case: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming: Optional[bool] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance with circuit breaker protection and fallback support.

        Args:
            use_case: The use case for this LLM (determines default config)
            model_provider: Override the provider (openai, anthropic, etc.)
            model_name: Override the model name
            temperature: Override temperature setting
            max_tokens: Override max tokens setting
            streaming: Override streaming setting
            **kwargs: Additional arguments passed to the LLM constructor

        Returns:
            BaseChatModel instance

        Raises:
            LLMCircuitOpenError: If the circuit breaker is open for the provider
            RuntimeError: If LLM creation fails and no fallback is available
        """
        llm_config = settings.llm_providers_config
        use_case_config = None
        if use_case:
            use_case_config = llm_config.get("use_cases", {}).get(use_case)
            if not use_case_config:
                logger.warning(f"Use case '{use_case}' not found in configuration")
            else:
                if not model_provider:
                    model_provider = use_case_config.get("provider")
                if not model_name:
                    model_name = use_case_config.get("model")

        if not model_provider:
            model_provider = llm_config.get("default_provider", "openai")

        # Check circuit breaker before attempting to create LLM
        circuit_breaker = get_circuit_breaker()
        if not circuit_breaker.can_execute(model_provider):
            reset_time = circuit_breaker.get_reset_time(model_provider)
            logger.warning(
                f"Circuit breaker OPEN for provider '{model_provider}'. "
                f"Attempting fallback..."
            )
            # Try fallback immediately if circuit is open
            return LLMFactory._try_fallback_provider(
                use_case=use_case,
                use_case_config=use_case_config,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                original_error=LLMCircuitOpenError(model_provider, reset_time),
                **kwargs
            )

        providers = llm_config.get("providers", {})
        provider_config = providers.get(model_provider)

        if not provider_config:
            raise ValueError(f"Provider '{model_provider}' not found in configuration")

        if not provider_config.get("enabled", False):
            raise ValueError(f"Provider '{model_provider}' is not enabled in configuration")
        if not model_name:
            model_name = provider_config.get("default_model")
            if not model_name:
                raise ValueError(f"No model specified for provider '{model_provider}'")
        class_name = provider_config.get("class")
        if not class_name:
            raise ValueError(f"No class specified for provider '{model_provider}'")

        provider_class = LLMFactory._import_provider_class(model_provider, class_name)
        llm_kwargs = LLMFactory._build_llm_kwargs(
            provider=model_provider,
            model=model_name,
            provider_config=provider_config,
            use_case_config=use_case_config,
            **kwargs
        )
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens
        if streaming is not None and model_provider not in ["huggingface"]:
            llm_kwargs["streaming"] = streaming
        llm_kwargs["callbacks"] = list(llm_kwargs.get("callbacks") or []) + [
            ProviderHealthCallback(model_provider, use_case or "default")
        ]

        try:
            start_time = time.time()
            llm = provider_class(**llm_kwargs)

            # Record successful creation
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"LLM created successfully",
                extra={
                    "provider": model_provider,
                    "model": model_name,
                    "use_case": use_case or "default",
                    "latency_ms": round(latency_ms, 2)
                }
            )
            return llm

        except Exception as e:
            # Record failure in circuit breaker
            circuit_breaker.record_failure(model_provider)
            logger.error(
                f"LLM creation failed for provider '{model_provider}': {type(e).__name__}: {e}"
            )

            # Try fallback if configured
            return LLMFactory._try_fallback_provider(
                use_case=use_case,
                use_case_config=use_case_config,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
                original_error=e,
                **kwargs
            )

    @staticmethod
    def _try_fallback_provider(
        use_case: Optional[str],
        use_case_config: Optional[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        streaming: Optional[bool],
        original_error: Exception,
        **kwargs
    ) -> BaseChatModel:
        """
        Try to create LLM using fallback provider.

        Args:
            use_case: Original use case
            use_case_config: Use case configuration
            temperature: Temperature setting
            max_tokens: Max tokens setting
            streaming: Streaming setting
            original_error: The original error that triggered fallback
            **kwargs: Additional arguments

        Returns:
            BaseChatModel from fallback provider

        Raises:
            RuntimeError: If fallback is not configured or also fails
        """
        llm_config = settings.llm_providers_config
        fallback_config = llm_config.get("fallback_strategy", {})

        if fallback_config.get("enabled") and use_case_config:
            fallback_provider = use_case_config.get("fallback_provider")
            fallback_model = use_case_config.get("fallback_model")

            if fallback_provider and fallback_model:
                # Check if fallback provider's circuit is also open
                circuit_breaker = get_circuit_breaker()
                if not circuit_breaker.can_execute(fallback_provider):
                    raise RuntimeError(
                        f"Both primary and fallback providers are unavailable. "
                        f"Original error: {original_error}"
                    )

                logger.warning(
                    f"Trying fallback provider: {fallback_provider}/{fallback_model}"
                )
                try:
                    return LLMFactory.create_llm(
                        model_provider=fallback_provider,
                        model_name=fallback_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        streaming=streaming,
                        **kwargs
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Both primary and fallback providers failed. "
                        f"Primary: {original_error}, Fallback: {fallback_error}"
                    ) from fallback_error

        raise RuntimeError(
            f"Failed to create LLM: {original_error}"
        ) from original_error

    @staticmethod
    def get_embedding_function(
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> tuple[Embeddings, int]:
        llm_config = settings.llm_providers_config
        embeddings_config = llm_config.get("use_cases", {}).get("embeddings", {})
        if not model_provider:
            model_provider = embeddings_config.get("provider") or llm_config.get("default_embedding_provider", "openai")

        if not model_name:
            model_name = embeddings_config.get("model")
            if not model_name:
                provider_config = llm_config.get("providers", {}).get(model_provider, {})
                model_name = provider_config.get("default_embedding_model")

        if not model_name:
            model_name = settings.openai_embedding_model

        providers = llm_config.get("providers", {})
        provider_config = providers.get(model_provider, {})

        if model_provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            api_key = LLMFactory._get_api_key(model_provider, provider_config)
            kwargs = {
                "model": model_name,
                "request_timeout": EMBEDDING_REQUEST_TIMEOUT_SECONDS,
                "max_retries": DEFAULT_MAX_RETRIES,
            }
            if api_key:
                kwargs["api_key"] = api_key
            # Support custom base URL from environment
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url

            embeddings = OpenAIEmbeddings(**kwargs)

            # Get dimensions from config
            embedding_models = provider_config.get("embedding_models", {})
            model_config = embedding_models.get(model_name, {})
            dimensions = model_config.get("dimensions", 1536)

            return embeddings, dimensions

        elif model_provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            # HuggingFace dimensions vary by model, default to 768
            return embeddings, 768

        else:
            raise ValueError(f"Embeddings not supported for provider: {model_provider}")

    @staticmethod
    @lru_cache(maxsize=10)
    def get_available_providers() -> list:
        
        llm_config = settings.llm_providers_config
        providers = llm_config.get("providers", {})
        return [name for name, config in providers.items() if config.get("enabled", False)]

    @staticmethod
    def get_provider_models(provider: str) -> list:
        llm_config = settings.llm_providers_config
        provider_config = llm_config.get("providers", {}).get(provider, {})
        models = provider_config.get("models", {})
        return list(models.keys())

    @staticmethod
    def get_use_cases() -> list:

        llm_config = settings.llm_providers_config
        use_cases = llm_config.get("use_cases", {})
        return list(use_cases.keys())


# Backward compatibility functions
def create_llm(model_provider: str = "openai", model_name: str = None,
               temperature: float = 0.0, max_tokens: int = None, streaming: bool = True):
    """
    Legacy function for backward compatibility.
    Prefer using LLMFactory.create_llm() with use_case parameter.
    """
    if model_name is None:
        model_name = settings.openai_model
    return LLMFactory.create_llm(
        model_provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else 1024,
        streaming=streaming
    )


def get_embedding_function(model_provider: str = "openai", model_name: str = None):
    """
    Legacy function for backward compatibility.
    """
    if model_name is None:
        model_name = settings.openai_embedding_model
    return LLMFactory.get_embedding_function(
        model_provider=model_provider,
        model_name=model_name
    )
