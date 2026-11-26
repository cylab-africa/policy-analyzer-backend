"""
Unified Query Handler - Orchestrates the complete query processing pipeline

This module provides a unified interface for processing user queries, integrating:
- LLM Factory (dynamic LLM selection)
- Prompt Manager (dynamic prompts)
- Entity Detection (automatic metadata extraction)
- Filter Building (metadata filtering)
- Vector Store (unified search)
- Conversation History (context management)
"""

import asyncio
import contextvars
import logging
import re
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Iterable, Set, Tuple
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import lru_cache, wraps

from app.core.config import settings
from app.services.llm_factory import LLMFactory, get_circuit_breaker, get_llm_metrics
from app.services.prompt_manager import get_prompt_manager
from app.services.entity_detector import get_entity_detector
from app.services.filter_builder import get_filter_builder
from app.services.vector_store import get_vector_stores_v2

# New conversation management imports
from app.domain.models import MessageRole, Visibility, Session
from app.services.conversation_service import (
    ConversationService,
    get_conversation_service,
    SessionNotFoundError,
    SessionExpiredError,
    UnauthorizedAccessError
)


# ============================================================================
# Whole-word term matching
# ============================================================================
#
# Language detection and the web-search triggers look for words and phrases in the question.
# Matching them as plain substrings finds them inside other words ("now" in "know", "count" in
# "country", "ces" in "services", "sur" in "ensure"), so English questions were taken for French
# and web search ran for most questions. Terms only match as whole words or phrases.

@lru_cache(maxsize=None)
def _term_regex(terms: Tuple[str, ...]) -> "re.Pattern[str]":
    # Longest first, so a phrase wins over a word it starts with ("combien de" over "combien")
    alternatives = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


def _normalize(text: str) -> str:
    return text.lower().replace("’", "'")  # typographic apostrophe, as in "j’ai"


def matched_terms(text: str, terms: Iterable[str]) -> Set[str]:
    """The distinct terms that appear in ``text`` as whole words or phrases (case-insensitive)."""
    terms = tuple(terms)
    if not terms:
        return set()
    return {m.group(0) for m in _term_regex(terms).finditer(_normalize(text))}


# Words that only French uses, or that a policy question in English would not contain.
# One is enough to call the question French.
_STRONG_FRENCH = (
    "quelle", "quel", "quels", "quelles",  # what/which
    "est-ce que", "qu'est-ce",  # is it, what is it
    "pourquoi",  # why
    "en matière de", "à propos de",  # regarding, about
    "j'ai", "j'aime", "j'utilise",  # I have, I like, I use
    "c'est", "n'est", "d'accord",  # it is, is not, agree
    "s'il vous plaît", "s'il te plaît",  # please
    "pouvez-vous", "peux-tu", "puis-je",  # can you, may I
    "camerounais", "camerounaise",  # Cameroonian
    "français", "française",  # French
    "utiliser", "répondre", "expliquer",  # use, respond, explain
)

# Common French words; two different ones are needed. Words that are also ordinary English, or
# abbreviations in this field, are left out: "au" (African Union), "sa" (South Africa), "non",
# "son", "ton", "par" and "pays".
_WEAK_FRENCH = (
    "comment", "donc", "comme", "mais",
    "dans", "sur", "avec", "pour",
    "de la", "du", "des", "le", "la", "les", "une", "aux",
    "et", "ou", "en", "est", "sont", "qui", "que", "quoi",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "mon", "ma", "mes", "ta", "tes", "ses",
    "cette", "cet", "ces",
    "peut", "peux", "peuvent", "être", "avoir", "faire",
    "récente", "récent", "dernière", "dernier",
    "politique", "loi", "lois", "réglementation", "données",
    "merci", "bonjour", "bonsoir", "oui", "bien",
    "anglais",
)


# ============================================================================
# Structured Logger Configuration
# ============================================================================

class StructuredLogger:
    """
    Structured logging wrapper that adds context to all log messages.
    Provides consistent logging format for query processing operations.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Add handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Per request, not per process: concurrent requests each see their own query_id and
        # session_id. (A plain attribute was shared, so requests overwrote each other's context.)
        self._context_var: contextvars.ContextVar = contextvars.ContextVar(
            f"log_context_{name}", default={}
        )

    def set_context(self, **kwargs) -> None:
        """Set context that will be included in all subsequent log messages."""
        self._context_var.set({**self._context_var.get(), **kwargs})

    def clear_context(self) -> None:
        """Clear the current context."""
        self._context_var.set({})

    def _format_message(self, message: str, extra: Dict[str, Any] = None) -> str:
        """Format message with context and extra data."""
        all_data = {**self._context_var.get(), **(extra or {})}
        if all_data:
            data_str = " | ".join(f"{k}={v}" for k, v in all_data.items())
            return f"{message} | {data_str}"
        return message

    def info(self, message: str, **extra) -> None:
        self.logger.info(self._format_message(message, extra))

    def warning(self, message: str, **extra) -> None:
        self.logger.warning(self._format_message(message, extra))

    def error(self, message: str, **extra) -> None:
        self.logger.error(self._format_message(message, extra))

    def debug(self, message: str, **extra) -> None:
        self.logger.debug(self._format_message(message, extra))


# Global logger instance for query handler
query_logger = StructuredLogger("query_handler")


# ============================================================================
# Query Metrics Tracking
# ============================================================================

class QueryMetrics:
    """
    Track and record metrics for query processing.
    Provides detailed timing and success/failure tracking.
    """

    def __init__(self):
        self.start_time: float = 0
        self.timings: Dict[str, float] = {}
        self.query_id: str = ""
        self.session_id: str = ""
        self.tab: str = ""
        self.visibility: str = ""
        self.doc_count: int = 0
        self.web_search_used: bool = False
        self.success: bool = True
        self.error_message: Optional[str] = None

    def start(self, query_id: str, session_id: str = "", tab: str = "", visibility: str = "public"):
        """Start tracking a new query."""
        self.start_time = time.time()
        self.query_id = query_id
        self.session_id = session_id
        self.tab = tab
        self.visibility = visibility
        self.timings = {}
        self.success = True
        self.error_message = None

    def record_timing(self, stage: str, duration_ms: float):
        """Record timing for a specific stage."""
        self.timings[stage] = duration_ms

    def mark_error(self, error_message: str):
        """Mark the query as failed."""
        self.success = False
        self.error_message = error_message

    def get_total_time_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "tab": self.tab,
            "visibility": self.visibility,
            "total_time_ms": round(self.get_total_time_ms(), 2),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "doc_count": self.doc_count,
            "web_search_used": self.web_search_used,
            "success": self.success,
            "error_message": self.error_message
        }


@contextmanager
def track_stage(metrics: QueryMetrics, stage_name: str):
    """Context manager to track timing for a processing stage."""
    start = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start) * 1000
        metrics.record_timing(stage_name, duration_ms)
        query_logger.debug(f"Stage completed: {stage_name}", duration_ms=round(duration_ms, 2))


class DynamicQueryHandler:
    """
    Unified query handler that orchestrates the complete query processing pipeline.

    Now integrates with ConversationService for proper session and message management.
    Maintains backward compatibility with legacy storage while enabling new features.
    """

    def __init__(
        self,
        db_client=None,
        conversation_collection_name: str = "conversations",
        conversation_service: Optional[ConversationService] = None,
        use_new_conversation_system: bool = True
    ):
        """
        Initialize the query handler.

        Args:
            db_client: MongoDB client for legacy storage
            conversation_collection_name: MongoDB collection name
            conversation_service: New conversation service (optional, will create if not provided)
            use_new_conversation_system: Whether to use the new conversation management
        """
        self.db_client = db_client
        self.conversation_collection = conversation_collection_name
        self.use_new_conversation_system = use_new_conversation_system

        # Initialize components (lazy loading for performance)
        self._prompt_manager = None
        self._entity_detector = None
        self._filter_builder = None
        self._vector_stores = None
        self._conversation_service = conversation_service

        # Web search configuration (can be overridden per-query)
        self.default_enable_web_search = True
        self.default_search_tool_type = "tavily"

        # Multilingual indicators for web search triggers
        # Structure: Easy to extend with more languages (Portuguese, Arabic, Swahili, etc.)
        self._temporal_indicators = {
            # Matched as whole words (see matched_terms), so inflected forms are listed
            "en": [  # English
                "latest", "recent", "recently", "current", "currently", "now", "today",
                "this year", "this month", "update", "updates", "updated", "news",
                "2024", "2025", "2026"
            ],
            "fr": [  # French
                "récent", "récente", "récents", "récentes", "récemment",
                "dernier", "dernière", "derniers", "dernières",
                "actuel", "actuelle", "actuels", "actuelles", "actuellement", "maintenant",
                "aujourd'hui", "cette année", "ce mois", "mise à jour", "mises à jour",
                "nouvelles", "nouveau", "nouveaux", "nouvelle", "2024", "2025", "2026"
            ]
            # Easy to add more languages:
            # "pt": [...],  # Portuguese
            # "ar": [...],  # Arabic
            # "sw": [...],  # Swahili
        }

        self._fact_indicators = {
            "en": [  # English
                "how many", "statistics", "statistic", "data on", "numbers for",
                "percentage", "figure", "figures", "population", "count", "counts", "total",
                "number of", "how much"
            ],
            "fr": [  # French
                "combien", "combien de", "statistiques", "données sur",
                "chiffres", "pourcentage", "population", "nombre",
                "nombre de", "total", "quantité"
            ]
            # Easy to add more languages:
            # "pt": [...],  # Portuguese
            # "ar": [...],  # Arabic
            # "sw": [...],  # Swahili
        }

    @property
    def prompt_manager(self):
        """Lazy-load prompt manager."""
        if self._prompt_manager is None:
            self._prompt_manager = get_prompt_manager(self.db_client)
        return self._prompt_manager

    @property
    def entity_detector(self):
        """Lazy-load entity detector."""
        if self._entity_detector is None:
            self._entity_detector = get_entity_detector()
        return self._entity_detector

    @property
    def filter_builder(self):
        """Lazy-load filter builder."""
        if self._filter_builder is None:
            self._filter_builder = get_filter_builder(self.entity_detector)
        return self._filter_builder

    @property
    def vector_stores(self):
        """Lazy-load vector stores."""
        if self._vector_stores is None:
            self._vector_stores = get_vector_stores_v2()
        return self._vector_stores

    @property
    def conversation_service(self) -> ConversationService:
        """Lazy-load conversation service."""
        if self._conversation_service is None:
            self._conversation_service = get_conversation_service()
        return self._conversation_service

    async def detect_tab(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        enabled_tabs = settings.get_enabled_tabs()

        if not enabled_tabs:
            return "general"

        # Format conversation history
        history_text = ""
        if conversation_history:
            recent = conversation_history[-5:]  # Last 5 messages
            history_text = "\n".join([
                f"User: {msg.get('user_query', '')}\nAssistant: {msg.get('assistant_response', '')}"
                for msg in recent
            ])

        # Get prompt
        prompt = self.prompt_manager.get_prompt(
            "tab_detection",
            variables={
                "supported_tabs": ", ".join(enabled_tabs),
                "conversation_history": history_text or "No previous conversation",
                "query": query
            }
        )

        # Create LLM for tab detection
        llm = LLMFactory.create_llm(use_case="tab_detection")

        try:
            response = await llm.ainvoke(prompt)
            detected_tab = response.content.strip().lower()

            # Validate detected tab
            if detected_tab in enabled_tabs:
                return detected_tab
            else:
                return enabled_tabs[0] if enabled_tabs else "general"

        except Exception as e:
            print(f"Error detecting tab: {e}")
            return enabled_tabs[0] if enabled_tabs else "general"

    async def retrieve_documents(
        self,
        query: str,
        visibility: str = "public",
        tab: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_auto_filters: bool = True
    ) -> List[Dict[str, Any]]:
        # Entity detection, embedding and the vector search all use synchronous clients. Run
        # them in a worker thread so that other requests are served meanwhile.
        return await asyncio.to_thread(
            self._retrieve_documents_sync,
            query, visibility, tab, metadata_filters, use_auto_filters,
        )

    def _retrieve_documents_sync(
        self,
        query: str,
        visibility: str = "public",
        tab: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_auto_filters: bool = True
    ) -> List[Dict[str, Any]]:
        store = self.vector_stores.get(visibility)
        if not store:
            raise ValueError(f"Invalid visibility: {visibility}")

        if use_auto_filters:
            detection_config = settings.metadata_config.get("detection_config", {})
            detected_entities = self.entity_detector.detect_all_entities(query, detection_config)
            is_comparative = detected_entities.get("is_comparative", False)
            is_quantitative = detected_entities.get("is_quantitative", False)
            detected_countries = detected_entities.get("country", [])
            retrieval_config = settings.retrieval_config
            default_k = retrieval_config.get("default_k", 3)

            # OPTIMIZATION: Reduced top_k values for faster retrieval
            if is_quantitative:
                top_k = 30  # Reduced from 50
            elif is_comparative:
                top_k = 6   # Reduced from 10
            else:
                top_k = 5   # Reduced from 8

            # Automatic filtering
            filter_combinations = self.filter_builder.build_filter_for_vectorstore(
                query, tab, metadata_filters
            )

            # Dynamic max_total_docs: For comparative queries with multiple countries,
            # ensure we retrieve enough documents to cover all countries
            max_total_docs = None  # Use default from config
            if is_comparative and len(detected_countries) >= 2:
                # Ensure at least top_k documents per country
                min_docs_needed = len(detected_countries) * top_k
                default_max = retrieval_config.get("max_total_docs", 12)
                if min_docs_needed > default_max:
                    max_total_docs = min_docs_needed
                    print(f"[QUERY_HANDLER] Comparative query with {len(detected_countries)} countries - adjusting max_total_docs to {max_total_docs}")

            documents = store.search_with_filter_combinations(
                query, filter_combinations, top_k, max_total_docs
            )
        else:
            # Simple search with provided filters
            documents = store.search_with_metadata(
                query, tab, metadata_filters
            )

        return documents

    def _detect_query_language(self, query: str) -> str:
        """
        Simple language detection for query (currently supports English and French).

        Args:
            query: User query

        Returns:
            Language code: "en" for English, "fr" for French, defaults to "en"
        """
        strong_count = len(matched_terms(query, _STRONG_FRENCH))
        weak_count = len(matched_terms(query, _WEAK_FRENCH))

        # Strong evidence: any strong indicator, or 2+ different weak ones
        if strong_count >= 1 or weak_count >= 2:
            print(f"[LANGUAGE_DETECTION] French detected: strong={strong_count}, weak={weak_count}")
            return "fr"

        print(f"[LANGUAGE_DETECTION] English (default): strong={strong_count}, weak={weak_count}")
        return "en"  # Default to English unless clear French evidence

    def _get_out_of_scope_message(self, language: str = "en") -> str:
        """
        Get out-of-scope message in the appropriate language.

        Args:
            language: Language code ("en" or "fr")

        Returns:
            Localized out-of-scope message
        """
        # Written as Markdown, which the website renders. List items must start with "- ": lines
        # that start with a "•" character are run together into one paragraph.
        messages = {
            "en": """I specialize in data protection and digital identity topics in the African context.

Your question does not appear to be related to these areas.

I can help you with questions about:

- Data protection and privacy laws (GDPR, national data protection acts, etc.)
- Digital identity systems and authentication
- Personal data governance and security
- Consent management and user rights
- Biometric systems and digital ID cards
- Cybersecurity and data breach regulations

Please ask a question related to these topics for the most helpful response.""",

            "fr": """Je me spécialise dans la protection des données et l'identité numérique dans le contexte africain.

Votre question ne semble pas être liée à ces domaines.

Je peux vous aider avec des questions sur :

- Les lois de protection des données et de la vie privée (RGPD, lois nationales de protection des données, etc.)
- Les systèmes d'identité numérique et d'authentification
- La gouvernance et la sécurité des données personnelles
- La gestion du consentement et les droits des utilisateurs
- Les systèmes biométriques et les cartes d'identité numériques
- La cybersécurité et la réglementation des violations de données

Veuillez poser une question relative à ces sujets pour obtenir une réponse utile."""

            # Easy to add more languages:
            # "pt": """...""",  # Portuguese
            # "ar": """...""",  # Arabic
            # "sw": """...""",  # Swahili
        }

        return messages.get(language, messages["en"])

    def _has_ambiguous_references(self, query: str) -> bool:
        """
        Fast check if query contains ambiguous references that need context.
        Returns True if query likely needs enhancement, False otherwise.
        """
        query_lower = query.lower()

        # Ambiguous pronouns and determiners
        ambiguous_words = [
            'this', 'that', 'these', 'those', 'it', 'they', 'them',
            'he', 'she', 'his', 'her', 'their', 'its'
        ]

        # Check for ambiguous references at word boundaries
        words = query_lower.split()
        has_reference = any(word in ambiguous_words for word in words)

        # Also check for phrases like "based on" which often need context
        contextual_phrases = [
            'based on', 'according to', 'as mentioned', 'as discussed',
            'from earlier', 'from above', 'like before', 'previously'
        ]
        has_contextual = any(phrase in query_lower for phrase in contextual_phrases)

        return has_reference or has_contextual

    async def _enhance_query_with_context(self, query: str, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Enhance ambiguous queries with context from conversation history.
        For example: "based on this comparison" → "based on Kenya and Rwanda comparison"
        """
        try:
            print(f"[QUERY_ENHANCEMENT] Inside _enhance_query_with_context")
            print(f"[QUERY_ENHANCEMENT] Query to enhance: '{query}'")
            print(f"[QUERY_ENHANCEMENT] History messages count: {len(conversation_history)}")

            # Build recent context - prioritize most recent exchange
            recent_messages = conversation_history[-4:]  # Last 2 exchanges
            context_lines = []

            # Mark the most recent exchange clearly
            total_msgs = len(recent_messages)
            for i, msg in enumerate(recent_messages):
                role = "User" if msg.get("role") in ("human", "user") else "Assistant"
                content = msg.get("content", "")[:400]  # More context for better understanding

                # Mark most recent exchange
                if i >= total_msgs - 2:
                    context_lines.append(f"[MOST RECENT] {role}: {content}")
                else:
                    context_lines.append(f"{role}: {content}")
                print(f"[QUERY_ENHANCEMENT] Context message: {role} - {content[:80]}...")

            context_str = "\n".join(context_lines)
            print(f"[QUERY_ENHANCEMENT] Built context string ({len(context_str)} chars)")

            llm = LLMFactory.create_llm(use_case="tab_detection")  # Lightweight LLM
            print(f"[QUERY_ENHANCEMENT] Created LLM instance")

            enhancement_prompt = f"""Given the conversation history and current query, rewrite the query to be more explicit by resolving any ambiguous references (like "it", "this", "these", "that", etc.).

CONVERSATION HISTORY (oldest to newest):
{context_str}

CURRENT QUERY: {query}

CRITICAL: Focus on the MOST RECENT topic in the conversation (the last exchange). The user is most likely referring to that.

If the query contains ambiguous references:
- "it" usually refers to the most recently discussed system, country, or topic
- "this", "that" usually refers to the most recent subject
- "these", "those" refer to recently mentioned items

IMPORTANT RULES:
1. Use the MOST RECENT context (last assistant response) to resolve ambiguity
2. If discussing Cameroon's ID system most recently, "it" refers to Cameroon's system, NOT earlier topics
3. Return ONLY the rewritten query, nothing else. No explanation.

Examples:
- After discussing Cameroon's digital ID: "What can I use it for?" → "What can I use Cameroon's digital identity card for?"
- After discussing Kenya's law: "Is it mandatory?" → "Is Kenya's data protection law mandatory?"
- "tell me about Tanzania's law" → "tell me about Tanzania's law" (already clear)

REWRITTEN QUERY:"""

            print(f"[QUERY_ENHANCEMENT] Invoking LLM for enhancement...")
            response = await llm.ainvoke(enhancement_prompt)
            enhanced = response.content.strip()
            print(f"[QUERY_ENHANCEMENT] LLM response: '{enhanced}'")

            # Basic validation - don't return if it's way too long or seems wrong
            if len(enhanced) > len(query) * 3 or len(enhanced) < 5:
                print(f"[QUERY_ENHANCEMENT] Validation failed - enhanced query too long ({len(enhanced)} chars) or too short. Using original.")
                return query

            print(f"[QUERY_ENHANCEMENT] Validation passed - returning enhanced query")
            return enhanced

        except Exception as e:
            print(f"[QUERY_ENHANCEMENT] Error enhancing query: {e}. Using original query.")
            import traceback
            traceback.print_exc()
            return query

    async def _is_query_relevant_to_domain(self, query: str, conversation_history: list = None) -> bool:

        try:
            enabled_tabs = settings.get_enabled_tabs()
            supported_tabs_str = ", ".join(enabled_tabs) if enabled_tabs else "data protection and digital identity"

            llm = LLMFactory.create_llm(use_case="tab_detection")  # Reuse lightweight LLM

            # Build context from conversation history if available
            context_str = ""
            if conversation_history and len(conversation_history) > 0:
                print(f"[RELEVANCE_CHECK] Found {len(conversation_history)} messages in history")
                recent_messages = conversation_history[-4:]  # Last 2 exchanges (4 messages)
                context_lines = []
                for msg in recent_messages:
                    role = "User" if msg.get("role") in ("human", "user") else "Assistant"
                    content = msg.get("content", "")[:300]  # Increase limit to see more context
                    context_lines.append(f"{role}: {content}")
                    print(f"[RELEVANCE_CHECK] Context message: {role} - {content[:80]}...")
                context_str = f"\n\nRecent conversation context:\n" + "\n".join(context_lines)
                print(f"[RELEVANCE_CHECK] Built context string with {len(context_lines)} messages")
            else:
                print(f"[RELEVANCE_CHECK] No conversation history available")

            relevance_prompt = f"""You are a relevance checker for a specialized knowledge base about {supported_tabs_str} in the African context.

                Determine if the following user query is relevant to these topics:
                - Data protection and privacy laws (GDPR, national data protection acts, etc.)
                - Digital identity systems and authentication
                - Personal data governance and security
                - Consent management and user rights
                - Biometric systems and digital ID cards
                - Cybersecurity and data breach regulations
                - Comparative analysis of data protection laws between countries
                - Regulatory compliance and operational considerations for organizations
                - USE CASES, APPLICATIONS, and BENEFITS of digital ID or data protection systems
                - Requirements, processes, and procedures for obtaining IDs or complying with laws
{context_str}

                Current user query: "{query}"

                CRITICAL RULES FOR FOLLOW-UP QUESTIONS:
                1. If there is conversation context above, this is a FOLLOW-UP question
                2. Follow-up questions like "What can I use it for?", "How does it work?", "What are the benefits?", "Tell me more", "What are the requirements?" are ALWAYS RELEVANT if the previous conversation was about digital ID or data protection
                3. Questions about practical applications, use cases, benefits, requirements of previously discussed systems are RELEVANT
                4. When in doubt with a follow-up question, mark as RELEVANT
                5. Only mark as NOT_RELEVANT if the query is CLEARLY off-topic (weather, food, entertainment, sports, etc.)

                SPECIAL CASE - META-QUESTIONS:
                6. Questions ABOUT the conversation itself are RELEVANT (complaints, clarifications, language requests)
                   - "Why did you respond in English?" → RELEVANT (meta-question, user wants correction)
                   - "J'ai posé en français pourquoi tu réponds en anglais?" → RELEVANT (language complaint)
                   - "Can you explain that again?" → RELEVANT (clarification request)
                   - "Please respond in French" → RELEVANT (language preference)

                Note: Queries may be in English or French (or other languages). Focus on the TOPIC, not the language.

                Respond with ONLY "RELEVANT" or "NOT_RELEVANT". No explanation needed.

                Examples WITH context about digital ID:
                - "What can I use it for?" → RELEVANT (asking about use cases)
                - "What are the benefits?" → RELEVANT (follow-up about benefits)
                - "How do I apply?" → RELEVANT (asking about process)
                - "Tell me more" → RELEVANT (continuation)
                - "What are the requirements?" → RELEVANT (practical question)
                - "Is it mandatory?" → RELEVANT (policy question)

                Examples WITHOUT context:
                - "What is Kenya's data protection law?" → RELEVANT
                - "Quelle est la loi de protection des données du Cameroun?" → RELEVANT
                - "How many countries have digital ID systems?" → RELEVANT
                - "What's the weather in Paris?" → NOT_RELEVANT
                - "Best pizza restaurants in Lagos" → NOT_RELEVANT
                """

            response = await llm.ainvoke(relevance_prompt)
            is_relevant = "RELEVANT" in response.content.upper() and "NOT_RELEVANT" not in response.content.upper()

            if not is_relevant:
                print(f"[RELEVANCE_CHECK] Query is NOT relevant to domain: '{query[:100]}'")
            else:
                print(f"[RELEVANCE_CHECK] Query is relevant to domain")

            return is_relevant

        except Exception as e:
            print(f"[RELEVANCE_CHECK] Error checking relevance: {e}. Assuming relevant.")
            return True  # Default to relevant on error to avoid blocking valid queries

    def _should_use_web_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        enable_web_search: bool = True
    ) -> bool:
        """
        Determine if web search should be used as fallback.

        Args:
            query: The user's query
            documents: Retrieved documents from vector store
            enable_web_search: Whether web search is enabled

        Returns:
            True if web search should be used, False otherwise
        """
        if not enable_web_search:
            return False

        # Trigger web search if very few documents retrieved
        if len(documents) < 2:
            print(f"[WEB_SEARCH] Triggering web search - only {len(documents)} documents retrieved")
            return True

        # Combine temporal indicators from all supported languages
        # This allows detecting temporal queries in English, French, and any future languages
        temporal_indicators = []
        for lang_indicators in self._temporal_indicators.values():
            temporal_indicators.extend(lang_indicators)

        # Combine fact indicators from all supported languages
        fact_indicators = []
        for lang_indicators in self._fact_indicators.values():
            fact_indicators.extend(lang_indicators)

        # Whole words only: "now" must not match "know", nor "count" match "country"
        temporal = matched_terms(query, temporal_indicators)
        if temporal:
            print(f"[WEB_SEARCH] Triggering web search - temporal indicator: {sorted(temporal)}")
            return True

        # Trigger web search for fact-seeking queries with low document count (any language)
        if len(documents) < 4:
            facts = matched_terms(query, fact_indicators)
            if facts:
                print(f"[WEB_SEARCH] Triggering web search - fact indicator: {sorted(facts)} with low doc count")
                return True

        return False

    async def perform_web_search(self, query: str, search_tool_type: str = "tavily", max_results: int = 3) -> str:

        try:
            from app.services.search_tools import SearchToolFactory
            import time

            search_start = time.time()
            print(f"[WEB_SEARCH] Performing web search with {search_tool_type}...")

            search_results = await asyncio.to_thread(
                SearchToolFactory.perform_search,
                query=query,
                tool_type=search_tool_type,
                max_results=max_results
            )

            search_end = time.time()
            print(f"[WEB_SEARCH] Web search completed in {search_end - search_start:.2f}s")

            return search_results

        except Exception as e:
            print(f"[WEB_SEARCH] Error during web search: {str(e)}")
            return f"Error during web search: {str(e)}"

    async def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        streaming: bool = False,
        tab: Optional[str] = None,
        web_search_used: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        context_text = self._format_context(context_documents)
        available_tabs = self._format_available_tabs()

        # Format conversation history for context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]  # Last 2 exchanges
            history_lines = []
            for msg in recent_history:
                role = "User" if msg.get("role") in ("human", "user") else "Assistant"
                content = msg.get("content", "")[:500]  # Limit to avoid token overflow
                history_lines.append(f"{role}: {content}")
            conversation_context = "\n\nRECENT CONVERSATION:\n" + "\n\n".join(history_lines) + "\n"
            print(f"[GENERATE_RESPONSE] Including conversation context with {len(recent_history)} messages")

        # Format current tab display name
        current_tab_display = "General"
        if tab:
            tab_config = settings.tabs_config.get(tab, {})
            current_tab_display = tab_config.get("display_name", tab.replace("-", " ").title())

        detection_config = settings.metadata_config.get("detection_config", {})
        # May call the LLM (synchronous client): keep it off the event loop. The result for this
        # question is cached by the detector, so this is normally a lookup.
        detected_entities = await asyncio.to_thread(
            self.entity_detector.detect_all_entities, query, detection_config
        )
        is_comparative = detected_entities.get("is_comparative", False)
        is_quantitative = detected_entities.get("is_quantitative", False)
        detected_countries = detected_entities.get("country", [])

        # Detect query language to enforce response language
        query_language = self._detect_query_language(query)
        language_map = {
            "en": "English",
            "fr": "French",
            "pt": "Portuguese",
            "ar": "Arabic",
            "sw": "Swahili"
        }
        detected_language_name = language_map.get(query_language, "English")
        print(f"[LANGUAGE_DETECTION] Query language detected: {detected_language_name} ({query_language})")

        # Start with language instruction
        formatting_instructions = f"""

        🎯 RESPONSE LANGUAGE REQUIREMENT:
        The user's question is written in {detected_language_name}.
        You MUST write your ENTIRE response in {detected_language_name}.
        DO NOT respond in French unless the question itself is in French.
        DO NOT mix languages - use ONLY {detected_language_name} throughout your answer.
        """

        if is_quantitative:
            formatting_instructions += f"""

            IMPORTANT FORMATTING INSTRUCTION:
            This is a QUANTITATIVE query asking for counts or statistics.

            Please format your response to clearly answer the quantitative question:
            1. START with the direct numerical answer (e.g., "X countries have a digital ID policy")
            2. Provide a COMPLETE LIST of all countries/items found (use bullet points or numbered list)
            3. Include relevant percentages or statistics if applicable
            4. If available, mention any notable patterns or regional distribution

            Example format:
            **Answer:** 15 African countries have implemented digital ID policies.

            **Countries with Digital ID Policies:**
            1. Country A - Brief detail
            2. Country B - Brief detail
            ...

            **Key Statistics:**
            - Regional distribution: ...
            - Percentage of total: ...
            """
            print(f"[QUERY_HANDLER] Quantitative response formatting enabled")

        elif is_comparative and len(detected_countries) >= 2:
            # Comparative queries: Use table format
            countries_str = ", ".join(detected_countries)
            formatting_instructions += f"""

            IMPORTANT FORMATTING INSTRUCTION:
            This is a COMPARATIVE query comparing: {countries_str}

            Please format your response as a MARKDOWN TABLE with the following structure:
            - First column: Aspect/Feature being compared
            - Subsequent columns: One column per country ({countries_str})
            - Include key aspects like: Legal Framework, Year Enacted, Key Provisions, Enforcement, etc.
            - Add a brief summary paragraph after the table if needed

            Example format:
            | Aspect | {detected_countries[0]} | {detected_countries[1]} |
            |--------|---------|---------|
            | Legal Framework | ... | ... |
            | Year Enacted | ... | ... |
            """
            print(f"[QUERY_HANDLER] Comparative response formatting enabled for: {detected_countries}")

        # Add source citation instructions if web search was used
        if web_search_used:
            source_citation_instructions = """
            **IMPORTANT - SOURCE ATTRIBUTION:**
            This response includes information from EXTERNAL WEB SOURCES to supplement the knowledge base.

            You MUST explicitly state this in your response:
            - At the END of your response, add a clear notice like:
              "ℹ️ *Note: This response includes current information from external web sources to supplement our knowledge base.*"
            - If citing specific facts from web search, you may add "(from recent web sources)" inline

            This transparency is crucial for the user to understand the source of information.
            """
            formatting_instructions += source_citation_instructions

        # Get prompt with conversation context
        # Prepend conversation history to context if available
        full_context = conversation_context + context_text if conversation_context else context_text

        prompt = self.prompt_manager.get_prompt(
            "rag_response",
            variables={
                "context": full_context,
                "question": query,
                "available_tabs": available_tabs,
                "current_tab": current_tab_display,
                "formatting_instructions": formatting_instructions
            }
        )

        llm = LLMFactory.create_llm(
            use_case="query_response",
            streaming=streaming
        )

        if streaming:
            async for chunk in llm.astream(prompt):
                if hasattr(chunk, 'content'):
                    yield chunk.content
        else:
            # Non-streaming response
            response = await llm.ainvoke(prompt)
            yield response.content

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:

        if not documents:
            return "No relevant documents found."

        # OPTIMIZATION: Limit context size to prevent slow LLM generation
        # Approximate: 1 token ≈ 4 characters, target max 8000 tokens = 32000 chars
        MAX_CONTEXT_CHARS = 32000
        total_chars = 0

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = {k: v for k, v in doc.items() if k != "content"}

            # Format metadata
            meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])

            # Truncate individual document if too long
            if len(content) > 4000:
                content = content[:4000] + "... [truncated]"

            doc_text = f"Document {i} [{meta_str}]:\n{content}\n"

            # Check if adding this document exceeds limit
            if total_chars + len(doc_text) > MAX_CONTEXT_CHARS:
                print(f"[CONTEXT] Truncating at {i} documents to stay within token limit")
                break

            context_parts.append(doc_text)
            total_chars += len(doc_text)

        result = "\n".join(context_parts)
        print(f"[CONTEXT] Total context size: {len(result)} chars (~{len(result)//4} tokens)")
        return result

    def _format_available_tabs(self) -> str:
        """
        Format enabled tabs with descriptions for the RAG prompt.

        Returns:
            Formatted string with tab names and descriptions
        """

        tabs_config = settings.tabs_config
        enabled_tabs = settings.get_enabled_tabs()

        if not enabled_tabs:
            return "General legal and regulatory information"

        tab_descriptions = []
        for tab_name in enabled_tabs:
            config = tabs_config.get(tab_name, {})
            display_name = config.get("display_name", tab_name.replace("-", " ").title())
            description = config.get("description", "")

            if description:
                tab_descriptions.append(f"- {display_name}: {description}")
            else:
                tab_descriptions.append(f"- {display_name}")

        return "\n".join(tab_descriptions)

    async def save_conversation(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        tab: str = None,
        metadata: Dict[str, Any] = None,
        client_id: Optional[str] = None
    ) -> None:
        """
        Save a conversation exchange (user query + assistant response).

        Uses the new ConversationService if enabled, with fallback to MongoDB.

        Args:
            session_id: The session ID
            user_query: The user's query
            assistant_response: The assistant's response
            tab: The current tab/topic
            metadata: Additional metadata
            client_id: Optional client ID for multi-tenant support
        """
        # Try new conversation system first
        if self.use_new_conversation_system:
            try:
                # Ensure session exists (create if needed)
                visibility = Visibility.PRIVATE if client_id else Visibility.PUBLIC
                session, is_new = await self.conversation_service.get_or_create_session(
                    session_id=session_id,
                    client_id=client_id,
                    visibility=visibility
                )

                if is_new:
                    query_logger.debug(f"Created new session {session_id} for conversation")

                # Add message pair
                user_metadata = {**(metadata or {}), "tab": tab}
                assistant_metadata = {**(metadata or {}), "tab": tab}

                await self.conversation_service.add_message_pair(
                    session_id=session_id,
                    user_content=user_query,
                    assistant_content=assistant_response,
                    user_metadata=user_metadata,
                    assistant_metadata=assistant_metadata
                )

                query_logger.debug(f"Saved conversation to new system for session {session_id}")

                # Cleanup old messages periodically (every 10th save)
                import random
                if random.random() < 0.1:
                    await self.conversation_service.cleanup_old_messages(session_id, keep_recent=50)

                return  # Success, don't fall through to legacy

            except Exception as e:
                query_logger.warning(f"Error saving to new conversation system: {e}, falling back to legacy")

        # Fallback to legacy MongoDB
        if not self.db_client:
            return

        try:
            collection = self.db_client.get_collection(self.conversation_collection)

            conversation_doc = {
                "session_id": session_id,
                "user_query": user_query,
                "assistant_response": assistant_response,
                "tab": tab,
                "timestamp": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "client_id": client_id
            }

            collection.insert_one(conversation_doc)
        except Exception as e:
            query_logger.error(f"Error saving conversation to MongoDB: {e}")

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
        client_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Uses the new ConversationService if enabled, with fallback to legacy SQLite.

        Args:
            session_id: Conversation session ID
            limit: Maximum number of messages to retrieve
            client_id: Optional client ID for access control (new system only)

        Returns:
            List of conversation messages
        """
        # Try new conversation system first
        if self.use_new_conversation_system:
            try:
                history = await self.conversation_service.get_conversation_history(
                    session_id=session_id,
                    limit=limit,
                    client_id=client_id
                )
                if history:
                    query_logger.debug(
                        f"Retrieved {len(history)} messages from new conversation system"
                    )
                    return history
            except (SessionNotFoundError, SessionExpiredError) as e:
                # Session doesn't exist in new system, try legacy
                query_logger.debug(f"Session not in new system: {e}, trying legacy")
            except UnauthorizedAccessError:
                query_logger.warning(f"Unauthorized access attempt to session {session_id}")
                return []
            except Exception as e:
                query_logger.warning(f"Error in new conversation system: {e}, falling back to legacy")

        # Fallback to legacy SQLite
        try:
            from app.db.chat_history import database as chat_history_db
            history = chat_history_db.get_session_history(session_id, limit=limit)
            return history
        except Exception as e:
            query_logger.error(f"Error retrieving conversation history: {e}")
            return []

    async def get_context_history(
        self, session_id: str, query: str, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """
        The messages that came before the current question. The chat route stores the question
        before it is answered, so it is left out here: it is not context for itself.
        """
        history = await self.get_conversation_history(session_id, limit=limit + 1)
        if history and history[-1].get("role") in ("human", "user") and history[-1].get("content") == query:
            history = history[:-1]
        return history[-limit:]

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        visibility: str = "public",
        tab: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        streaming: bool = True,
        save_conversation: bool = True,
        use_auto_filters: Optional[bool] = None,
        enable_web_search: Optional[bool] = None,
        search_tool_type: str = "tavily"
    ) -> AsyncGenerator[str, None]:
        # Initialize metrics tracking
        metrics = QueryMetrics()
        query_id = str(uuid.uuid4())[:8]  # Short ID for log correlation
        metrics.start(query_id, session_id or "", tab or "", visibility)

        # Set logging context for this query
        query_logger.set_context(query_id=query_id, session_id=session_id or "none")
        query_logger.info(
            "Query processing started",
            query_preview=query[:50] + "..." if len(query) > 50 else query,
            visibility=visibility,
            tab=tab or "auto-detect"
        )

        # Load conversation history (reduced limit for faster loading)
        conversation_history = []
        with track_stage(metrics, "history_loading"):
            if session_id:
                conversation_history = await self.get_context_history(session_id, query)
                query_logger.debug(
                    "Loaded conversation history",
                    message_count=len(conversation_history)
                )
            else:
                query_logger.debug("No session_id provided, conversation history empty")

        # OPTIMIZATION: Parallel execution of independent operations
        # We can run enhancement, relevance check, and tab detection in parallel

        async def enhance_query_task():
            """Task for query enhancement"""
            if conversation_history and len(conversation_history) > 0:
                # OPTIMIZATION: Skip enhancement if query doesn't have ambiguous references
                if not self._has_ambiguous_references(query):
                    print(f"[QUERY_ENHANCEMENT] Skipping - no ambiguous references detected")
                    return query

                print(f"[QUERY_ENHANCEMENT] Starting enhancement (ambiguous references found)")
                try:
                    enhanced = await self._enhance_query_with_context(query, conversation_history)
                    if enhanced != query:
                        print(f"[QUERY_ENHANCEMENT] ✓ Enhanced: '{query}' → '{enhanced}'")
                    else:
                        print(f"[QUERY_ENHANCEMENT] Query unchanged")
                    return enhanced
                except Exception as e:
                    print(f"[QUERY_ENHANCEMENT] ERROR: {e}")
                    return query
            else:
                print(f"[QUERY_ENHANCEMENT] Skipping - no conversation history")
                return query

        async def check_relevance_task():
            """Task for relevance checking (only for public queries)"""
            if visibility != "private":
                return await self._is_query_relevant_to_domain(query, conversation_history)
            return True  # Private queries skip relevance check

        async def detect_tab_task():
            """Task for tab detection (only if not provided)"""
            if tab is None:
                return await self.detect_tab(query, conversation_history)
            return tab

        # Run tasks in parallel for maximum speed
        with track_stage(metrics, "parallel_llm_ops"):
            enhanced_query, is_relevant, detected_tab = await asyncio.gather(
                enhance_query_task(),
                check_relevance_task(),
                detect_tab_task()
            )

        query_logger.debug(
            "Parallel LLM operations completed",
            query_enhanced=(enhanced_query != query),
            is_relevant=is_relevant,
            detected_tab=detected_tab
        )

        # Handle relevance check result
        if not is_relevant:
            detected_language = self._detect_query_language(query)
            out_of_scope_message = self._get_out_of_scope_message(detected_language)
            query_logger.warning("Query rejected as out of scope", language=detected_language)
            metrics.mark_error("out_of_scope")
            query_logger.info("Query processing completed (rejected)", **metrics.to_dict())
            query_logger.clear_context()
            yield out_of_scope_message
            return

        # Set tab for filtering
        if tab is None:
            tab = detected_tab
            tab_for_filtering = None
        else:
            tab_for_filtering = tab
            print(f"[QUERY_HANDLER] Using explicitly provided tab: {tab}")
        if use_auto_filters is None:
            if visibility == "private" and metadata_filters and metadata_filters.get("client_id"):
                use_auto_filters_resolved = False
                tab_for_filtering = None
                print(f"[QUERY_HANDLER] Private document query with client_id - only filtering by client_id (pure semantic search)")
            else:
                use_auto_filters_resolved = True
        else:
            # Use explicit override
            use_auto_filters_resolved = use_auto_filters
            print(f"[QUERY_HANDLER] Smart filtering explicitly set to: {use_auto_filters_resolved}")

        # OPTIMIZATION: Reduced top_k for faster retrieval (6 instead of 10)
        with track_stage(metrics, "document_retrieval"):
            documents = await self.retrieve_documents(
                enhanced_query,  # Use context-enhanced query
                visibility=visibility,
                tab=tab_for_filtering,  # Only pass tab if explicitly provided
                metadata_filters=metadata_filters,
                use_auto_filters=use_auto_filters_resolved
            )

        metrics.doc_count = len(documents)
        query_logger.info(
            "Document retrieval completed",
            doc_count=len(documents),
            auto_filters=use_auto_filters_resolved
        )

        # 3.5. Web search fallback (if enabled and needed)
        # Determine if web search should be used
        if enable_web_search is None:
            # Disable web search for private documents by default (privacy requirement)
            enable_web_search_resolved = False if visibility == "private" else self.default_enable_web_search
        else:
            enable_web_search_resolved = enable_web_search

        web_search_used = False
        if self._should_use_web_search(query, documents, enable_web_search_resolved):
            # Note: For public documents, query relevance was already verified at the start
            # For private documents, web search is disabled by default
            # So if we get here, the query should be relevant

            with track_stage(metrics, "web_search"):
                web_search_context = await self.perform_web_search(
                    query,
                    search_tool_type=search_tool_type,
                    max_results=3
                )

            # Only use web search results if they're valid (not an error message)
            if web_search_context and "Error" not in web_search_context:
                query_logger.info(
                    "Web search completed successfully",
                    search_tool=search_tool_type
                )
                # Create a pseudo-document for web search results
                web_doc = {
                    "content": f"\n\n=== SUPPLEMENTAL WEB SEARCH RESULTS ===\n{web_search_context}",
                    "metadata": {"source": "web_search", "type": "supplemental"}
                }
                documents.append(web_doc)
                web_search_used = True
                metrics.web_search_used = True
                metrics.doc_count = len(documents)
            else:
                query_logger.warning("Web search failed or returned errors, using only vector store results")

        # 4. Generate response with conversation context
        response_chunks = []
        generation_start = time.time()
        try:
            async for chunk in self.generate_response(
                query,
                documents,
                streaming,
                tab=tab,
                web_search_used=web_search_used,
                conversation_history=conversation_history
            ):
                response_chunks.append(chunk)
                yield chunk

            generation_time_ms = (time.time() - generation_start) * 1000
            metrics.record_timing("response_generation", generation_time_ms)

        except Exception as e:
            query_logger.error(
                "Response generation failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            metrics.mark_error(f"generation_failed: {type(e).__name__}")
            raise

        # 5. Save conversation
        if save_conversation and session_id:
            with track_stage(metrics, "save_conversation"):
                full_response = "".join(response_chunks)
                # Extract client_id from metadata_filters for multi-tenant support
                client_id = metadata_filters.get("client_id") if metadata_filters else None
                await self.save_conversation(
                    session_id=session_id,
                    user_query=query,
                    assistant_response=full_response,
                    tab=tab,
                    metadata={
                        "query_id": query_id,
                        "visibility": visibility,
                        "num_documents": len(documents),
                        "filters_used": metadata_filters,
                        "web_search_used": web_search_used
                    },
                    client_id=client_id
                )

        # Log final metrics
        metrics.tab = tab or detected_tab
        query_logger.info(
            "Query processing completed successfully",
            **metrics.to_dict()
        )
        query_logger.clear_context()

    async def process_query_non_streaming(
        self,
        query: str,
        session_id: Optional[str] = None,
        visibility: str = "public",
        tab: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_auto_filters: Optional[bool] = None,
        enable_web_search: Optional[bool] = None,
        search_tool_type: str = "tavily"
    ) -> Dict[str, Any]:
        response_parts = []
        async for chunk in self.process_query(
            query=query,
            session_id=session_id,
            visibility=visibility,
            tab=tab,
            metadata_filters=metadata_filters,
            streaming=False,
            use_auto_filters=use_auto_filters,
            enable_web_search=enable_web_search,
            search_tool_type=search_tool_type
        ):
            response_parts.append(chunk)

        return {
            "response": "".join(response_parts),
            "tab": tab,
            "visibility": visibility
        }


# Global instance
_query_handler: Optional[DynamicQueryHandler] = None


def get_query_handler(
    db_client=None,
    use_container: bool = False,
    use_new_conversation_system: bool = True
) -> DynamicQueryHandler:
    """
    Get or create the query handler singleton.

    Args:
        db_client: Optional MongoDB client (for legacy storage)
        use_container: If True, use the DI container to create the handler
        use_new_conversation_system: Whether to use the new conversation management

    Returns:
        The query handler instance
    """
    global _query_handler

    if _query_handler is None:
        if use_container:
            # Use dependency injection container
            from app.core.container import get_container
            container = get_container()
            _query_handler = container.query_handler()
        else:
            # Direct instantiation (legacy behavior)
            if db_client is None:
                from app.db.base_db import mongo_client
                db_client = mongo_client

            _query_handler = DynamicQueryHandler(
                db_client=db_client,
                use_new_conversation_system=use_new_conversation_system
            )

    return _query_handler


def reset_query_handler() -> None:
    """Reset the query handler singleton. Useful for testing."""
    global _query_handler
    _query_handler = None
    query_logger.info("Query handler reset")
