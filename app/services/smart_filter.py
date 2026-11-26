from typing import Dict, Any, List, Optional
from app.services.entity_detector import get_entity_detector
from app.core.config import settings


class SmartFilterService:
    def __init__(self):
        self.entity_detector = get_entity_detector()
        self.metadata_config = settings.metadata_config
        self.tabs_config = settings.tabs_config
        self.filter_strategy = self.metadata_config.get("filter_strategy", {})
        self.auto_detect_enabled = self.metadata_config.get("auto_detect", True)
        query_indicators = self.filter_strategy.get("query_indicators", {})

        self.broad_query_indicators = query_indicators.get("broad", [
            "africa", "african", "continent", "region", "regional",
            "compare", "comparison", "across", "different", "various",
            "all countries", "multiple countries", "several countries",
            "overview", "landscape", "general", "common", "typical",
            "all", "every", "each", "most"
        ])

        self.specific_query_indicators = query_indicators.get("specific", [
            "specific", "particular", "in", "at", "for",
            "what is", "how does", "describe", "explain",
            "tell me about", "information about"
        ])

    def get_applicable_metadata_filters(self, tab: Optional[str] = None) -> List[str]:
        """
        Get the list of metadata filters applicable for a given tab.

        Args:
            tab: Tab name (e.g., "digital-id", "data-protection"). If None, uses common filters.

        Returns:
            List of metadata filter keys that should be considered for this tab
        """
        if tab and tab in self.tabs_config:
            # Get metadata filters defined for this specific tab
            tab_filters = self.tabs_config[tab].get("metadata_filters", [])
            return tab_filters

        # Fallback: return all important metadata keys from config
        return self.metadata_config.get("important_keys", ["country", "document_type"])

    def extract_smart_filters(self, query: str, tab: Optional[str] = None, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata filters intelligently based on query intent.
        Fully dynamic - only applies filters that are configured for the given tab.

        Args:
            query: User query string
            tab: Current tab (e.g., "digital-id", "data-protection")
            client_id: Optional client ID for multi-tenant filtering

        Returns:
            Dictionary containing:
            - filters: Dict of metadata filters to apply
            - detected_entities: All entities detected in the query
            - filter_reasoning: Explanation of filtering decisions
            - applicable_filters: List of filter keys applicable for this tab
        """
        result = {
            "filters": {},
            "detected_entities": {},
            "filter_reasoning": [],
            "applicable_filters": []
        }

        # Get the list of applicable filters for this tab
        applicable_filters = self.get_applicable_metadata_filters(tab)
        result["applicable_filters"] = applicable_filters

        if tab:
            result["filter_reasoning"].append(f"Tab: '{tab}' - Applicable filters: {applicable_filters}")
        else:
            result["filter_reasoning"].append(f"No tab specified - Using default filters: {applicable_filters}")
        if client_id:
            result["filters"]["client_id"] = client_id
            result["filter_reasoning"].append(f"Applied client_id filter for multi-tenant isolation")
        if not self.auto_detect_enabled:
            result["filter_reasoning"].append("Metadata auto-detection is disabled in config")
            return result
        detection_config = self.metadata_config.get("detection_config", {})
        detected_entities = self.entity_detector.detect_all_entities(query, detection_config)
        result["detected_entities"] = detected_entities
        is_comparative = detected_entities.get("is_comparative", False)
        is_quantitative = detected_entities.get("is_quantitative", False)
        is_broad_llm = detected_entities.get("is_broad", False)
        regions = detected_entities.get("regions", [])
        is_broad_query = is_broad_llm or self._is_broad_query(query)
        has_multiple_countries = len(detected_entities.get("country", [])) > 1

        if is_quantitative:
            result["filter_reasoning"].append(f"Query type: QUANTITATIVE (counting/statistics)")
        elif is_comparative:
            result["filter_reasoning"].append(f"Query type: COMPARATIVE (comparing multiple entities)")
        elif is_broad_query:
            result["filter_reasoning"].append(f"Query type: BROAD/EXPLORATORY (LLM detected: {is_broad_llm})")
        else:
            result["filter_reasoning"].append(f"Query type: SPECIFIC/FOCUSED")

        if regions:
            result["filter_reasoning"].append(f"Regions mentioned: {regions}")

        if has_multiple_countries:
            result["filter_reasoning"].append(f"Multiple countries detected: {detected_entities.get('country', [])}")
        self._apply_dynamic_filters(result, detected_entities, applicable_filters, is_broad_query, query, is_comparative)

        if not result["filters"] or result["filters"] == {"client_id": client_id}:
            if client_id:
                pass
            else:
                result["filters"] = None

        return result

    def _apply_dynamic_filters(
        self,
        result: Dict[str, Any],
        detected_entities: Dict[str, List[str]],
        applicable_filters: List[str],
        is_broad_query: bool,
        query: str,
        is_comparative: bool = False
    ):

        if "country" in applicable_filters:
            countries = detected_entities.get("country", [])
            if countries:
                if is_comparative:
                    result["filter_reasoning"].append(f"✗ Skipped country filter (comparative query detected by LLM)")
                elif len(countries) == 1 and not is_broad_query:
                    result["filters"]["country"] = countries[0]
                    result["filter_reasoning"].append(f"✓ Applied country filter: '{countries[0]}' (single country, specific query)")
                elif len(countries) > 1:
                    result["filter_reasoning"].append(f"✗ Skipped country filter (multiple countries: {len(countries)})")
                else:
                    if self._should_apply_single_country_filter_for_broad_query(query):
                        result["filters"]["country"] = countries[0]
                        result["filter_reasoning"].append(f"✓ Applied country filter: '{countries[0]}' (broad but country-focused)")
                    else:
                        result["filter_reasoning"].append(f"✗ Skipped country filter (broad exploratory query)")
            else:
                result["filter_reasoning"].append("No countries detected in query")
        if "isoAlpha2" in applicable_filters:
            iso_codes = detected_entities.get("isoAlpha2", [])
            if "country" in result["filters"] and iso_codes:
                result["filters"]["isoAlpha2"] = iso_codes[0]
                result["filter_reasoning"].append(f"✓ Applied isoAlpha2 filter: '{iso_codes[0]}' (derived from country)")
        excluded_keys = ["country", "isoAlpha2", "tab", "visibility", "client_id",
                         "is_comparative", "is_broad", "regions", "is_quantitative"]

        for filter_key in applicable_filters:
            if filter_key not in excluded_keys:
                if filter_key in detected_entities:
                    values = detected_entities[filter_key]
                    if values and len(values) == 1:
                        result["filters"][filter_key] = values[0]
                        result["filter_reasoning"].append(f"✓ Applied custom filter '{filter_key}': '{values[0]}'")

    def _is_broad_query(self, query: str) -> bool:
        query_lower = query.lower()

        broad_count = sum(1 for indicator in self.broad_query_indicators if indicator in query_lower)

        specific_count = sum(1 for indicator in self.specific_query_indicators if indicator in query_lower)

        return broad_count > specific_count

    def _get_broad_indicators(self, query: str) -> List[str]:
        """Get list of broad indicators found in query."""
        query_lower = query.lower()
        return [ind for ind in self.broad_query_indicators if ind in query_lower]

    def _should_apply_single_country_filter_for_broad_query(self, query: str) -> bool:
        query_lower = query.lower()

        possessive_patterns = ["'s", "'s", "of [country]"]
        for pattern in possessive_patterns:
            if pattern in query_lower or "'s" in query_lower:
                return True
        location_patterns = [" in ", " for ", " at "]
        if any(pattern in query_lower for pattern in location_patterns):
            return True
        return False

_smart_filter_service: Optional[SmartFilterService] = None


def get_smart_filter_service() -> SmartFilterService:
    """Get singleton instance of SmartFilterService."""
    global _smart_filter_service

    if _smart_filter_service is None:
        _smart_filter_service = SmartFilterService()

    return _smart_filter_service
