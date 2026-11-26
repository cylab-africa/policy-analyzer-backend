from typing import List, Dict, Any, Optional
from itertools import product

from app.core.config import settings
from app.services.entity_detector import EntityDetector, get_entity_detector


class MetadataFilterBuilder:
    def __init__(self, entity_detector: Optional[EntityDetector] = None, config: Optional[Dict[str, Any]] = None):
        self.entity_detector = entity_detector or get_entity_detector()
        self.config = config or settings.metadata_config

    def extract_filters(
        self,
        query: str,
        tab: Optional[str] = None,
        explicit_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
       
        filters = {}
        auto_detect = self.config.get("auto_detect", True)
        if auto_detect:
            detection_config = self.config.get("detection_config", {})
            detected = self.entity_detector.detect_all_entities(query, detection_config)
            invalid_keys = {"is_comparative", "is_broad", "regions", "is_quantitative"}

            for key, values in detected.items():
                if key not in invalid_keys and values:  
                    filters[key] = values
        if tab:
            filters["tab"] = tab
        if explicit_filters:
            filters.update(explicit_filters)

        # 4. Validate filters
        important_keys = self.config.get("important_keys", [])
        validated_filters = {}
        for key in important_keys:
            if key in filters:
                validated_filters[key] = filters[key]

        # Include any additional filters not in important_keys
        for key, value in filters.items():
            if key not in validated_filters:
                validated_filters[key] = value

        return validated_filters

    def build_filter_combinations(
        self,
        filters: Dict[str, Any],
        combine_with_or: Optional[bool] = None
    ) -> List[Dict[str, Any]]:

        if combine_with_or is None:
            filter_strategy = self.config.get("filter_strategy", {})
            combine_with_or = filter_strategy.get("combine_with_or", False)

        if not filters:
            filter_strategy = self.config.get("filter_strategy", {})
            allow_empty = filter_strategy.get("allow_empty_filters", True)
            return [{}] if allow_empty else []

        # Separate scalar and list values
        scalar_filters = {}
        list_filters = {}

        for key, value in filters.items():
            if isinstance(value, list):
                list_filters[key] = value
            else:
                scalar_filters[key] = value

        if not list_filters:
            # No lists, return single filter
            return [filters]

        # Generate all combinations of list filters
        keys = list(list_filters.keys())
        values = [list_filters[k] for k in keys]

        combinations = []
        for combination in product(*values):
            filter_dict = scalar_filters.copy()
            for i, key in enumerate(keys):
                filter_dict[key] = combination[i]
            combinations.append(filter_dict)

        return combinations if combinations else [{}]

    def build_filter_for_vectorstore(
        self,
        query: str,
        tab: Optional[str] = None,
        explicit_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        filters = self.extract_filters(query, tab, explicit_filters)
        combinations = self.build_filter_combinations(filters)

        return combinations

    def filter_by_tab(
        self,
        base_filters: List[Dict[str, Any]],
        tab: str
    ) -> List[Dict[str, Any]]:
        updated = []
        for filter_dict in base_filters:
            new_filter = filter_dict.copy()
            new_filter["tab"] = tab
            updated.append(new_filter)

        return updated if updated else [{"tab": tab}]

    def filter_by_visibility(
        self,
        base_filters: List[Dict[str, Any]],
        visibility: str
    ) -> List[Dict[str, Any]]:
        updated = []
        for filter_dict in base_filters:
            new_filter = filter_dict.copy()
            new_filter["visibility"] = visibility
            updated.append(new_filter)

        return updated if updated else [{"visibility": visibility}]

    def optimize_filters(
        self,
        filters: List[Dict[str, Any]],
        max_combinations: int = 10
    ) -> List[Dict[str, Any]]:
        if len(filters) <= max_combinations:
            return filters

        # Strategy 1: Remove duplicate filters
        unique_filters = []
        seen = set()

        for f in filters:
            # Create a hashable representation
            filter_tuple = tuple(sorted(f.items()))
            if filter_tuple not in seen:
                seen.add(filter_tuple)
                unique_filters.append(f)

        if len(unique_filters) <= max_combinations:
            return unique_filters

        # Strategy 2: Prioritize filters with more constraints
        scored = [(len(f), f) for f in unique_filters]
        scored.sort(reverse=True)

        return [f for _, f in scored[:max_combinations]]

    def get_important_filters(
        self,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:

        important_keys = self.config.get("important_keys", [])
        return {k: v for k, v in filters.items() if k in important_keys}

    def explain_filters(
        self,
        filters: List[Dict[str, Any]]
    ) -> str:
        if not filters:
            return "No filters applied (searching all documents)"

        if len(filters) == 1:
            f = filters[0]
            parts = []
            for key, value in f.items():
                parts.append(f"{key}={value}")
            return f"Filtering by: {', '.join(parts)}"

        # Multiple filters
        common = {}
        varying = {}

        # Find common and varying keys
        first = filters[0]
        for key in first:
            values = [f.get(key) for f in filters]
            if all(v == first[key] for v in values):
                common[key] = first[key]
            else:
                varying[key] = list(set(values))

        explanation_parts = []

        if common:
            common_str = ", ".join([f"{k}={v}" for k, v in common.items()])
            explanation_parts.append(f"Common filters: {common_str}")

        if varying:
            for key, values in varying.items():
                values_str = " OR ".join([str(v) for v in values if v is not None])
                explanation_parts.append(f"{key}: {values_str}")

        return "; ".join(explanation_parts)


# Global instance
_filter_builder: Optional[MetadataFilterBuilder] = None


def get_filter_builder(entity_detector: Optional[EntityDetector] = None) -> MetadataFilterBuilder:

    global _filter_builder

    if _filter_builder is None:
        _filter_builder = MetadataFilterBuilder(entity_detector=entity_detector)

    return _filter_builder
