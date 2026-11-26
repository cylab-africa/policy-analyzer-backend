import copy
import json
import threading
import time
from collections import OrderedDict
from typing import List, Dict, Any, Optional
from pathlib import Path
from difflib import SequenceMatcher

from app.core.config import settings, PROJECT_ROOT
from app.services.llm_factory import LLMFactory


class EntityDetector:

    def __init__(self, llm=None):
        self.llm = llm
        self._african_countries = None
        self._country_names_proper_case = None  # Store original casing from data
        self._country_cache: Dict[str, List[str]] = {}
        # detect_all_entities results for recent questions. One question is analysed by the
        # retrieval step, the filter builder and the answer step, and with use_llm each analysis
        # is a model call; the cache turns the second and third into lookups.
        self._entities_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._entities_cache_size = 256
        # Short-lived: it only needs to cover one request, and a failed model call (which returns
        # empty defaults) should not stick to a question for long
        self._entities_cache_seconds = 300.0
        self._entities_lock = threading.Lock()

        # Country aliases mapping (common names → official names in data file)
        self._country_aliases = {
            # English common names
            "ivory coast": "cote d'ivoire",
            "côte d'ivoire": "cote d'ivoire",
            "cape verde": "cabo verde",
            "swaziland": "eswatini",

            # Abbreviations
            "drc": "democratic republic of the congo",
            "dr congo": "democratic republic of the congo",
            "congo-kinshasa": "democratic republic of the congo",
            "congo-brazzaville": "republic of congo",

            # Alternative spellings
            "sao tome and principe": "sao tome & principe",
            "são tomé and príncipe": "sao tome & principe",

            # Historical names
            "zaire": "democratic republic of the congo",

            # French names (Francophone African countries)
            "cameroun": "cameroon",
            "afrique du sud": "south africa",
            "égypte": "egypt",
            "maroc": "morocco",
            "tunisie": "tunisia",
            "algérie": "algeria",
            "sénégal": "senegal",
            "côte d'ivoire": "cote d'ivoire",
            "mali": "mali",
            "niger": "niger",
            "tchad": "chad",
            "burkina faso": "burkina faso",
            "bénin": "benin",
            "togo": "togo",
            "guinée": "guinea",
            "guinée équatoriale": "equatorial guinea",
            "guinée-bissau": "guinea-bissau",
            "madagascar": "madagascar",
            "république démocratique du congo": "democratic republic of the congo",
            "rdc": "democratic republic of the congo",
            "république du congo": "republic of congo",
            "rwanda": "rwanda",
            "burundi": "burundi",
            "djibouti": "djibouti",
            "comores": "comoros",
            "gabon": "gabon",
            "centrafrique": "central african republic",
            "république centrafricaine": "central african republic",
            "maurice": "mauritius",
            "seychelles": "seychelles",
        }

    def _load_african_countries(self) -> Dict[str, str]:
        if self._african_countries is not None:
            return self._african_countries

        try:
            data_path = PROJECT_ROOT / "app" / "data" / "countries_law_data.json"
            if not data_path.exists():
                print(f"Warning: countries_law_data.json not found at {data_path}")
                return {}

            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract country names and ISO codes
            countries = {}
            proper_case_names = {}  # Map lowercase → proper case
            for entry in data:
                country = entry.get("country")
                iso_code = entry.get("isoAlpha2")
                if country and iso_code:
                    countries[country.lower()] = iso_code
                    proper_case_names[country.lower()] = country  # Keep original casing

            self._african_countries = countries
            self._country_names_proper_case = proper_case_names
            return countries

        except Exception as e:
            print(f"Error loading African countries data: {e}")
            return {}

    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.7) -> bool:
        ratio = SequenceMatcher(None, query.lower(), target.lower()).ratio()
        return ratio >= threshold

    def _resolve_country_alias(self, country_name: str) -> Optional[str]:
        """
        Resolve a country name or alias to the official name in our data.

        Args:
            country_name: Country name (could be alias like "Ivory Coast")

        Returns:
            Official country name with proper casing if found, None otherwise
        """
        african_countries = self._load_african_countries()
        country_lower = country_name.lower()

        # Check if it's already an official name
        if country_lower in african_countries:
            # Return with proper casing from data
            return self._country_names_proper_case.get(country_lower, country_name)

        # Check if it's an alias
        if country_lower in self._country_aliases:
            official_name_lower = self._country_aliases[country_lower]
            if official_name_lower in african_countries:
                # Return with proper casing from data
                return self._country_names_proper_case.get(official_name_lower, official_name_lower.title())

        return None

    def _detect_countries_with_llm(self, query: str) -> List[str]:
        """Use LLM to detect countries mentioned in the query with high accuracy."""
        from app.services.llm_factory import LLMFactory

        african_countries = self._load_african_countries()
        country_list = ", ".join(sorted([c.title() for c in african_countries.keys()]))

        alias_examples = [
            "Ivory Coast (also known as Cote d'Ivoire)",
            "Cape Verde (also known as Cabo Verde)",
            "Swaziland (now called Eswatini)",
            "DRC (Democratic Republic of the Congo)",
            "Cameroun (French for Cameroon)",
            "Afrique du Sud (French for South Africa)"
            "Ruanda (French for Rwanda)",
            "République du Congo (French for Republic of Congo)",
            "Égypte (French for Egypt)"
        ]
        alias_note = "Note: You can use common names, abbreviations, or French names like " + ", ".join(alias_examples[:3])

        # Create a prompt for precise country detection
        prompt = f"""You are a precise country name extractor. Extract ONLY the African country names explicitly mentioned in the user's query.
            Rules:
            1. Return ONLY countries that are directly mentioned in the query (in English, French, or common aliases)
            2. Do NOT infer or guess countries based on context
            3. You can recognize: English names, French names (e.g., "Cameroun", "Afrique du Sud"), common names (e.g., "Ivory Coast"), and abbreviations (e.g., "DRC")
            4. If no African country is mentioned, return "NONE"
            5. Return multiple countries separated by commas if multiple are mentioned

            {alias_note}

            Available African countries: {country_list}

            User query: "{query}"

            Detected countries (or NONE):"""

        try:
            llm = LLMFactory.create_llm(
                model_provider="openai",
                model_name=settings.openai_model,
                temperature=0.0,
                max_tokens=50,
                streaming=False
            )

            response = llm.invoke(prompt)
            result = response.content.strip()

            print(f"[LLM Country Detection] Query: '{query}' → Result: '{result}'")

            if result.upper() == "NONE" or not result:
                return []

            # Parse the response and resolve aliases
            detected = []
            for country in result.split(','):
                country = country.strip()

                # Try to resolve alias or official name
                resolved = self._resolve_country_alias(country)
                if resolved:
                    detected.append(resolved)
                else:
                    print(f"[LLM Country Detection] Warning: Could not resolve '{country}'")

            return detected

        except Exception as e:
            print(f"Error in LLM country detection: {e}")
            # Fallback to fuzzy matching
            return self._detect_countries_fuzzy(query, 0.85)

    def _detect_countries_fuzzy(self, query: str, threshold: float = 0.85) -> List[str]:
        """Fuzzy matching fallback for country detection."""
        african_countries = self._load_african_countries()
        if not african_countries:
            return []

        detected = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # First pass: Look for exact matches only
        for country_name, iso_code in african_countries.items():
            # Exact match (whole country name appears in query)
            if country_name in query_lower:
                detected.append(country_name.title())
                continue

            # Check if ISO code appears
            if iso_code and iso_code.lower() in query_lower:
                detected.append(country_name.title())
                continue

            # Exact word match (country name is a single word in query)
            if country_name in query_words:
                detected.append(country_name.title())

        # If we found exact matches, return those
        if detected:
            detected = list(dict.fromkeys(detected))
            return detected

        # Second pass: Fuzzy matching
        for country_name, iso_code in african_countries.items():
            country_words = country_name.split()
            for word in query_words:
                if len(word) >= 5:
                    for country_word in country_words:
                        if len(country_word) >= 4 and self._fuzzy_match(word, country_word, threshold):
                            detected.append(country_name.title())
                            break
                    if country_name.title() in detected:
                        break

        detected = list(dict.fromkeys(detected))
        return detected

    def detect_countries(
        self,
        query: str,
        threshold: float = 0.85,
        use_cache: bool = True,
        use_llm: bool = True  # New parameter to enable LLM detection
    ) -> List[str]:
        cache_key = f"llm:{query.lower()}" if use_llm else f"fuzzy:{query.lower()}:{threshold}"
        if use_cache and cache_key in self._country_cache:
            return self._country_cache[cache_key]

        # Use LLM for country detection (more accurate)
        if use_llm:
            detected = self._detect_countries_with_llm(query)
        else:
            # Fallback to fuzzy matching
            detected = self._detect_countries_fuzzy(query, threshold)

        # Cache result
        if use_cache:
            self._country_cache[cache_key] = detected

        return detected

    def detect_document_types(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        use_llm: bool = True
    ) -> List[str]:
        # Get document types from config if not provided
        if document_types is None:
            metadata_config = settings.metadata_config
            doc_type_config = metadata_config.get("detection_config", {}).get("document_type", {})
            document_types = doc_type_config.get("options", [])

        if not document_types:
            return []

        detected = []

        # 1. Keyword-based detection (fast path)
        keyword_mapping = {
            "cross_border": ["cross_border_policy", "cross-border", "transborder", "international transfer"],
            "digital_id": ["digital_id_policy", "digital identity", "digital id", "e-id", "electronic identity"],
            "data_protection": ["data_protection_policy", "data protection", "privacy", "gdpr", "personal data"],
            "privacy": ["privacy_policy", "privacy", "personal information"],
            "regulatory": ["regulatory_framework", "regulation", "compliance", "legal framework"],
            "compliance": ["compliance_guideline", "compliance", "guideline", "standard"]
        }

        query_lower = query.lower()
        for doc_type in document_types:
            # Direct match
            if doc_type.replace("_", " ") in query_lower:
                detected.append(doc_type)
                continue

            # Keyword match
            for key, keywords in keyword_mapping.items():
                if key in doc_type:
                    for keyword in keywords:
                        if keyword.lower() in query_lower:
                            detected.append(doc_type)
                            break
                    if doc_type in detected:
                        break

        # 2. LLM-based classification (if enabled and LLM available)
        if use_llm and self.llm and not detected:
            detected = self._llm_classify_document_type(query, document_types)

        # Remove duplicates
        return list(dict.fromkeys(detected))

    def _llm_classify_document_type(
        self,
        query: str,
        document_types: List[str]
    ) -> List[str]:
        try:
            from app.services.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()

            prompt = prompt_manager.get_prompt(
                "document_type_classifier",
                variables={
                    "text": query[:500], 
                    "document_types": ", ".join(document_types)
                }
            )

            response = self.llm.invoke(prompt)
            result = response.content.strip()
            if result in document_types:
                return [result]
            elif "," in result:
                types = [t.strip() for t in result.split(",")]
                return [t for t in types if t in document_types]

        except Exception as e:
            print(f"Warning: LLM document type classification failed: {e}")

        return []

    def detect_iso_codes(self, query: str) -> List[str]:
        african_countries = self._load_african_countries()
        detected = []

        query_upper = query.upper()
        words = query_upper.split()

        for iso_code in african_countries.values():
            if iso_code in words:
                detected.append(iso_code)

        return list(dict.fromkeys(detected))

    def extract_metadata_with_llm(self, query: str, tab: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to extract ALL relevant metadata from the query in one call.
        More efficient than multiple separate calls.
        """
        from app.services.llm_factory import LLMFactory

        african_countries = self._load_african_countries()
        country_list = ", ".join(sorted([c.title() for c in african_countries.keys()]))

        prompt = f"""Extract metadata from the user's query. Return ONLY valid JSON.

        Query: "{query}"
        Tab Context: {tab or 'general'}

        Extract:
        1. "countries": List of African country names explicitly mentioned. Recognize English names, French names (e.g., "Cameroun" → Cameroon, "Afrique du Sud" → South Africa), common names ("Ivory Coast"), and abbreviations ("DRC")
        2. "regions": Regions mentioned like "East Africa", "West Africa", "Southern Africa", "Africa" (or empty list)
        3. "is_comparative": true if query compares multiple countries, false otherwise
        4. "is_broad": true if query is about general/continental topics, false if country-specific
        5. "is_quantitative": true if query asks for counts/statistics (e.g., "how many countries", "what percentage", "count of"), false otherwise

        Note: The query may be in English or French. Detect country names in either language.

        Available African countries: {country_list}

        Return JSON in this exact format:
        {{
        "countries": ["Country1", "Country2"],
        "regions": ["Region1"],
        "is_comparative": false,
        "is_broad": false,
        "is_quantitative": false
        }}

        JSON:"""

        try:
            llm = LLMFactory.create_llm(
                model_provider="openai",
                model_name=settings.openai_model,
                temperature=0.0,
                max_tokens=150,
                streaming=False
            )

            response = llm.invoke(prompt)
            result = response.content.strip()

            # Clean up the response
            if result.startswith("```json"):
                result = result[7:]
            elif result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            import json
            metadata = json.loads(result)

            # Validate and resolve country aliases
            validated_countries = []
            for country in metadata.get("countries", []):
                resolved = self._resolve_country_alias(country)
                if resolved:
                    validated_countries.append(resolved)
                else:
                    print(f"[LLM Metadata Extraction] Warning: Could not resolve country '{country}'")

            metadata["countries"] = validated_countries

            print(f"[LLM Metadata Extraction] Query: '{query}' → {metadata}")

            return metadata

        except Exception as e:
            print(f"Error in LLM metadata extraction: {e}")
            return {
                "countries": [],
                "regions": [],
                "is_comparative": False,
                "is_broad": False,
                "is_quantitative": False
            }

    def detect_all_entities(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        if config is None:
            config = settings.metadata_config.get("detection_config", {})

        key = json.dumps([query, config], sort_keys=True, default=str)
        with self._entities_lock:
            entry = self._entities_cache.get(key)
            if entry is not None and time.monotonic() - entry[0] < self._entities_cache_seconds:
                self._entities_cache.move_to_end(key)
                return copy.deepcopy(entry[1])  # callers may change what they get

        results = self._detect_all_entities_uncached(query, config)

        with self._entities_lock:
            self._entities_cache[key] = (time.monotonic(), copy.deepcopy(results))
            self._entities_cache.move_to_end(key)
            while len(self._entities_cache) > self._entities_cache_size:
                self._entities_cache.popitem(last=False)
        return results

    def _detect_all_entities_uncached(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        results = {}

        # Use comprehensive LLM extraction if enabled
        use_llm = config.get("country", {}).get("use_llm", True)

        if use_llm:
            # Extract all metadata at once with LLM (more efficient)
            llm_metadata = self.extract_metadata_with_llm(query)

            # Detect countries
            if config.get("country", {}).get("enabled", True):
                results["country"] = llm_metadata.get("countries", [])

            # Store additional context
            results["regions"] = llm_metadata.get("regions", [])
            results["is_comparative"] = llm_metadata.get("is_comparative", False)
            results["is_broad"] = llm_metadata.get("is_broad", False)
            results["is_quantitative"] = llm_metadata.get("is_quantitative", False)
        else:
            # Fallback to fuzzy matching
            if config.get("country", {}).get("enabled", True):
                threshold = config.get("country", {}).get("confidence_threshold", 0.85)
                results["country"] = self.detect_countries(query, threshold=threshold, use_llm=False)

            # Set defaults for query analysis fields
            results["is_quantitative"] = False

        # Detect ISO codes (derived from countries)
        if config.get("isoAlpha2", {}).get("enabled", True):
            results["isoAlpha2"] = self.detect_iso_codes(query)

        return results

    def validate_entities(
        self,
        entities: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        if config is None:
            config = settings.metadata_config.get("detection_config", {})

        validated = {}

        for entity_type, values in entities.items():
            type_config = config.get(entity_type, {})
            if not type_config.get("enabled", True):
                continue

            # Validate values based on configuration
            if entity_type == "document_type":
                valid_options = type_config.get("options", [])
                validated[entity_type] = [v for v in values if v in valid_options]
            else:
                validated[entity_type] = values

        return validated


_entity_detector: Optional[EntityDetector] = None


def get_entity_detector(llm=None) -> EntityDetector:

    global _entity_detector

    if _entity_detector is None:
        # Create LLM if not provided
        if llm is None:
            try:
                llm = LLMFactory.create_llm(use_case="metadata_filtering")
            except Exception as e:
                print(f"Warning: Could not create LLM for entity detection: {e}")
                llm = None

        _entity_detector = EntityDetector(llm=llm)

    return _entity_detector
