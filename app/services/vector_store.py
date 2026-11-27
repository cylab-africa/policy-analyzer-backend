from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from app.core.config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_qdrant import QdrantVectorStore as QDStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangchainVectorStore
import numpy as np

from app.services.llm_factory import LLMFactory
from app.services.embedding_cache import with_query_cache
from app.services.query_cache import get_query_cache_or_none



# Query-level fields that the entity detector returns but no document has
_QUERY_ONLY_KEYS = {"is_comparative", "is_broad", "regions", "is_quantitative"}


def _sanitize_filter(filter: Dict[str, Any]) -> Dict[str, Any]:
    """
    {field: value} conditions for a document filter: query-level fields are dropped, and a list
    keeps its first value (filter combinations are expanded before they get here).
    """
    conditions = {}
    for key, value in (filter or {}).items():
        if key in _QUERY_ONLY_KEYS:
            continue
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        conditions[key] = value
    return conditions


def _tenant_scope(filter_combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The client_id the searches were limited to, kept by every fallback (empty if none)."""
    for f in filter_combinations or []:
        if f.get("client_id"):
            return {"client_id": f["client_id"]}
    return {}


def _cache_filters(store_name: str, filter: Dict[str, Any]) -> Dict[str, Any]:
    """
    The filters to file a search under in the query cache. The public and private stores share
    one cache, so the store is part of the key: otherwise a private search and a public one with
    the same filters would be served each other's results.
    """
    return {**(filter or {}), "__store__": store_name}


class VectorStore:
    @staticmethod
    def get_mongo_vector_store(connection_string: str, database_name: str, collection_name: str, embedding_function: Any, store_index: str) -> MongoDBAtlasVectorSearch:
        """
        Create a MongoDB client instance.
        """
        client = MongoClient(connection_string)
        db = client[database_name]
        collection = db[collection_name]
        
        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_function,
            index_name=store_index,
        )
    
class VectorStoreV2(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, query: str, filter: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_documents(self, filter: Dict[str, Any]):
        pass

    @abstractmethod
    def update_document(self, document_id: str, updates: Dict[str, Any]):
        pass

    @abstractmethod
    def clear_store(self):
        pass

    @abstractmethod
    def get_vector_store(self)->LangchainVectorStore:
        pass

    def search_with_metadata(
        self,
        query: str,
        tab: str = None,
        metadata_filters: Dict[str, Any] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering (tab-aware).

        Args:
            query: Search query
            tab: Optional tab filter
            metadata_filters: Additional metadata filters
            top_k: Number of results (uses config default if None)

        Returns:
            List of matching documents with metadata
        """
        # Build combined filter
        combined_filter = {}
        if tab:
            combined_filter["tab"] = tab
        if metadata_filters:
            combined_filter.update(metadata_filters)

        # Get top_k from config if not specified
        if top_k is None:
            retrieval_config = settings.retrieval_config
            top_k = retrieval_config.get("default_k", 3)

        return self.search(query, combined_filter, top_k)

    def search_with_filter_combinations(
        self,
        query: str,
        filter_combinations: List[Dict[str, Any]],
        top_k: int = None,
        max_total_docs: int = None,
        fallback_to_unfiltered: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with multiple filter combinations and aggregate results.
        Falls back to unfiltered search if no results found.

        Args:
            query: Search query
            filter_combinations: List of filter dictionaries
            top_k: Results per filter combination
            max_total_docs: Maximum total documents to return
            fallback_to_unfiltered: If True, retry without filters when no results found

        Returns:
            Deduplicated and aggregated results
        """
        if top_k is None:
            retrieval_config = settings.retrieval_config
            top_k = retrieval_config.get("default_k", 3)

        if max_total_docs is None:
            retrieval_config = settings.retrieval_config
            max_total_docs = retrieval_config.get("max_total_docs", 12)

        all_results = []
        seen_content = set()

        for filter_dict in filter_combinations:
            results = self.search(query, filter_dict, top_k)

            for doc in results:
                # Deduplicate by content (first 100 chars)
                content_key = doc.get("content", "")[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_results.append(doc)

        # Apply max_total_docs limit after collecting from all filter combinations
        if len(all_results) > max_total_docs:
            all_results = all_results[:max_total_docs]

        # Fallback: If no results found with filters, try without filters
        if not all_results and fallback_to_unfiltered and filter_combinations:
            # Check if filters had country/specific constraints
            has_country_filter = any('country' in f for f in filter_combinations)

            if has_country_filter:
                print(f"\n[FALLBACK] No results with country filters. Retrying with tab-only filter for broader search...")

                # Retry with only tab filter (remove country filters)
                tab_only_filters = []
                for f in filter_combinations:
                    tab_filter = {k: v for k, v in f.items() if k in ['tab', 'visibility', 'client_id']}
                    if tab_filter and tab_filter not in tab_only_filters:
                        tab_only_filters.append(tab_filter)

                if tab_only_filters:
                    for filter_dict in tab_only_filters:
                        results = self.search(query, filter_dict, top_k * 2)  # Get more results

                        for doc in results:
                            content_key = doc.get("content", "")[:100]
                            if content_key not in seen_content:
                                seen_content.add(content_key)
                                all_results.append(doc)

                    if all_results:
                        print(f"[FALLBACK] Found {len(all_results)} documents using semantic search (no country filter)")
                        # Apply limit after collecting from all fallback filters
                        if len(all_results) > max_total_docs:
                            all_results = all_results[:max_total_docs]

        # Final fallback: drop every filter except the owner of private documents. Searching with
        # no filter at all would return other clients' private documents.
        if not all_results and fallback_to_unfiltered:
            scope = _tenant_scope(filter_combinations)
            print(f"\n[FINAL FALLBACK] No results with any filters. Searching with only {scope or 'no filters'}...")
            results = self.search(query, scope, top_k * 2)

            for doc in results:
                content_key = doc.get("content", "")[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_results.append(doc)

            if all_results:
                print(f"[FINAL FALLBACK] Found {len(all_results)} documents using pure semantic search (no filters)")
                # Apply limit after collecting all fallback results
                if len(all_results) > max_total_docs:
                    all_results = all_results[:max_total_docs]

        return all_results

def get_vector_stores():
    """
    [DEPRECATED] Old tab-based vector store system.

    This function is deprecated in favor of get_vector_stores_v2() which uses
    visibility-based stores (public/private) with tab as metadata filter.

    For migration:
    - Old: vector_stores[tab]
    - New: get_vector_stores_v2()["public"].search_with_metadata(query, tab=tab)

    This function is maintained for backward compatibility only.
    """
    print("Warning: get_vector_stores() is deprecated. Use get_vector_stores_v2() instead.")
    vector_stores = {}
    from app.services.llm_factory import LLMFactory
    embedding_function, _ = LLMFactory.get_embedding_function(
        model_provider=settings.model_provider,
        model_name=settings.openai_embedding_model
    )
    for tab in settings.supported_tabs:
        vector_stores[tab] = VectorStore.get_mongo_vector_store(
            connection_string=settings.mongodb_connection_string,
            database_name=settings.mongodb_database_name,
            collection_name=settings.get_mongodb_collection_name(tab),
            embedding_function=embedding_function,
            store_index=settings.get_mongodb_index_name(tab)
        )
    return vector_stores

class QdrantVectorStore(VectorStoreV2):
    def __init__(self, client: QdrantClient, collection_name: str, embedding_dimension: int, embedding_function: Any):
        self._client = client
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension
        # Reuses the cache-lookup embedding when similarity_search embeds the same query again
        self._embedding_function = with_query_cache(embedding_function)
        self._ensure_collection_exists()
        self._sparse_embeddings = FastEmbedSparse(
            model_name="Qdrant/bm25"
        )
        self._vector_store = None 
        self._init_vector_store()

    def _ensure_collection_exists(self):
        if not self._client.collection_exists(self._collection_name):
            print(f"Creating Qdrant collection: {self._collection_name}")
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    self._collection_name+"_dense":
                    VectorParams(
                    size=self._embedding_dimension,
                    distance=Distance.COSINE
                )},
                sparse_vectors_config={self._collection_name+"_sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=False,
                        ),
                        # BM25 needs inverse document frequency (IDF), so that rare words count for more
                        # than common ones. Qdrant computes it when searching.
                        modifier=Modifier.IDF,
                    )}
                )
            return

        print(f"Qdrant collection '{self._collection_name}' already exists.")
        self._ensure_sparse_idf()

        # # Debug: Check actual payload structure
        # try:
        #     # Get one point to inspect payload structure
        #     points = self._client.scroll(
        #         collection_name=self._collection_name,
        #         limit=1,
        #         with_payload=True
        #     )[0]

        #     if points:
        #         sample_payload = points[0].payload
        #         print(f"\n[DEBUG] Sample document payload keys: {list(sample_payload.keys())}")
        #         print(f"[DEBUG] Sample payload: {sample_payload}\n")
        # except Exception as e:
        #     print(f"Could not inspect payload: {e}")

        # Ensure payload indexes exist for filtering
        from qdrant_client.models import PayloadSchemaType
        try:
            # Create indexes for metadata fields (use nested path since LangChain stores under 'metadata')
            metadata_fields = ["metadata.tab", "metadata.country", "metadata.isoAlpha2", "metadata.visibility", "metadata.client_id"]
            for field in metadata_fields:
                try:
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    # print(f"Created payload index for field: {field}")
                except Exception as e:
                    # Index might already exist, which is fine
                    if "already exists" not in str(e).lower():
                        print(f"Note: Could not create index for {field}: {e}")
        except Exception as e:
            print(f"Warning: Could not create payload indexes: {e}")
    def _ensure_sparse_idf(self) -> None:
        """
        Turn on IDF for the keyword (BM25) vectors of a collection created without it. Without IDF,
        every word of a question counts the same, so a chunk that repeats a common word can outrank
        the only chunk that names a rare one, such as a country. IDF is applied when searching, so
        the chunks already stored do not need to be indexed again.
        """
        sparse_name = self._collection_name + "_sparse"
        try:
            sparse_vectors = self._client.get_collection(self._collection_name).config.params.sparse_vectors or {}
            params = sparse_vectors.get(sparse_name)
            if params is None or params.modifier == Modifier.IDF:
                return
            self._client.update_collection(
                collection_name=self._collection_name,
                sparse_vectors_config={sparse_name: SparseVectorParams(modifier=Modifier.IDF)},
            )
            print(f"Turned on IDF for the keyword vectors of '{self._collection_name}'.")
        except Exception as e:
            # Searches still work without IDF, so this must not stop the backend from starting
            print(f"Warning: could not turn on IDF for the keyword vectors of '{self._collection_name}': {e}")

    def _init_vector_store(self):
        # note that the name QDStore is used to avoid name conflict with langchain VectorStore
        self._vector_store = QDStore(
            client=self._client,
            collection_name=self._collection_name,
            embedding=self._embedding_function,
            vector_name=self._collection_name+"_dense",
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=self._sparse_embeddings,
            sparse_vector_name=self._collection_name+"_sparse"
        )
            
    def add_documents(self, documents: List[Dict[str, Any]]):
        # Errors are raised, not printed: a caller must not report documents as stored when they
        # are not
        docs = [Document(page_content=doc['content'], metadata={k: v for k, v in doc.items() if k != 'content'}) for doc in documents]
        self._vector_store.add_documents(docs)

    def _to_qdrant_filter(self, filter: Dict[str, Any]):
        """A plain {field: value} filter as a Qdrant filter on the metadata payload, or None."""
        conditions = _sanitize_filter(filter)
        if not conditions:
            return None
        from qdrant_client.models import Filter as QdrantFilter, FieldCondition, MatchValue

        return QdrantFilter(must=[
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in conditions.items()
        ])

    def search(self, query: str, filter: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Check query cache first (None while Redis is down: then search without it)
            query_cache = get_query_cache_or_none()
            if query_cache is not None:
                # Generate embedding for cache lookup
                query_embedding = self._embedding_function.embed_query(query)
                query_embedding_np = np.array(query_embedding)

                # Try to get from cache
                cache_filters = _cache_filters(self._collection_name, filter)
                cached_results = query_cache.get(query, query_embedding_np, cache_filters)
                if cached_results is not None:
                    return cached_results

            qdrant_filter = self._to_qdrant_filter(filter)
            print(f"[VECTOR STORE SEARCH] Filters: {_sanitize_filter(filter) or 'none - searching all documents'}")

            results = self._vector_store.similarity_search(
                query,
                filter=qdrant_filter,
                k=top_k
            )

            print(f"[VECTOR STORE SEARCH] Found {len(results)} documents")
            if results:
                for idx, doc in enumerate(results[:3], 1):
                    print(f"  Doc {idx}: country={doc.metadata.get('country', 'N/A')}, tab={doc.metadata.get('tab', 'N/A')}")

            # Format results
            formatted_results = [{"content": doc.page_content, **doc.metadata} for doc in results]

            # Cache the results
            if query_cache is not None:
                query_cache.set(query, query_embedding_np, cache_filters, formatted_results)

            return formatted_results
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def delete_documents(self, filter: Dict[str, Any]):
        """Delete every chunk whose metadata matches ``filter``. An empty filter is refused."""
        qdrant_filter = self._to_qdrant_filter(filter)
        if qdrant_filter is None:
            raise ValueError("delete_documents needs a filter; use clear_store to delete everything")
        from qdrant_client.models import FilterSelector

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(filter=qdrant_filter),
        )

    def update_document(self, document_id: str, updates: Dict[str, Any]):
        # A chunk's vectors come from its text, so an update means deleting and re-adding it
        raise NotImplementedError("Updating a stored chunk is not supported; delete and re-add it")

    def clear_store(self):
        self._client.delete_collection(self._collection_name)
        self._ensure_collection_exists()
        self._init_vector_store()
        print(f"Vector store '{self._collection_name}' cleared successfully.")

    def get_vector_store(self):
        return self._vector_store


class MongoDBVectorStore(VectorStoreV2):
    def __init__(self, connection_string: str, database_name: str, collection_name: str, embedding_function: Any, store_index: str):
        self._vector_store = VectorStore.get_mongo_vector_store(
            connection_string=connection_string,
            database_name=database_name,
            collection_name=collection_name,
            # Reuses the cache-lookup embedding when similarity_search embeds the same query again
            embedding_function=with_query_cache(embedding_function),
            store_index=store_index
        )

    def add_documents(self, documents: List[Dict[str, Any]]):
        docs = [Document(page_content=doc['content'], metadata={k: v for k, v in doc.items() if k != 'content'}) for doc in documents]
        self._vector_store.add_documents(docs)

    @staticmethod
    def _to_mongo_filter(filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        A plain {field: value} filter in the form Atlas Vector Search pre-filters take. LangChain
        stores metadata as top-level fields, and each field must be declared as a filter field in
        the Atlas vector index.
        """
        conditions = _sanitize_filter(filter)
        return {key: {"$eq": value} for key, value in conditions.items()} or None

    def search(self, query: str, filter: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Check query cache first (None while Redis is down: then search without it)
            query_cache = get_query_cache_or_none()
            if query_cache is not None:
                # Generate embedding for cache lookup
                query_embedding = self._vector_store.embedding.embed_query(query)
                query_embedding_np = np.array(query_embedding)

                # Try to get from cache
                cache_filters = _cache_filters(self._vector_store.collection.name, filter)
                cached_results = query_cache.get(query, query_embedding_np, cache_filters)
                if cached_results is not None:
                    return cached_results

            # Before, this passed a Qdrant filter object, which MongoDB cannot use
            pre_filter = self._to_mongo_filter(filter)
            print(f"[VECTOR STORE SEARCH] Filters: {pre_filter or 'none - searching all documents'}")

            results = self._vector_store.similarity_search(
                query,
                k=top_k,
                pre_filter=pre_filter
            )

            print(f"[VECTOR STORE SEARCH] Found {len(results)} documents")
            if results:
                for idx, doc in enumerate(results[:3], 1):
                    print(f"  Doc {idx}: country={doc.metadata.get('country', 'N/A')}, tab={doc.metadata.get('tab', 'N/A')}")

            # Format results
            formatted_results = [{"content": doc.page_content, **doc.metadata} for doc in results]

            # Cache the results
            if query_cache is not None:
                query_cache.set(query, query_embedding_np, cache_filters, formatted_results)

            return formatted_results
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def delete_documents(self, filter: Dict[str, Any]):
        """Delete every chunk whose metadata matches ``filter``. An empty filter is refused."""
        conditions = _sanitize_filter(filter)
        if not conditions:
            raise ValueError("delete_documents needs a filter; use clear_store to delete everything")
        self._vector_store.collection.delete_many(conditions)

    def update_document(self, document_id: str, updates: Dict[str, Any]):
        # A chunk's vectors come from its text, so an update means deleting and re-adding it
        raise NotImplementedError("Updating a stored chunk is not supported; delete and re-add it")

    def clear_store(self):
        self._vector_store.collection.drop()
        print(f"Vector store '{self._vector_store.collection.name}' cleared successfully.")

    def get_vector_store(self):
        return self._vector_store


class VectorStoreFactory:
    @staticmethod
    def get_vector_store(store_type: str, **kwargs) -> VectorStoreV2:
        if store_type == "qdrant":
            # validate required kwargs
            required_kwargs = ["client", "collection_name", "embedding_dimension", "embedding_function"]
            for kwarg in required_kwargs:
                if kwarg not in kwargs:
                    raise ValueError(f"Missing required argument: {kwarg}")
            return QdrantVectorStore(
                client=kwargs.get("client"),
                collection_name=kwargs.get("collection_name"),
                embedding_dimension=kwargs.get("embedding_dimension"),
                embedding_function=kwargs.get("embedding_function")
            )
        elif store_type == "mongodb":
            # validate required kwargs
            required_kwargs = ["connection_string", "database_name", "collection_name", "embedding_function", "store_index"]
            for kwarg in required_kwargs:
                if kwarg not in kwargs:
                    raise ValueError(f"Missing required argument: {kwarg}")
            return MongoDBVectorStore(
                connection_string=kwargs.get("connection_string"),
                database_name=kwargs.get("database_name"),
                collection_name=kwargs.get("collection_name"),
                embedding_function=kwargs.get("embedding_function"),
                store_index=kwargs.get("store_index")
            )
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")

_vector_stores = None

def get_vector_stores_v2() -> Dict[str, VectorStoreV2]:
    print("Initializing vector stores...")
    global _vector_stores
    
    if _vector_stores is not None:
        return _vector_stores

    embedding_function, dims = LLMFactory.get_embedding_function(
        model_provider=settings.model_provider,
        model_name=settings.openai_embedding_model
    )
    if settings.vector_store == "mongodb":
        _vector_stores = {
            "public": VectorStoreFactory.get_vector_store(
                store_type=settings.vector_store,
                connection_string=settings.mongodb_connection_string,
                database_name=settings.mongodb_database_name,
                collection_name=settings.mongodb_collection_name_public,
                embedding_function=embedding_function,
                store_index=settings.mongodb_index_name_public
            ),
            "private": VectorStoreFactory.get_vector_store(
                store_type=settings.vector_store,
                connection_string=settings.mongodb_connection_string,
                database_name=settings.mongodb_database_name,
                collection_name=settings.mongodb_collection_name_private,
                embedding_function=embedding_function,
                store_index=settings.mongodb_index_name_private
            )
        }
    else:   
        client = QdrantClient(url=settings.qdrant_host,
                              api_key=settings.qdrant_api_key)
        _vector_stores = {
            "public": VectorStoreFactory.get_vector_store(
                store_type=settings.vector_store,
                client=client,
                collection_name=settings.qdrant_collection_name_public,
                embedding_dimension=dims,
                embedding_function=embedding_function
            ),
            "private": VectorStoreFactory.get_vector_store(
                store_type=settings.vector_store,
                client=client,
                collection_name=settings.qdrant_collection_name_private,
                embedding_dimension=dims,
                embedding_function=embedding_function
            )
        }

    return _vector_stores


# Helper functions for unified access

def get_public_vector_store() -> VectorStoreV2:
    """
    Get the public vector store instance.

    Returns:
        Public VectorStoreV2 instance
    """
    stores = get_vector_stores_v2()
    return stores["public"]


def get_private_vector_store() -> VectorStoreV2:
    """
    Get the private vector store instance.

    Returns:
        Private VectorStoreV2 instance
    """
    stores = get_vector_stores_v2()
    return stores["private"]


def search_all_stores(
    query: str,
    visibility: str = "public",
    tab: str = None,
    metadata_filters: Dict[str, Any] = None,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Search across vector stores with unified interface.

    Args:
        query: Search query
        visibility: "public" or "private"
        tab: Optional tab filter
        metadata_filters: Additional metadata filters
        top_k: Number of results

    Returns:
        List of matching documents

    Example:
        >>> results = search_all_stores(
        ...     "Kenya data protection laws",
        ...     visibility="public",
        ...     tab="data-protection"
        ... )
    """
    stores = get_vector_stores_v2()
    store = stores.get(visibility)

    if not store:
        raise ValueError(f"Invalid visibility: {visibility}")

    return store.search_with_metadata(query, tab, metadata_filters, top_k)


def search_with_auto_filters(
    query: str,
    visibility: str = "public",
    tab: str = None,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Search with automatic metadata filter extraction.

    Automatically detects entities (countries, document types) from query
    and applies appropriate filters.

    Args:
        query: Search query
        visibility: "public" or "private"
        tab: Optional tab override
        top_k: Number of results per filter combination

    Returns:
        Deduplicated list of matching documents

    Example:
        >>> results = search_with_auto_filters(
        ...     "What are Kenya's cross-border data policies?",
        ...     visibility="public"
        ... )
        # Automatically filters by country=Kenya, document_type=cross_border_policy
    """
    from app.services.filter_builder import get_filter_builder

    # Build filters automatically
    filter_builder = get_filter_builder()
    filter_combinations = filter_builder.build_filter_for_vectorstore(query, tab)

    # Get appropriate store
    stores = get_vector_stores_v2()
    store = stores.get(visibility)

    if not store:
        raise ValueError(f"Invalid visibility: {visibility}")

    # Search with filter combinations
    return store.search_with_filter_combinations(
        query, filter_combinations, top_k
    )
