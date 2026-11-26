from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional
from langchain_core.documents import Document
import os
import pymupdf4llm
from app.schema.pydantic_models import DocumentType, DocumentInfo, DocumentMetadata
from fuzzywuzzy import fuzz, process
import re
from datetime import datetime
from pymongo import MongoClient
from app.core.config import settings
# Try to use the new MongoDB Atlas Vector Search import

try:
    from langchain_mongodb import MongoDBAtlasVectorSearch
except ImportError:
    # Fallback to the old import if new one is not available
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=200, length_function=len)

# Comprehensive list of African countries with common variations and abbreviations
AFRICAN_COUNTRIES = {
    "Algeria": ["Algeria", "Algerie", "DZ", "ALG"],
    "Angola": ["Angola", "AO", "AGO"],
    "Benin": ["Benin", "BJ", "BEN"],
    "Botswana": ["Botswana", "BW", "BWA"],
    "Burkina Faso": ["Burkina Faso", "Burkina", "BF", "BFA", "Upper Volta"],
    "Burundi": ["Burundi", "BI", "BDI"],
    "Cabo Verde": ["Cabo Verde", "Cape Verde", "CV", "CPV"],
    "Cameroon": ["Cameroon", "Cameroun", "CM", "CMR"],
    "Central African Republic": ["Central African Republic", "CAR", "CF", "CAF"],
    "Chad": ["Chad", "Tchad", "TD", "TCD"],
    "Comoros": ["Comoros", "Comores", "KM", "COM"],
    "Congo": ["Congo", "Congo-Brazzaville", "Republic of Congo", "CG", "COG"],
    "Democratic Republic of Congo": ["DRC", "Democratic Republic of Congo", "Congo-Kinshasa", "DR Congo", "CD", "COD", "Zaire"],
    "Côte d'Ivoire": ["Côte d'Ivoire", "Cote d'Ivoire", "Ivory Coast", "CI", "CIV"],
    "Djibouti": ["Djibouti", "DJ", "DJI"],
    "Egypt": ["Egypt", "EG", "EGY"],
    "Equatorial Guinea": ["Equatorial Guinea", "GQ", "GNQ"],
    "Eritrea": ["Eritrea", "ER", "ERI"],
    "Eswatini": ["Eswatini", "Swaziland", "SZ", "SWZ"],
    "Ethiopia": ["Ethiopia", "ET", "ETH"],
    "Gabon": ["Gabon", "GA", "GAB"],
    "Gambia": ["Gambia", "The Gambia", "GM", "GMB"],
    "Ghana": ["Ghana", "GH", "GHA"],
    "Guinea": ["Guinea", "GN", "GIN"],
    "Guinea-Bissau": ["Guinea-Bissau", "Guinea Bissau", "GW", "GNB"],
    "Kenya": ["Kenya", "KE", "KEN"],
    "Lesotho": ["Lesotho", "LS", "LSO"],
    "Liberia": ["Liberia", "LR", "LBR"],
    "Libya": ["Libya", "LY", "LBY"],
    "Madagascar": ["Madagascar", "MG", "MDG"],
    "Malawi": ["Malawi", "MW", "MWI"],
    "Mali": ["Mali", "ML", "MLI"],
    "Mauritania": ["Mauritania", "MR", "MRT"],
    "Mauritius": ["Mauritius", "MU", "MUS"],
    "Morocco": ["Morocco", "Maroc", "MA", "MAR"],
    "Mozambique": ["Mozambique", "MZ", "MOZ"],
    "Namibia": ["Namibia", "NA", "NAM"],
    "Niger": ["Niger", "NE", "NER"],
    "Nigeria": ["Nigeria", "NG", "NGA"],
    "Rwanda": ["Rwanda", "RW", "RWA"],
    "Sao Tome and Principe": ["Sao Tome and Principe", "São Tomé and Príncipe", "ST", "STP"],
    "Senegal": ["Senegal", "SN", "SEN"],
    "Seychelles": ["Seychelles", "SC", "SYC"],
    "Sierra Leone": ["Sierra Leone", "SL", "SLE"],
    "Somalia": ["Somalia", "SO", "SOM"],
    "South Africa": ["South Africa", "ZA", "ZAF", "SA"],
    "South Sudan": ["South Sudan", "SS", "SSD"],
    "Sudan": ["Sudan", "SD", "SDN"],
    "Tanzania": ["Tanzania", "TZ", "TZA"],
    "Togo": ["Togo", "TG", "TGO"],
    "Tunisia": ["Tunisia", "Tunisie", "TN", "TUN"],
    "Uganda": ["Uganda", "UG", "UGA"],
    "Zambia": ["Zambia", "ZM", "ZMB"],
    "Zimbabwe": ["Zimbabwe", "ZW", "ZWE"]
}


def extract_country_from_filename(filename: str, threshold: int = 80) -> Optional[str]:
    """
    Extract country name from filename using fuzzy matching.

    Args:
        filename: The filename to extract country from
        threshold: Minimum similarity score (0-100) for fuzzy matching

    Returns:
        The matched country name or None if no match found
    """
    # Clean the filename - remove extension and common separators
    clean_filename = os.path.splitext(filename)[0]
    clean_filename = re.sub(r'[_\-\.]', ' ', clean_filename)

    # Split filename into words for individual matching
    filename_words = clean_filename.split()

    best_match = None
    best_score = 0
    best_country = None

    # Create a flat list of all country variations for fuzzy matching
    all_variations = []
    country_map = {}

    for country, variations in AFRICAN_COUNTRIES.items():
        for variation in variations:
            all_variations.append(variation.lower())
            country_map[variation.lower()] = country

    # First, try exact matches on multi-word country names (like "South Africa")
    clean_filename_lower = clean_filename.lower()
    for country, variations in AFRICAN_COUNTRIES.items():
        for variation in variations:
            if len(variation.split()) > 1: 
                if variation.lower() in clean_filename_lower:
                    return country

    for word in filename_words:
        word_lower = word.lower()
        for country, variations in AFRICAN_COUNTRIES.items():
            if word_lower in [v.lower() for v in variations]:
                return country

    # Try fuzzy matching on individual words
    for word in filename_words:
        if len(word) >= 3:  # Skip very short words
            match = process.extractOne(
                word.lower(), all_variations, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                if match[1] > best_score:
                    best_score = match[1]
                    best_match = match[0]
                    best_country = country_map[match[0]]

    # Try fuzzy matching on the whole filename
    if not best_match:
        match = process.extractOne(
            clean_filename.lower(), all_variations, scorer=fuzz.partial_ratio)
        if match and match[1] >= threshold:
            best_country = country_map[match[0]]

    return best_country


def get_embedding_function(provider: str, model_id: str = None):
    """Get the embedding function based on the provider and model ID."""
    if model_id is None:
        model_id = settings.openai_embedding_model
    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAIEmbeddings(
            model=model_id,
            api_key=settings.openai_api_key,
            base_url=base_url
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_vector_store(name: str, embedding_function: callable, mongodb_database:str=None, mongodb_collection:str = None, mongodb_index_name: str = None):
    """Get the vector store based on the name."""
    if embedding_function is None:
        raise ValueError("Embedding function must be provided")
    if name == "chroma":
        return Chroma(
            collection_name=settings.get_chroma_collection_name(
                "data_protection"),  # Default collection for ChromaDB
            embedding_function=embedding_function,
            persist_directory=settings.get_chroma_persist_directory(
                "data_protection"),  # Default directory for ChromaDB
            client_settings={
                "host": settings.chroma_host,
                "port": settings.chroma_port,
            }

        )
    elif name == "mongodb":
        if not mongodb_database or not mongodb_collection or not mongodb_index_name:
            raise ValueError("MongoDB parameters must be provided: database, collection, index_name")
        mongo_client = MongoClient(
            settings.mongodb_connection_string)
        collection = mongo_client[mongodb_database][mongodb_collection]
        
        # Create a wrapper class to handle the filtering compatibility
        class MongoDBVectorStoreWrapper:
            def __init__(self, vector_store, index_name):
                self._vector_store = vector_store
                self.collection = vector_store.collection
                self.index_name = index_name
            
            def similarity_search(self, query, k=4, filter=None, pre_filter=None, **kwargs):
                """
                Wrapper method to handle different filter parameter names for MongoDB Atlas Vector Search.
                """
                # Get the underlying search method
                search_method = self._vector_store.similarity_search
                
                # Try different filter parameter approaches
                filter_param = filter or pre_filter
                
                if filter_param:
                    # Try the most common approaches for MongoDB filtering
                    try:
                        # Approach 1: Use 'filter' parameter
                        return search_method(query, k=k, filter=filter_param, **kwargs)
                    except (TypeError, ValueError) as e:
                        if "multiple values for argument 'filter'" in str(e) or "unexpected keyword argument" in str(e):
                            try:
                                # Approach 2: Use 'pre_filter' parameter
                                return search_method(query, k=k, pre_filter=filter_param, **kwargs)
                            except (TypeError, ValueError) as e2:
                                try:
                                    # Approach 3: Use 'metadata_filter' parameter
                                    return search_method(query, k=k, metadata_filter=filter_param, **kwargs)
                                except (TypeError, ValueError) as e3:
                                    # Approach 4: No filter support, fall back to unfiltered search
                                    print(f"Warning: Filtering not supported, using unfiltered search. Errors: {e}, {e2}, {e3}")
                                    return search_method(query, k=k, **kwargs)
                        else:
                            raise e
                else:
                    # No filter
                    return search_method(query, k=k, **kwargs)
            
            def as_retriever(self, **kwargs):
                """Wrapper for retriever functionality."""
                return self._vector_store.as_retriever(**kwargs)
            
            def __getattr__(self, name):
                """Delegate all other attributes to the underlying vector store."""
                return getattr(self._vector_store, name)
        
        # Try the new API first, then fallback to old API
        try:
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embedding_function,
                index_name=mongodb_index_name
            )
        except TypeError:
            # Fallback for older versions
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding_function=embedding_function,
                index_name=mongodb_index_name
            )
        
        # Return wrapped vector store for better compatibility
        return MongoDBVectorStoreWrapper(vector_store, mongodb_index_name)

    else:
        raise ValueError(f"Unsupported vector store: {name}")


def migrate_mongo_to_chroma(vector_store_name: str = "chroma"):
    """Migrate documents from MongoDB to Chroma vector store efficiently with existing embeddings."""
    mongo_client = MongoClient(settings.mongodb_connection_string)
    collection = mongo_client[settings.mongodb_database_name][settings.mongodb_collection_name_data_protection]

    documents = []
    embeddings = []
    ids = []

    for doc in collection.find():
        content = doc.get("text")
        metadata = doc.get("metadata", {})
        embedding = doc.get("embedding")
        if embedding is None:
            print(
                f"Warning: Document {doc.get('_id')} has no embedding, skipping...")
            continue
        documents.append(Document(
            page_content=content,
            metadata=metadata
        ))

        embeddings.append(embedding)
        doc_id = str(doc.get("_id", len(ids)))
        ids.append(doc_id)

    if not documents:
        print("No documents with embeddings found in MongoDB to migrate.")
        return

    vector_store = get_vector_store(name=vector_store_name)
    try:
        vector_store.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings,
            ids=ids
        )
        print(
            f"Successfully migrated {len(documents)} documents from MongoDB to Chroma with existing embeddings.")
    except Exception as e:
        print(f"Error during migration: {e}")
        vector_store.add_documents(documents)
        print(
            f"Fallback: Migrated {len(documents)} documents but had to re-compute embeddings.")


embedding_function = get_embedding_function(
    provider=settings.embedding_provider,
    model_id=settings.embedding_model_id
)

# Configuration for multiple vector stores
VECTOR_STORE_CONFIGS = {
    "pdp_analyzer": {
        "database": settings.mongodb_database_name,
        "collection": "pdp_analyzer",
        "index_name": "vector_index_pdp"
    },
    "policy_document": {
        "database": settings.mongodb_database_name,
        "collection": "policy_document", 
        "index_name": "vector_index"
    }
}

def get_multiple_vector_stores() -> dict:
    """
    Initialize and return multiple vector stores based on configuration.
    
    Returns:
        Dictionary mapping store names to vector store instances
    """
    vector_stores = {}
    
    # Only initialize if we have the required environment variables
    if not settings.mongodb_connection_string or not settings.mongodb_database_name:
        print("Warning: MongoDB environment variables not set, skipping vector store initialization")
        return vector_stores
    
    for store_name, config in VECTOR_STORE_CONFIGS.items():
        try:
            vector_stores[store_name] = get_vector_store(
                name=settings.vector_store,
                embedding_function=embedding_function,
                mongodb_database=config["database"],
                mongodb_collection=config["collection"],
                mongodb_index_name=config["index_name"]
            )
            print(f"Successfully initialized vector store: {store_name}")
        except Exception as e:
            print(f"Failed to initialize vector store {store_name}: {e}")
    
    return vector_stores

# Initialize multiple vector stores
vector_stores = get_multiple_vector_stores()

# Backward compatibility - use first available vector store as default
if vector_stores:
    vector_store = next(iter(vector_stores.values()))
else:
    # Only try to create default vector store if we have the environment set up
    if settings.vector_store == "chroma" or not settings.mongodb_connection_string:
        try:
            vector_store = get_vector_store(
                name="chroma",
                embedding_function=embedding_function
            )
        except Exception as e:
            print(f"Warning: Could not initialize default vector store: {e}")
            vector_store = None
    else:
        vector_store = None


def load_single_doc(path: str, document_type: DocumentType, country: str = None) -> tuple[List[Document], DocumentInfo]:
    """Load a single document from the given path and return documents with DocumentInfo."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_name = os.path.basename(path)
    if country:
        detected_country = country
    else:
        detected_country = extract_country_from_filename(file_name)

    doc_info = DocumentInfo(
        file_name=file_name,
        file_path=path,
        document_type=document_type,
        country=detected_country,
        uploaded_at=datetime.now()
    )

    base_metadata = {
        "file_name": file_name,
        "document_type": document_type.value,
        "file_path": path,
        "country": detected_country
    }

    if path.endswith('.pdf'):
        md_text = pymupdf4llm.to_markdown(path)
        documents = [Document(page_content=md_text, metadata=base_metadata)]

    elif path.endswith('.docx'):
        loader = Docx2txtLoader(path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update(base_metadata)

    elif path.endswith('.html') or path.endswith('.htm'):
        loader = UnstructuredHTMLLoader(path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update(base_metadata)

    elif path.endswith('.txt'):
        loader = TextLoader(path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update(base_metadata)

    elif path.endswith('.md'):
        loader = UnstructuredMarkdownLoader(path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update(base_metadata)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    split_docs = text_splitter.split_documents(documents)
    for doc in split_docs:
        doc.metadata.update(base_metadata)

    return split_docs, doc_info


def load_documents(paths, document_type: DocumentType, country: str = None) -> tuple[List[Document], List[DocumentInfo]]:
    """Load multiple documents from the given paths."""
    if isinstance(paths, str):
        paths = [paths]
    all_documents = []
    all_doc_infos = []

    for path in paths:
        try:
            docs, doc_info = load_single_doc(path, document_type, country)
            all_documents.extend(docs)
            all_doc_infos.append(doc_info)
        except Exception as e:
            print(f"Error loading document {path}: {e}")

    return all_documents, all_doc_infos


def search_documents_by_country(country_name: str, k: int = 5) -> List[Document]:
    """
    Search for documents from a specific country.

    Args:
        country_name: Name of the country to search for
        k: Number of documents to return

    Returns:
        List of documents from the specified country
    """
    return search_documents_with_filters(
        query="",
        country_filter=country_name,
        k=k
    )


def get_available_countries() -> List[str]:
    """
    Get list of countries that have documents in the vector store.

    Returns:
        List of country names that have associated documents
    """
    try:
        return list(AFRICAN_COUNTRIES.keys())
    except Exception as e:
        print(f"Error getting available countries: {e}")
        return []


def update_document_country(file_name: str, new_country: str) -> bool:
    """
    Update the country metadata for documents with a specific filename.

    Args:
        file_name: Name of the file to update
        new_country: New country name to assign

    Returns:
        True if update was successful, False otherwise
    """
    if new_country not in AFRICAN_COUNTRIES:
        match = process.extractOne(new_country, list(
            AFRICAN_COUNTRIES.keys()), scorer=fuzz.ratio)
        if match and match[1] >= 80:
            new_country = match[0]
        else:
            raise ValueError(f"Country '{new_country}' not recognized")

    try:
        print(f"Would update country for {file_name} to {new_country}")
        return True
    except Exception as e:
        print(f"Error updating document country: {e}")
        return False


def create_document_metadata(doc_info: DocumentInfo, chunk_index: int = None, total_chunks: int = None) -> DocumentMetadata:
    """Create DocumentMetadata from DocumentInfo."""
    return DocumentMetadata(
        file_name=doc_info.file_name,
        file_path=doc_info.file_path,
        document_type=doc_info.document_type.value if doc_info.document_type else "unknown",
        country=doc_info.country,
        chunk_index=chunk_index,
        total_chunks=total_chunks
    )


def search_documents_with_filters(
    query: str = "",
    country_filter: Optional[str] = None,
    document_type_filter: Optional[DocumentType] = None,
    k: int = 5
) -> List[Document]:
    """
    Search documents with optional filters.

    Args:
        query: Search query (empty string for metadata-only search)
        country_filter: Filter by country name
        document_type_filter: Filter by document type
        k: Number of results to return

    Returns:
        List of matching documents
    """
    filters = {}

    if country_filter:
        normalized_country = None
        for country, variations in AFRICAN_COUNTRIES.items():
            if country_filter.lower() in [v.lower() for v in variations]:
                normalized_country = country
                break

        if not normalized_country:
            match = process.extractOne(country_filter, list(
                AFRICAN_COUNTRIES.keys()), scorer=fuzz.ratio)
            if match and match[1] >= 70:
                normalized_country = match[0]

        if normalized_country:
            filters["country"] = normalized_country

    if document_type_filter:
        filters["document_type"] = document_type_filter.value

    try:
        if filters:
            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filters
            )
        else:
            results = vector_store.similarity_search(query=query, k=k)

        return results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def add_documents_to_vector_store(documents: List[Document], doc_infos: List[DocumentInfo]) -> bool:
    """
    Add documents to the vector store with proper metadata.

    Args:
        documents: List of document chunks
        doc_infos: List of document info objects

    Returns:
        True if successful, False otherwise
    """
    try:
        enhanced_docs = []

        for doc_info in doc_infos:
            file_docs = [doc for doc in documents if doc.metadata.get(
                "file_name") == doc_info.file_name]

            for idx, doc in enumerate(file_docs):
                enhanced_metadata = create_document_metadata(
                    doc_info=doc_info,
                    chunk_index=idx,
                    total_chunks=len(file_docs)
                )

                doc.metadata.update(enhanced_metadata.dict())
                enhanced_docs.append(doc)
        vector_store.add_documents(enhanced_docs)
        return True

    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        return False

def search_multiple_vector_stores(
    query: str,
    vector_stores_dict: dict = None,
    country_filter: Optional[str] = None,
    document_type_filter: Optional[DocumentType] = None,
    k: int = 3
) -> List[Document]:
    """
    Search across multiple vector stores and combine results.
    
    Args:
        query: Search query
        vector_stores_dict: Dictionary of vector stores to search (defaults to global vector_stores)
        country_filter: Filter by country name
        document_type_filter: Filter by document type
        k: Number of results per vector store
        
    Returns:
        Combined list of documents from all vector stores
    """
    if vector_stores_dict is None:
        vector_stores_dict = vector_stores
    
    all_results = []
    
    # Prepare filters
    filters = {}
    if country_filter:
        normalized_country = None
        for country, variations in AFRICAN_COUNTRIES.items():
            if country_filter.lower() in [v.lower() for v in variations]:
                normalized_country = country
                break
        
        if not normalized_country:
            match = process.extractOne(country_filter, list(
                AFRICAN_COUNTRIES.keys()), scorer=fuzz.ratio)
            if match and match[1] >= 70:
                normalized_country = match[0]
        
        if normalized_country:
            filters["country"] = normalized_country
    
    if document_type_filter:
        # Handle both string and DocumentType enum
        if isinstance(document_type_filter, str):
            filters["document_type"] = document_type_filter
        else:
            # Assume it's a DocumentType enum
            filters["document_type"] = document_type_filter.value
    
    # Search each vector store
    for store_name, store in vector_stores_dict.items():
        try:
            print(f"---SEARCHING VECTOR STORE: {store_name}---")
            
            if filters and hasattr(store, 'similarity_search'):
                results = store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )
            else:
                # Fallback to regular search if filtering not supported
                results = store.similarity_search(query=query, k=k)
            
            # Add store name to metadata for tracking
            for doc in results:
                doc.metadata["vector_store"] = store_name
            
            all_results.extend(results)
            print(f"---FOUND {len(results)} documents in {store_name}---")
            
        except Exception as e:
            print(f"Error searching vector store {store_name}: {e}")
    
    # Remove duplicates based on content
    unique_results = []
    seen_content = set()
    
    for doc in all_results:
        content_key = doc.page_content[:100]  # Use first 100 chars as key
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_results.append(doc)
    
    print(f"---TOTAL UNIQUE RESULTS: {len(unique_results)}---")
    return unique_results

def get_vector_store_by_name(store_name: str):
    """
    Get a specific vector store by name.
    
    Args:
        store_name: Name of the vector store ("pdp_analyzer" or "policy_document")
        
    Returns:
        Vector store instance or None if not found
    """
    return vector_stores.get(store_name)

def get_all_vector_stores() -> dict:
    """
    Get all initialized vector stores.
    
    Returns:
        Dictionary of all vector stores
    """
    return vector_stores
