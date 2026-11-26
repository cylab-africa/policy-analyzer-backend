from fuzzywuzzy import fuzz, process
from dotenv import load_dotenv
load_dotenv()
from app.core.config import settings
from app.db.db_utils import (
    insert_chat_history, get_recent_messages, count_conversation_messages,
    get_conversation_messages, delete_old_messages
)
from app.schema.pydantic_models import QueryInput, QueryOutput, DocumentType
from app.db.chroma_utils import (
    get_vector_store, get_embedding_function, AFRICAN_COUNTRIES, extract_country_from_filename,
    search_multiple_vector_stores, get_all_vector_stores, get_vector_store_by_name
)
from langgraph.graph import END, StateGraph, START
from langchain import hub
from langchain.schema import Document
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from typing import List, Literal, Optional, Dict
import os
import re

# from langchain_community.tools.tavily_search import TavilySearchResults


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class MultiQuery(BaseModel):
    """Generate multiple queries for better retrieval coverage."""

    queries: List[str] = Field(
        description="List of 3-5 reformulated queries that approach the question from different angles"
    )


class DocumentTypeDetection(BaseModel):
    """Detect document types relevant to the query."""

    document_types: List[str] = Field(
        description="List of relevant document types: cross_border_policy, digital_id_policy, data_protection_policy"
    )


class ConversationSummary(BaseModel):
    """Summarize a conversation history."""

    summary: str = Field(
        description="A concise summary of the conversation context, key topics discussed, and important conclusions"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        attempt_count: number of retrieval attempts
        detected_countries: list of countries detected in the query
        multi_queries: list of reformulated queries
        detected_document_types: list of document types relevant to the query
        conversation_context: summarized conversation history
        session_id: session identifier for history tracking
    """
    question: str
    generation: str
    documents: List[Document]
    attempt_count: int
    detected_countries: List[str]
    multi_queries: List[str]
    detected_document_types: List[str]
    conversation_context: str
    session_id: str


class AdaptiveRAGSystem:
    """
    Adaptive RAG system that intelligently routes queries and self-corrects.

    Supports multiple MongoDB Atlas vector stores:
    - pdp_analyzer collection with vector_index_pdp index
    - policy_document collection with vector_index index

    Features:
    - Multi-store document retrieval with automatic deduplication
    - Country and document type filtering across all stores
    - Multi-query generation for better retrieval coverage
    - Conversation history support
    - Intelligent routing between vector stores and web search
    """

    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: Optional[str] = None,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.0,
        max_attempts: int = 3
    ):
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_attempts = max_attempts

        # Initialize components
        self._setup_llm()
        self._setup_vectorstore()
        self._setup_web_search()
        self._setup_chains()
        self._build_graph()

    def _setup_llm(self):
        """Initialize LLM components."""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        self.llm = ChatOpenAI(model=self.model_name,
                              temperature=self.temperature)
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
        )

    def _setup_vectorstore(self):
        """Initialize multiple vectorstore retrievers."""
        try:
            # Get all available vector stores
            self.vector_stores = get_all_vector_stores()

            if not self.vector_stores:
                print("Warning: No vector stores available")
                self.retrievers = {}
                self.default_retriever = None
                return

            # Create retrievers for each vector store
            self.retrievers = {}
            for store_name, store in self.vector_stores.items():
                try:
                    self.retrievers[store_name] = store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    print(
                        f"Successfully initialized retriever for: {store_name}")
                except Exception as e:
                    print(
                        f"Warning: Could not initialize retriever for {store_name}: {e}")

            # Set default retriever (first available one for backward compatibility)
            self.default_retriever = next(
                iter(self.retrievers.values())) if self.retrievers else None

            # Also maintain the old single vectorstore/retriever for compatibility
            self.vectorstore = next(
                iter(self.vector_stores.values())) if self.vector_stores else None
            self.retriever = self.default_retriever

        except Exception as e:
            print(f"Warning: Could not initialize vectorstores: {e}")
            self.vector_stores = {}
            self.retrievers = {}
            self.default_retriever = None
            self.vectorstore = None
            self.retriever = None

    def _setup_web_search(self):
        """Initialize web search tool."""
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key
            self.web_search_tool = TavilySearch(k=3)
        else:
            self.web_search_tool = None
            print("Warning: No Tavily API key provided, web search disabled")

    def _setup_chains(self):
        """Setup various LLM chains for routing, grading, and generation."""

        router_system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains policy documents related to cross-border policies, digital ID policies, 
        data protection policies, and African governance documents.
        Use the vectorstore for questions about:
        - Policy analysis and interpretation
        - Cross-border regulations
        - Digital identity frameworks
        - Data protection laws
        - African governance and policy matters
        
        Use web search for:
        - Recent news and current events
        - General information not related to policies
        - Real-time data and statistics
        - Questions about recent developments
        """

        route_prompt = ChatPromptTemplate.from_messages([
            ("system", router_system),
            ("human", "{question}"),
        ])

        structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.question_router = route_prompt | structured_llm_router

        grader_system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", grader_system),
            ("human",
             "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.retrieval_grader = grade_prompt | structured_llm_grader

        hallucination_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", hallucination_system),
            ("human",
             "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])

        structured_llm_hallucination = self.llm.with_structured_output(
            GradeHallucinations)
        self.hallucination_grader = hallucination_prompt | structured_llm_hallucination

        answer_system = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", answer_system),
            ("human",
             "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])

        structured_llm_answer = self.llm.with_structured_output(GradeAnswer)
        self.answer_grader = answer_prompt | structured_llm_answer

        rewriter_system = """You are a question re-writer that converts an input question to a better version 
        that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying 
        semantic intent / meaning. Focus on policy-related terms and concepts."""

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", rewriter_system),
            ("human",
             "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])

        self.question_rewriter = rewrite_prompt | self.llm | StrOutputParser()
        try:
            rag_prompt = hub.pull("rlm/rag-prompt")
        except (Exception, KeyboardInterrupt) as e:
            # Log the specific error for debugging while falling back to default prompt
            print(f"Warning: Failed to pull rag-prompt from hub: {type(e).__name__}: {e}")
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant for question-answering tasks focused on policy analysis. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use three sentences maximum and keep the answer concise.
                
                Consider the conversation history when formulating your response to maintain continuity.
                
                Retrieved Context: {context}
                
                Conversation Context: {conversation_context}"""),
                ("human", "{question}")
            ])
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

    def _detect_countries_in_query(self, query: str, threshold: int = 80) -> List[str]:
        """
        Detect African countries mentioned in the query using fuzzy matching.
        Based on the existing extract_country_from_filename logic but adapted for query text.

        Args:
            query: The user query
            threshold: Minimum similarity score for fuzzy matching

        Returns:
            List of detected country names
        """
        detected_countries = []
        query_lower = query.lower()

        filename_extracted = extract_country_from_filename(query, threshold)
        if filename_extracted:
            detected_countries.append(filename_extracted)
        segment_countries = self._extract_countries_from_text_segments(query)
        for country in segment_countries:
            if country not in detected_countries:
                detected_countries.append(country)
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        clean_query = re.sub(r'\s+', ' ', clean_query.strip())
        query_words = clean_query.split()
        all_variations = []
        country_map = {}
        for country, variations in AFRICAN_COUNTRIES.items():
            for variation in variations:
                all_variations.append(variation.lower())
                country_map[variation.lower()] = country
        for country, variations in AFRICAN_COUNTRIES.items():
            for variation in variations:
                if len(variation.split()) > 1:  # Multi-word country names
                    if variation.lower() in query_lower:
                        if country not in detected_countries:
                            detected_countries.append(country)
        for word in query_words:
            word_lower = word.lower()
            for country, variations in AFRICAN_COUNTRIES.items():
                if word_lower in [v.lower() for v in variations]:
                    if country not in detected_countries:
                        detected_countries.append(country)
        for word in query_words:
            if len(word) >= 3:  # Skip very short words
                try:
                    match = process.extractOne(
                        word.lower(), all_variations, scorer=fuzz.ratio)
                    if match and match[1] >= threshold:
                        country = country_map[match[0]]
                        if country not in detected_countries:
                            detected_countries.append(country)
                except Exception:
                    continue

        return detected_countries

    def _extract_countries_from_text_segments(self, text: str) -> List[str]:
        """
        Extract countries from text by treating segments as filenames.
        This leverages the existing extract_country_from_filename function.

        Args:
            text: Text to extract countries from

        Returns:
            List of detected countries
        """
        countries = []
        segments = re.split(r'[.,;!?\n\r]+', text)

        for segment in segments:
            segment = segment.strip()
            if len(segment) > 2:  # Skip very short segments
                country = extract_country_from_filename(segment)
                if country and country not in countries:
                    countries.append(country)

        return countries

    def _generate_multi_queries(self, query: str) -> List[str]:
        """
        Generate multiple reformulated queries for better retrieval coverage.

        Args:
            query: The original user query

        Returns:
            List of reformulated queries
        """
        try:
            multi_query_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI assistant that generates multiple search queries for better information retrieval.
                Your task is to generate 3-5 different versions of the user question that:
                1. Approach the topic from different angles
                2. Use different terminology and keywords
                3. Focus on specific aspects of the question
                4. Maintain the core intent of the original question
                
                For policy-related queries, consider variations that include:
                - Different policy terms (regulation, framework, guidelines, standards)
                - Different perspectives (implementation, compliance, impact, requirements)
                - Different scopes (national, regional, cross-border, international)"""),
                ("human", "Original query: {query}")
            ])

            structured_multi_query = self.llm.with_structured_output(
                MultiQuery)
            multi_query_chain = multi_query_prompt | structured_multi_query

            result = multi_query_chain.invoke({"query": query})
            return result.queries

        except Exception as e:
            print(f"Multi-query generation failed: {e}")
            return [query]

    def _build_graph(self):
        """Build the LangGraph workflow."""
        def retrieve(state):
            """Retrieve documents from multiple vectorstores using multi-query, country filtering, and document type filtering."""
            print("---RETRIEVE FROM MULTIPLE VECTOR STORES---")
            question = state["question"]
            detected_countries = state.get("detected_countries", [])
            multi_queries = state.get("multi_queries", [question])
            detected_document_types = state.get("detected_document_types", [])
            conversation_context = state.get("conversation_context", "")
            session_id = state.get("session_id", "")

            if not self.vector_stores:
                print("---NO VECTOR STORES AVAILABLE---")
                return {
                    "documents": [],
                    "question": question,
                    "attempt_count": state.get("attempt_count", 0),
                    "detected_countries": detected_countries,
                    "multi_queries": multi_queries,
                    "detected_document_types": detected_document_types,
                    "conversation_context": conversation_context,
                    "session_id": session_id
                }

            all_documents = []
            unique_docs = {}  # Use dict to avoid duplicates based on content

            # Retrieve documents for each query across all vector stores
            for query in multi_queries:
                print(f"---RETRIEVING FOR QUERY: {query[:50]}...---")

                # Create combinations of filters
                filter_combinations = []

                if detected_countries and detected_document_types:
                    # Create filters for each country-document_type combination
                    for country in detected_countries:
                        for doc_type in detected_document_types:
                            filter_combinations.append({
                                "country": country,
                                "document_type": doc_type
                            })
                elif detected_countries:
                    # Only country filters
                    for country in detected_countries:
                        filter_combinations.append({"country": country})
                elif detected_document_types:
                    # Only document type filters
                    for doc_type in detected_document_types:
                        filter_combinations.append({"document_type": doc_type})
                else:
                    # No specific filters
                    filter_combinations.append({})

                # Search across all vector stores for each filter combination
                for filter_dict in filter_combinations:
                    if filter_dict:
                        filter_desc = ", ".join(
                            [f"{k}={v}" for k, v in filter_dict.items()])
                        print(f"---APPLYING FILTER: {filter_desc}---")

                    # Search each vector store
                    for store_name, store in self.vector_stores.items():
                        try:
                            print(f"---SEARCHING STORE: {store_name}---")

                            # Use similarity search with metadata filtering if supported
                            if hasattr(store, 'similarity_search') and filter_dict:
                                docs = store.similarity_search(
                                    query,
                                    k=3,
                                    filter=filter_dict
                                )
                            else:
                                # Fallback to regular retrieval if filtering not supported
                                retriever = self.retrievers.get(store_name)
                                if retriever:
                                    docs = retriever.invoke(query)
                                else:
                                    docs = store.similarity_search(query, k=3)

                            # Add store name to metadata for tracking
                            for doc in docs:
                                doc.metadata["vector_store"] = store_name
                                # Use first 100 chars as key
                                doc_key = doc.page_content[:100]
                                if doc_key not in unique_docs:
                                    unique_docs[doc_key] = doc

                            print(
                                f"---FOUND {len(docs)} documents in {store_name}---")

                        except Exception as e:
                            print(
                                f"Filtered retrieval failed for {store_name} with {filter_dict}: {e}")
                            # Fallback to regular retrieval
                            try:
                                retriever = self.retrievers.get(store_name)
                                if retriever:
                                    docs = retriever.invoke(query)
                                    for doc in docs:
                                        doc.metadata["vector_store"] = store_name
                                        doc_key = doc.page_content[:100]
                                        if doc_key not in unique_docs:
                                            unique_docs[doc_key] = doc
                            except Exception as fallback_e:
                                print(
                                    f"Fallback retrieval also failed for {store_name}: {fallback_e}")

            all_documents = list(unique_docs.values())

            # Limit total documents to avoid overwhelming the context
            max_docs = 12  # Increased since we're searching multiple stores
            if len(all_documents) > max_docs:
                all_documents = all_documents[:max_docs]

            print(
                f"---RETRIEVED {len(all_documents)} UNIQUE DOCUMENTS FROM ALL STORES---")
            if detected_document_types:
                print(
                    f"---DOCUMENT TYPES FILTERED: {detected_document_types}---")

            # Log which stores contributed documents
            store_contributions = {}
            for doc in all_documents:
                store = doc.metadata.get("vector_store", "unknown")
                store_contributions[store] = store_contributions.get(
                    store, 0) + 1

            if store_contributions:
                print(f"---STORE CONTRIBUTIONS: {store_contributions}---")

            return {
                "documents": all_documents,
                "question": question,
                "attempt_count": state.get("attempt_count", 0),
                "detected_countries": detected_countries,
                "multi_queries": multi_queries,
                "detected_document_types": detected_document_types,
                "conversation_context": conversation_context,
                "session_id": session_id
            }

        def generate(state):
            """Generate answer using RAG chain with conversation context."""
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]
            detected_countries = state.get("detected_countries", [])
            multi_queries = state.get("multi_queries", [])
            detected_document_types = state.get("detected_document_types", [])
            conversation_context = state.get("conversation_context", "")
            session_id = state.get("session_id", "")

            if isinstance(documents, list) and documents:
                context = "\n\n".join([doc.page_content for doc in documents])
            elif hasattr(documents, 'page_content'):
                context = documents.page_content
            else:
                context = str(documents)

            # Include conversation context in generation
            generation = self.rag_chain.invoke({
                "context": context,
                "question": question,
                "conversation_context": conversation_context or "No previous conversation context."
            })

            return {
                "documents": documents,
                "question": question,
                "generation": generation,
                "attempt_count": state.get("attempt_count", 0),
                "detected_countries": detected_countries,
                "multi_queries": multi_queries,
                "detected_document_types": detected_document_types,
                "conversation_context": conversation_context,
                "session_id": session_id
            }

        def grade_documents(state):
            """Grade relevance of retrieved documents."""
            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]
            detected_countries = state.get("detected_countries", [])
            multi_queries = state.get("multi_queries", [])
            detected_document_types = state.get("detected_document_types", [])

            filtered_docs = []
            for d in documents:
                score = self.retrieval_grader.invoke({
                    "question": question,
                    "document": d.page_content
                })
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")

            return {
                "documents": filtered_docs,
                "question": question,
                "attempt_count": state.get("attempt_count", 0),
                "detected_countries": detected_countries,
                "multi_queries": multi_queries,
                "detected_document_types": detected_document_types
            }

        def transform_query(state):
            """Transform the query to produce better questions with multi-query approach."""
            print("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]
            detected_countries = state.get("detected_countries", [])
            detected_document_types = state.get("detected_document_types", [])

            # Generate new multi-queries for better retrieval
            new_multi_queries = self._generate_multi_queries(question)

            # Use the first transformed query as the primary question
            better_question = new_multi_queries[0] if new_multi_queries else question

            return {
                "documents": documents,
                "question": better_question,
                "attempt_count": state.get("attempt_count", 0) + 1,
                "detected_countries": detected_countries,
                "multi_queries": new_multi_queries,
                "detected_document_types": detected_document_types
            }

        def web_search(state):
            """Web search based on the question."""
            print("---WEB SEARCH---")
            question = state["question"]
            detected_countries = state.get("detected_countries", [])
            multi_queries = state.get("multi_queries", [question])
            detected_document_types = state.get("detected_document_types", [])

            if not self.web_search_tool:
                # Fallback if web search is not available
                fallback_doc = Document(
                    page_content="Web search is not available. Please provide more specific policy-related queries."
                )
                return {
                    "documents": [fallback_doc],
                    "question": question,
                    "attempt_count": state.get("attempt_count", 0),
                    "detected_countries": detected_countries,
                    "multi_queries": multi_queries,
                    "detected_document_types": detected_document_types
                }

            # Use the first multi-query for web search (usually the most comprehensive)
            search_query = multi_queries[0] if multi_queries else question

            # If countries are detected, include them in the search
            if detected_countries:
                search_query = f"{search_query} {' '.join(detected_countries)}"

            # If document types are detected, include relevant terms
            if detected_document_types:
                type_terms = []
                for doc_type in detected_document_types:
                    if doc_type == DocumentType.DIGITAL_ID_POLICY.value:
                        type_terms.append("digital identity")
                    elif doc_type == DocumentType.CROSS_BORDER_POLICY.value:
                        type_terms.append("cross-border")
                    elif doc_type == DocumentType.DATA_PROTECTION_POLICY.value:
                        type_terms.append("data protection")

                if type_terms:
                    search_query = f"{search_query} {' '.join(type_terms)}"

            docs = self.web_search_tool.invoke({"query": search_query})
            
            # Handle different response formats from TavilySearch
            try:
                if isinstance(docs, dict):
                    if "results" in docs:
                        results = docs["results"]
                        web_results = "\n".join([d.get("content", "") for d in results if isinstance(d, dict)])
                    elif "error" in docs:
                        web_results = f"Web search error: {docs['error']}"
                    else:
                        web_results = str(docs)
                elif isinstance(docs, list):
                    web_results_list = []
                    for d in docs:
                        if isinstance(d, dict):
                            # Try different possible content keys
                            content = d.get("content") or d.get("snippet") or d.get("text") or str(d)
                        else:
                            content = str(d)
                        web_results_list.append(content)
                    web_results = "\n".join(web_results_list)
                else:
                    # Fallback for other formats
                    web_results = str(docs)
            except Exception as e:
                web_results = f"Error processing web search results: {e}. Raw response: {docs}"
            
            web_results_doc = Document(page_content=web_results)

            return {
                "documents": [web_results_doc],
                "question": question,
                "attempt_count": state.get("attempt_count", 0),
                "detected_countries": detected_countries,
                "multi_queries": multi_queries,
                "detected_document_types": detected_document_types
            }

        # Edge functions
        def route_question(state):
            """Route question to web search or RAG."""
            print("---ROUTE QUESTION---")
            question = state["question"]

            if not self.vector_stores:
                print("---ROUTE QUESTION TO WEB SEARCH (NO VECTOR STORES)---")
                return "web_search"

            source = self.question_router.invoke({"question": question})
            if source.datasource == "web_search":
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif source.datasource == "vectorstore":
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"

        def decide_to_generate(state):
            """Decide whether to generate an answer or re-generate a question."""
            print("---ASSESS GRADED DOCUMENTS---")
            filtered_documents = state["documents"]
            attempt_count = state.get("attempt_count", 0)

            if not filtered_documents:
                if attempt_count < self.max_attempts:
                    print("---DECISION: ALL DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
                    return "transform_query"
                else:
                    print(
                        "---DECISION: MAX ATTEMPTS REACHED, GENERATE WITH EMPTY CONTEXT---")
                    return "generate"
            else:
                print("---DECISION: GENERATE---")
                return "generate"

        def grade_generation_v_documents_and_question(state):
            """Grade generation for hallucinations and question answering."""
            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]
            attempt_count = state.get("attempt_count", 0)

            score = self.hallucination_grader.invoke({
                "documents": documents,
                "generation": generation
            })
            grade = score.binary_score

            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                print("---GRADE GENERATION vs QUESTION---")
                score = self.answer_grader.invoke({
                    "question": question,
                    "generation": generation
                })
                grade = score.binary_score
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    if attempt_count < self.max_attempts:
                        return "not useful"
                    else:
                        print("---MAX ATTEMPTS REACHED, ACCEPTING ANSWER---")
                        return "useful"
            else:
                print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
                if attempt_count < self.max_attempts:
                    return "not supported"
                else:
                    print("---MAX ATTEMPTS REACHED, ACCEPTING ANSWER---")
                    return "useful"

        # Build the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("web_search", web_search)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("transform_query", transform_query)

        # Add edges
        workflow.add_conditional_edges(
            START,
            route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )

        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        workflow.add_edge("transform_query", "retrieve")

        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile the graph
        self.app = workflow.compile()

    def query(self, question: str) -> str:
        """
        Process a query through the adaptive RAG system with country filtering, document type filtering, and multi-query.

        Args:
            question: The user's question

        Returns:
            The generated answer
        """
        # Analyze the query
        print("---ANALYZING QUERY---")

        # Detect countries in the query
        detected_countries = self._detect_countries_in_query(question)
        if detected_countries:
            print(f"---DETECTED COUNTRIES: {detected_countries}---")

        # Detect document types in the query
        detected_document_types = self._detect_document_types_in_query(
            question)
        if detected_document_types:
            print(f"---DETECTED DOCUMENT TYPES: {detected_document_types}---")

        # Generate multiple query variations
        multi_queries = self._generate_multi_queries(question)
        print(f"---GENERATED {len(multi_queries)} QUERY VARIATIONS---")

        inputs = {
            "question": question,
            "attempt_count": 0,
            "detected_countries": detected_countries,
            "multi_queries": multi_queries,
            "detected_document_types": detected_document_types
        }

        result = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                result = value

        return result.get("generation", "I couldn't generate an answer to your question.")

    def process_query_input(self, query_input: QueryInput) -> QueryOutput:
        """
        Process a QueryInput object and return a QueryOutput with conversation history support.

        Args:
            query_input: QueryInput object containing the query and metadata

        Returns:
            QueryOutput object with the response and metadata
        """
        response = self.query_with_history(
            question=query_input.query,
            session_id=query_input.session_id,
            model_provider=query_input.model_provider,
            model_id=query_input.model_id,
            save_to_history=True
        )

        return QueryOutput(
            response=response,
            session_id=query_input.session_id,
            model_provider=query_input.model_provider,
            model_id=query_input.model_id
        )

    def query_with_history(
        self,
        question: str,
        session_id: str,
        model_provider: str = "openai",
        model_id: str = None,
        save_to_history: bool = True
    ) -> str:
        """
        Process a query through the adaptive RAG system with conversation history support.

        Args:
            question: The user's question
            session_id: Session identifier for conversation history
            model_provider: Model provider identifier
            model_id: Model identifier
            save_to_history: Whether to save this interaction to history

        Returns:
            The generated answer
        """
        if model_id is None:
            model_id = settings.openai_model
        print("---LOADING CONVERSATION CONTEXT---")
        conversation_context = self._get_conversation_context(session_id)
        if conversation_context:
            print(
                f"---CONVERSATION CONTEXT LOADED: {len(conversation_context)} characters---")

        print("---ANALYZING QUERY---")

        detected_countries = self._detect_countries_in_query(question)
        if detected_countries:
            print(f"---DETECTED COUNTRIES: {detected_countries}---")

        detected_document_types = self._detect_document_types_in_query(
            question)
        if detected_document_types:
            print(f"---DETECTED DOCUMENT TYPES: {detected_document_types}---")

        multi_queries = self._generate_multi_queries(question)
        print(f"---GENERATED {len(multi_queries)} QUERY VARIATIONS---")

        inputs = {
            "question": question,
            "attempt_count": 0,
            "detected_countries": detected_countries,
            "multi_queries": multi_queries,
            "detected_document_types": detected_document_types,
            "conversation_context": conversation_context,
            "session_id": session_id
        }

        result = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                result = value

        response = result.get(
            "generation", "I couldn't generate an answer to your question.")

        if save_to_history and session_id:
            try:
                insert_chat_history(session_id, question,
                                    response, model_provider, model_id)
                self._cleanup_old_conversation_data(session_id)

            except Exception as e:
                print(f"Failed to save conversation history: {e}")

        return response

    def analyze_query(self, question: str) -> dict:
        """
        Analyze a query to show detected countries, document types, and generated variations.

        Args:
            question: The user's question

        Returns:
            Dictionary containing analysis results
        """
        detected_countries = self._detect_countries_in_query(question)
        detected_document_types = self._detect_document_types_in_query(
            question)
        multi_queries = self._generate_multi_queries(question)

        return {
            "original_query": question,
            "detected_countries": detected_countries,
            "detected_document_types": detected_document_types,
            "multi_queries": multi_queries,
            "country_filter_applicable": len(detected_countries) > 0,
            "document_type_filter_applicable": len(detected_document_types) > 0,
            "filter_strategy": self._get_filter_strategy(detected_countries, detected_document_types)
        }

    def _get_filter_strategy(self, countries: List[str], doc_types: List[str]) -> str:
        """
        Determine the filtering strategy based on detected countries and document types.

        Args:
            countries: List of detected countries
            doc_types: List of detected document types

        Returns:
            String describing the filter strategy
        """
        if countries and doc_types:
            return f"Combined filtering: {len(countries)} countries × {len(doc_types)} document types"
        elif countries:
            return f"Country filtering: {len(countries)} countries"
        elif doc_types:
            return f"Document type filtering: {len(doc_types)} types"
        else:
            return "No specific filtering - broad search"

    def _detect_document_types_in_query(self, query: str) -> List[str]:
        """
        Detect relevant document types based on keywords and context in the query.

        Args:
            query: The user query

        Returns:
            List of relevant document type strings
        """
        detected_types = []
        query_lower = query.lower()

        document_type_keywords = {
            DocumentType.DIGITAL_ID_POLICY: [
                'digital id', 'digital identity', 'e-id', 'electronic identity',
                'biometric', 'identity verification', 'digital identification',
                'id system', 'identity management', 'authentication',
                'identity card', 'digital citizenship', 'e-identity',
                'identity framework', 'digital credential', 'identity platform'
            ],
            DocumentType.CROSS_BORDER_POLICY: [
                'cross border', 'cross-border', 'transborder', 'international',
                'regional', 'multilateral', 'bilateral', 'trade agreement',
                'border control', 'migration', 'customs', 'transit',
                'cross boundary', 'inter-country', 'regional integration',
                'free movement', 'trade facilitation', 'border management'
            ],
            DocumentType.DATA_PROTECTION_POLICY: [
                'data protection', 'privacy', 'personal data', 'data privacy',
                'gdpr', 'data security', 'information protection',
                'data governance', 'consent', 'data subject', 'data controller',
                'data processing', 'privacy rights', 'data breach',
                'data retention', 'anonymization', 'pseudonymization'
            ]
        }

        for doc_type, keywords in document_type_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if doc_type.value not in detected_types:
                        detected_types.append(doc_type.value)
                    break

        try:
            doc_type_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at identifying policy document types relevant to user queries.
                
                Analyze the query and determine which of these document types are relevant:
                - cross_border_policy: policies about international/regional cooperation, trade, migration, border management
                - digital_id_policy: policies about digital identity, e-ID systems, biometric authentication, identity verification
                - data_protection_policy: policies about privacy, data protection, GDPR compliance, personal data handling
                
                A query might be relevant to multiple document types. Return all relevant types.
                If none are clearly relevant, return an empty list.
                
                Examples:
                - "digital identity in Kenya" → ["digital_id_policy"]
                - "cross-border data sharing" → ["cross_border_policy", "data_protection_policy"]
                - "GDPR compliance" → ["data_protection_policy"]
                - "regional trade agreements" → ["cross_border_policy"]"""),
                ("human", "Query: {query}")
            ])

            structured_doc_type_detector = self.llm.with_structured_output(
                DocumentTypeDetection)
            doc_type_chain = doc_type_prompt | structured_doc_type_detector

            result = doc_type_chain.invoke({"query": query})

            for doc_type in result.document_types:
                if doc_type not in detected_types:
                    detected_types.append(doc_type)

        except Exception as e:
            print(f"LLM document type detection failed: {e}")
        if not detected_types:
            detected_types = [
                DocumentType.DIGITAL_ID_POLICY.value,
                DocumentType.CROSS_BORDER_POLICY.value,
                DocumentType.DATA_PROTECTION_POLICY.value
            ]
            print("---NO SPECIFIC DOCUMENT TYPES DETECTED, USING ALL TYPES---")

        return detected_types

    def _summarize_conversation_history(self, messages: List[Dict]) -> str:
        """
        Summarize conversation history to maintain context while reducing token usage.

        Args:
            messages: List of conversation messages

        Returns:
            Summarized conversation context
        """
        if not messages:
            return ""

        try:
            conversation_text = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_text += f"{role}: {msg['content']}\n\n"

            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at summarizing policy-related conversations.
                
                Summarize the conversation below, focusing on:
                1. Key policy topics discussed (digital identity, cross-border policies, data protection)
                2. Specific countries or regions mentioned
                3. Important conclusions or decisions reached
                4. Current context that would be relevant for continuing the conversation
                
                Keep the summary concise but comprehensive, maintaining important details that would help understand future questions in this conversation.
                """),
                ("human", "Conversation to summarize:\n\n{conversation}")
            ])

            structured_summarizer = self.llm.with_structured_output(
                ConversationSummary)
            summary_chain = summary_prompt | structured_summarizer

            result = summary_chain.invoke({"conversation": conversation_text})
            return result.summary

        except Exception as e:
            print(f"Conversation summarization failed: {e}")
            topics = set()
            for msg in messages:
                if msg["role"] == "user":
                    # Extract key terms
                    content_lower = msg["content"].lower()
                    if any(term in content_lower for term in ["digital", "identity", "id"]):
                        topics.add("digital identity")
                    if any(term in content_lower for term in ["cross", "border", "trade"]):
                        topics.add("cross-border policies")
                    if any(term in content_lower for term in ["data", "privacy", "protection"]):
                        topics.add("data protection")

            topic_str = ", ".join(topics) if topics else "policy discussions"
            return f"Previous conversation covered {topic_str} with {len(messages)//2} exchanges."

    def _get_conversation_context(self, session_id: str, max_message_pairs: int = 10) -> str:
        """
        Get conversation context, summarizing older messages if conversation is long.

        Args:
            session_id: Session identifier
            max_message_pairs: Maximum number of recent message pairs to keep in full

        Returns:
            Conversation context string
        """
        if not session_id:
            return ""

        total_message_count = count_conversation_messages(session_id)

        if total_message_count == 0:
            return ""

        if total_message_count <= max_message_pairs:
            messages = get_recent_messages(session_id, total_message_count)
            return self._format_messages_for_context(messages)
        else:
            recent_messages = get_recent_messages(
                session_id, max_message_pairs // 2)

            older_message_count = total_message_count - \
                (max_message_pairs // 2)
            if older_message_count > 0:
                sample_size = min(older_message_count, 20)
                all_messages = get_conversation_messages(
                    session_id, sample_size + (max_message_pairs // 2))
                older_messages = all_messages[:-len(
                    recent_messages)] if recent_messages else all_messages

                summary = self._summarize_conversation_history(older_messages)
                recent_context = self._format_messages_for_context(
                    recent_messages)

                return f"Previous conversation summary: {summary}\n\nRecent messages:\n{recent_context}"
            else:
                return self._format_messages_for_context(recent_messages)

    def _format_messages_for_context(self, messages: List[Dict]) -> str:
        """Format messages for inclusion in conversation context."""
        if not messages:
            return ""

        formatted = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")

        return "\n".join(formatted)

    def _cleanup_old_conversation_data(self, session_id: str, keep_recent: int = 50):
        """
        Clean up old conversation data to prevent database bloat.

        Args:
            session_id: Session identifier
            keep_recent: Number of recent message pairs to keep in database
        """
        try:
            total_count = count_conversation_messages(session_id)
            if total_count > keep_recent * 2:
                print(
                    f"Cleaning up old conversation data for session {session_id}")
                delete_old_messages(session_id, keep_recent)
        except Exception as e:
            print(f"Failed to cleanup old conversation data: {e}")

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of message pairs to retrieve

        Returns:
            List of conversation messages
        """
        return get_recent_messages(session_id, limit)

    def clear_conversation_history(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        try:
            delete_old_messages(session_id, 0)  # Keep 0 messages = delete all
            print(f"Cleared conversation history for session: {session_id}")
        except Exception as e:
            print(f"Failed to clear conversation history: {e}")

    def get_conversation_summary(self, session_id: str) -> str:
        """
        Get a summary of the conversation history.

        Args:
            session_id: Session identifier

        Returns:
            Conversation summary
        """
        messages = get_conversation_messages(
            session_id, 20)  # Get up to 20 message pairs
        if not messages:
            return "No conversation history available."

        return self._summarize_conversation_history(messages)

    def count_conversation_messages(self, session_id: str) -> int:
        """
        Count the number of messages in a conversation session.

        Args:
            session_id: Session identifier

        Returns:
            Number of messages in the session
        """
        return count_conversation_messages(session_id)

    def get_vector_store_info(self) -> Dict[str, Dict]:
        """
        Get information about available vector stores.

        Returns:
            Dictionary with information about each vector store
        """
        info = {}
        for store_name, store in self.vector_stores.items():
            info[store_name] = {
                "type": type(store).__name__,
                "has_retriever": store_name in self.retrievers,
                "available": True
            }
        return info

    def search_specific_vector_store(
        self,
        store_name: str,
        query: str,
        country_filter: Optional[str] = None,
        document_type_filter: Optional[str] = None,
        k: int = 5
    ) -> List[Document]:
        """
        Search a specific vector store by name.

        Args:
            store_name: Name of the vector store to search
            query: Search query
            country_filter: Filter by country name
            document_type_filter: Filter by document type
            k: Number of results to return

        Returns:
            List of documents from the specified store
        """
        if store_name not in self.vector_stores:
            print(f"Vector store '{store_name}' not found")
            return []

        store = self.vector_stores[store_name]

        # Prepare filters
        filters = {}
        if country_filter:
            # Use the same country normalization logic
            normalized_country = None
            for country, variations in AFRICAN_COUNTRIES.items():
                if country_filter.lower() in [v.lower() for v in variations]:
                    normalized_country = country
                    break

            if not normalized_country:
                from fuzzywuzzy import process
                match = process.extractOne(country_filter, list(
                    AFRICAN_COUNTRIES.keys()), scorer=fuzz.ratio)
                if match and match[1] >= 70:
                    normalized_country = match[0]

            if normalized_country:
                filters["country"] = normalized_country

        if document_type_filter:
            filters["document_type"] = document_type_filter

        try:
            if filters and hasattr(store, 'similarity_search'):
                results = store.similarity_search(
                    query=query,
                    k=k,
                    filter=filters
                )
            else:
                results = store.similarity_search(query=query, k=k)

            # Add store name to metadata
            for doc in results:
                doc.metadata["vector_store"] = store_name

            return results
        except Exception as e:
            print(f"Error searching vector store {store_name}: {e}")
            return []

    def demonstrate_multi_store_search(self, query: str) -> Dict:
        """
        Demonstrate searching across multiple vector stores and return detailed results.

        Args:
            query: Search query

        Returns:
            Dictionary with results from each store and analysis
        """
        results = {
            "query": query,
            "vector_stores_info": self.get_vector_store_info(),
            "store_results": {},
            "combined_results": [],
            "analysis": {}
        }

        # Search each store individually
        for store_name in self.vector_stores.keys():
            print(f"---SEARCHING {store_name}---")
            store_results = self.search_specific_vector_store(
                store_name=store_name,
                query=query,
                k=3
            )

            results["store_results"][store_name] = {
                "count": len(store_results),
                "documents": [
                    {
                        "content_preview": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    } for doc in store_results
                ]
            }

            results["combined_results"].extend(store_results)

        # Analysis
        results["analysis"] = {
            "total_documents_found": len(results["combined_results"]),
            "stores_with_results": len([s for s in results["store_results"].values() if s["count"] > 0]),
            "store_contributions": {
                store: data["count"]
                for store, data in results["store_results"].items()
            }
        }

        return results
