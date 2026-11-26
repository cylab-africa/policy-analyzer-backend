import sqlite3
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class CoTRAGSystem:
    def __init__(self, llm, streaming_llm, embedding_model, retriever, history_store, session_id,
                 supported_tabs,
                 enable_web_search=False, search_tool_type="tavily", max_docs_per_query=2, max_total_docs=4,
                 is_comparative=False, detected_countries=None):

        self.llm = llm
        self.embeddings = embedding_model
        self.retriever = retriever
        self.streaming_llm = streaming_llm

        self.history_store = history_store
        self.session_id = session_id
        self.enable_web_search = enable_web_search
        self.search_tool_type = search_tool_type
        self.supported_tabs = supported_tabs

        # Configurable document retrieval limits
        self.max_docs_per_query = max_docs_per_query  # Docs per sub-query
        self.max_total_docs = max_total_docs  # Total unique docs to return

        # Comparative query metadata
        self.is_comparative = is_comparative
        self.detected_countries = detected_countries or []

        self._analysis_cache = {}
        self._context_cache = {}

    def _get_query_hash(self, query: str) -> str:
        """Create a simple hash for caching"""
        import hashlib
        return hashlib.md5(query.lower().strip().encode(), usedforsecurity=False).hexdigest()[:16]

    def analyze_and_decompose_query(self, user_query: str) -> Dict:
        """Use CoT to analyze query relevance and decompose into sub-queries if needed"""

        query_hash = self._get_query_hash(user_query)
        if query_hash in self._analysis_cache:

            return self._analysis_cache[query_hash]

        history = self.get_session_history_text()
        has_history = "No previous conversation." not in history
        with open("app/data/base_cot.txt", "r") as f:
            base_cot = f.read()

        cot_prompt = ChatPromptTemplate.from_template(
            base_cot
        )

        chain = cot_prompt | self.llm | StrOutputParser()

        history_context = ""
        if has_history:
            history_context = f"""
            CONVERSATION HISTORY:
            {history}
            
            IMPORTANT: If the user query seems like a follow-up question (e.g., "tell me more", "explain further", 
            "go more detailed", "what about X"), consider it RELEVANT to our scope because it refers to the previous 
            conversation which was about our topics. Use the conversation history to determine the actual topic.
            """

        response = chain.invoke({
            "query": user_query,
            "history_context": history_context
        })

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
            else:
                raise json.JSONDecodeError(
                    "No JSON found in response", response, 0)

            follow_up_indicators = ["more detail", "tell me more", "explain further", "elaborate",
                                    "go deeper", "continue", "go on", "what about", "based on your last answer"]

            is_likely_follow_up = any(indicator in user_query.lower()
                                      for indicator in follow_up_indicators)

            if has_history and is_likely_follow_up and not analysis.get('is_relevant', False):
                analysis['is_relevant'] = True
                analysis['is_direct'] = False
                analysis['reasoning'] = "Query appears to be a follow-up to previous conversation about our topics"
                analysis['main_intent'] = "Get more detailed information about the previously discussed topic"
                analysis['search_strategy'] = "multi_query_search"
                analysis['is_follow_up'] = True

                last_exchanges = history.split(
                    '\n')[-2:] if '\n' in history else [history]
                topic_indicators = [word for exchange in last_exchanges
                                    for word in exchange.split()
                                    if word.lower() not in {"the", "a", "an", "and", "or", "but", "in", "on", "at"}]

                topic_terms = " ".join(topic_indicators[:10])
                analysis['sub_queries'] = [
                    f"detailed explanation {topic_terms}",
                    f"advanced concepts {topic_terms}",
                    f"specific aspects {topic_terms}"
                ]

            if analysis.get('is_direct', False) and analysis.get('is_relevant', False):
                analysis['sub_queries'] = [user_query]
                analysis['search_strategy'] = 'direct_search'

            self._analysis_cache[query_hash] = analysis

            return analysis
        except json.JSONDecodeError:
            print(" Failed to parse CoT analysis response. Returning default structure.")
            default_analysis = {
                "is_relevant": False,
                "is_direct": False,
                "reasoning": "Failed to analyze query",
                "main_intent": None,
                "sub_queries": [],
                "search_strategy": "basic"
            }
            self._analysis_cache[query_hash] = default_analysis
            return default_analysis

    def retrieve_context_with_subqueries(self, sub_queries: List[str], is_direct: bool = False) -> str:
        if not self.retriever or not sub_queries:
            return "No relevant context available."

        if is_direct:
            docs = self.retriever.invoke(sub_queries[0])
            return "\n\n".join([doc.page_content for doc in docs[:5]])
        else:
            all_docs = []
            for query in sub_queries:
                docs = self.retriever.invoke(query)
                all_docs.extend(docs)
            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            print(
                f"Retrieved {len(unique_docs)} unique documents for {len(sub_queries)} sub-queries.")
            vector_context = "\n\n".join(
                [doc.page_content for doc in unique_docs[:8]])
            if self.enable_web_search and len(unique_docs) < 3:
                web_context = self.perform_web_search(sub_queries[0])
                if web_context and "Error" not in web_context:
                    combined_context = vector_context + "\n\n" + web_context
                    return combined_context

            return vector_context

    def perform_web_search(self, query: str) -> str:
        """
        Perform a web search for additional context

        Args:
            query: Search query

        Returns:
            str: Search results as text
        """
        try:
            from app.services.search_tools import SearchToolFactory

            search_start = time.time()

            search_results = SearchToolFactory.perform_search(
                query=query,
                tool_type=self.search_tool_type,
                max_results=3
            )

            search_end = time.time()
            # print(
            #     f"🕐 [SEARCH] Web search completed in {search_end - search_start:.2f}s")

            return search_results

        except Exception as e:
            return f"Error during web search: {str(e)}"

    async def generate_response_stream(self, user_query: str, analysis: Dict, context: str, history: str):
        stream_start = time.time()
        supported_tabs_str = " and ".join(self.supported_tabs)

        if not analysis["is_relevant"]:
            response = f"""I specialize in {supported_tabs_str} topics. Your question appears to be about: {analysis['reasoning']}
            The information you're looking for is not available in my knowledge base, which focuses specifically on:
            - {supported_tabs_str}
            Please ask questions related to these topics for the most helpful responses."""
            stream_time = time.time() - stream_start
            yield response
            return

        print("Generating response with context and history...")
        prompt_start = time.time()

        # Build formatting instructions based on query type
        formatting_instructions = ""
        if self.is_comparative and len(self.detected_countries) >= 2:
            countries_str = ", ".join(self.detected_countries)
            formatting_instructions = f"""

            IMPORTANT FORMATTING INSTRUCTION:
            This is a COMPARATIVE query comparing: {countries_str}

            Please format your response as a MARKDOWN TABLE with the following structure:
            - First column: Aspect/Feature being compared
            - Subsequent columns: One column per country ({countries_str})
            - Include key aspects like: Legal Framework, Year Enacted, Key Provisions, Enforcement, etc.
            - Add a brief summary paragraph after the table if needed

            Example format:
            | Aspect | {self.detected_countries[0]} | {self.detected_countries[1]} |
            |--------|---------|---------|
            | Legal Framework | ... | ... |
            | Year Enacted | ... | ... |
            """
        else:
            formatting_instructions = ""

        response_prompt = ChatPromptTemplate.from_template("""
            You are an expert assistant specializing in {supported_tabs_str}.
            Note: Please ensure that your responses are relevant to the African context and regulations.
            Provide concise and accurate information.
            Do not over-explain or provide unnecessary details.

            ANALYSIS OF USER'S QUESTION:
            - Main Intent: {main_intent}
            - Search Strategy: {search_strategy}
            - Sub-queries explored: {sub_queries}

            RELEVANT CONTEXT:
            {context}

            CONVERSATION HISTORY:
            {history}

            USER'S ORIGINAL QUESTION: {query}
            {formatting_instructions}

            Using the analysis and context above, provide a comprehensive answer that:
            1. Directly addresses the user's main intent
            2. Uses specific information from the context when available
            3. Maintains focus on data protection and digital identity topics
            4. References previous conversation if relevant
            5. FOLLOW THE FORMATTING INSTRUCTIONS ABOVE if provided

            Answer:""")

        chain = response_prompt | self.streaming_llm | StrOutputParser()
        prompt_end = time.time()
        streaming_start = time.time()
        chunk_count = 0
        first_chunk_time = None
        supported_tabs_str = "and ".join(self.supported_tabs)

        async for chunk in chain.astream({
            "query": user_query,
            "main_intent": analysis["main_intent"],
            "search_strategy": analysis["search_strategy"],
            "sub_queries": ", ".join(analysis["sub_queries"]),
            "context": context,
            "history": history,
            "supported_tabs_str": supported_tabs_str,
            "formatting_instructions": formatting_instructions
        }):
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time()
            yield chunk

        total_generate_time = time.time() - stream_start
        actual_streaming_time = time.time() - streaming_start
        # print(f" [GENERATE] Response generation completed:")
        # print(f"  - Prompt setup: {prompt_end - prompt_start:.2f}s")
        # print(f"  - LLM streaming: {actual_streaming_time:.2f}s")
        # print(f"  - Total generation time: {total_generate_time:.2f}s")
        # print(f"  - Chunks processed: {chunk_count}")

    def generate_response(self, user_query: str, analysis: Dict, context: str, history: str) -> str:
        """Generate final response using CoT analysis and retrieved context"""

        if not analysis["is_relevant"]:
            return f"""I specialize in {self.supported_tabs_str} topics. Your question appears to be about: {analysis['reasoning']}
                    Please ask questions related to these topics for the most helpful responses."""

            response_prompt = ChatPromptTemplate.from_template("""
                    You are an expert assistant specializing in data protection and digital identity.

                    ANALYSIS OF USER'S QUESTION:
                    - Main Intent: {main_intent}
                    - Search Strategy: {search_strategy}
                    - Sub-queries explored: {sub_queries}

                    RELEVANT CONTEXT:
                    {context}

                    CONVERSATION HISTORY:
                    {history}

                    USER'S ORIGINAL QUESTION: {query}

                    Using the analysis and context above, provide a comprehensive answer that:
                    1. Directly addresses the user's main intent
                    2. Uses specific information from the context when available
                    3. Maintains focus on data protection and digital identity topics
                    4. References previous conversation if relevant

                    Answer:""")

        chain = response_prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "query": user_query,
            "main_intent": analysis["main_intent"],
            "search_strategy": analysis["search_strategy"],
            "sub_queries": ", ".join(analysis["sub_queries"]),
            "context": context,
            "history": history
        })

    def get_session_history_text(self) -> str:
        """Get formatted session history"""
        recent_messages = self.history_store.get_session_history(
            self.session_id, limit=8)
        if not recent_messages:
            return "No previous conversation."

        formatted = []
        for msg in recent_messages[-6:]:  # Last 6 messages for context
            role = "Human" if msg["role"] == "human" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def process_query(self, user_query: str) -> str:
        """Main CoT processing pipeline"""
        print(" Analyzing query...")

        # Step 1: Analyze and decompose query
        analysis = self.analyze_and_decompose_query(user_query)

        print(
            f" Relevance: {' Relevant' if analysis['is_relevant'] else ' Not Relevant'}")
        if analysis['is_relevant']:
            if analysis.get('is_direct', False):
                print("Direct query - using original query for retrieval")
            else:
                print(
                    f" Complex query - decomposing into {len(analysis['sub_queries'])} sub-queries")
            print(f" Intent: {analysis['main_intent']}")

        # Step 2: Retrieve context if relevant
        if analysis["is_relevant"] and analysis["sub_queries"]:
            print(" Retrieving context...")
            context = self.retrieve_context_with_subqueries(
                analysis["sub_queries"],
                analysis.get('is_direct', False)
            )
        else:
            context = ""

        # Step 3: Get session history
        history = self.get_session_history_text()

        # Step 4: Generate response
        print(" Generating response...")
        response = self.generate_response(
            user_query, analysis, context, history)

        return response

    async def process_query_non_streaming(self, user_query: str) -> str:
        """Non-streaming version of the query processing pipeline"""
        print(
            f" [NON-STREAMING] Starting process_query_non_streaming at {datetime.now()}")

        analysis = self.analyze_and_decompose_query(user_query)
        print(
            f" Relevance: {' Relevant' if analysis['is_relevant'] else ' Not Relevant'}")
        if analysis["is_relevant"] and analysis["sub_queries"]:
            context = self.retrieve_context_optimized(
                analysis["sub_queries"],
                analysis.get('is_direct', False)
            )
        else:
            context = ""

        # Step 3: Get session history
        history = self.get_session_history_text()

        # Step 4: Generate response (non-streaming)
        if not analysis["is_relevant"]:
            supported_tabs_str = " and ".join(self.supported_tabs)
            return f"""I specialize in {supported_tabs_str} topics. Your question appears to be about: {analysis['reasoning']}
            
            The information you're looking for is not available in my knowledge base, which focuses specifically on:
            - {supported_tabs_str}

            Please ask questions related to these topics for the most helpful responses."""

        response_prompt = ChatPromptTemplate.from_template("""
            You are an expert assistant specializing in {supported_tabs_str}.

            ANALYSIS OF USER'S QUESTION:
            - Main Intent: {main_intent}
            - Search Strategy: {search_strategy}
            - Sub-queries explored: {sub_queries}

            RELEVANT CONTEXT:
            {context}

            CONVERSATION HISTORY:
            {history}

            USER'S ORIGINAL QUESTION: {query}

            Using the analysis and context above, provide a comprehensive answer that:
            1. Directly addresses the user's main intent
            2. Uses specific information from the context when available
            3. Maintains focus on data protection and digital identity topics
            4. References previous conversation if relevant

            Answer:""")

        chain = response_prompt | self.llm | StrOutputParser()
        supported_tabs_str = " and ".join(self.supported_tabs)

        response = chain.invoke({
            "query": user_query,
            "main_intent": analysis["main_intent"],
            "search_strategy": analysis["search_strategy"],
            "sub_queries": ", ".join(analysis["sub_queries"]),
            "context": context,
            "history": history,
            "supported_tabs_str": supported_tabs_str
        })

        return response

    async def process_query_stream_optimized(self, user_query: str):
        """Optimized CoT processing pipeline with parallel execution and faster context retrieval"""
        start_time = time.time()
        print(
            f"[OPTIMIZED] Starting optimized streaming at {datetime.now()}")

        import asyncio
        loop = asyncio.get_event_loop()

        # Step 1: Start all operations in parallel
        print("Starting parallel operations...")

        # Analysis and history can run in parallel
        analysis_future = loop.run_in_executor(
            None, self.analyze_and_decompose_query, user_query)
        history_future = loop.run_in_executor(
            None, self.get_session_history_text)

        # Wait for analysis first (needed for context retrieval decision)
        analysis_start = time.time()
        analysis = await analysis_future
        analysis_end = time.time()
        print(
            f" [OPTIMIZED] Analysis completed in {analysis_end - analysis_start:.2f}s")

        print(
            f" Relevance: {' Relevant' if analysis['is_relevant'] else ' Not Relevant'}")

        # Send metadata first (session_id and other info)
        metadata = {
            "session_id": self.session_id,
            "is_follow_up": analysis.get('is_follow_up', False),
            "is_relevant": analysis.get('is_relevant', False),
            "type": "metadata"
        }
        yield metadata  # Changed to return the dict directly instead of JSON string

        # Step 2: For non-relevant queries, start streaming immediately
        if not analysis["is_relevant"]:
            setup_time = time.time() - start_time
            print(
                f"[OPTIMIZED] Non-relevant query, immediate response in {setup_time:.2f}s")
            response = f"""I specialize in data protection and digital identity topics. Your question appears to be about: {analysis['reasoning']}

                    The information you're looking for is not available in my knowledge base, which focuses specifically on:
                    - Data protection and privacy regulations (GDPR, etc.)
                    - Digital identity and authentication systems  
                    - Personal data governance and security
                    - Consent management and user rights

                    Please ask questions related to these topics for the most helpful responses."""
            content_packet = {
                "type": "content",
                "content": response
            }
            yield content_packet
            return

        if analysis.get('is_direct', False):
            print(" Direct query - optimized retrieval")
        else:
            print(
                f" Complex query - {len(analysis['sub_queries'])} sub-queries")
        print(f" Intent: {analysis['main_intent']}")

        context_start = time.time()
        if analysis["sub_queries"]:
            context_future = loop.run_in_executor(
                None,
                self.retrieve_context_optimized,
                analysis["sub_queries"],
                analysis.get('is_direct', False)
            )
        else:
            context_future = asyncio.create_task(self._create_empty_context())
        context, history = await asyncio.gather(context_future, history_future)

        context_end = time.time()
        setup_time = context_end - start_time
        generation_start = time.time()
        chunk_count = 0
        first_chunk_time = None

        async for chunk in self.generate_response_stream(user_query, analysis, context, history):
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time()
                # print(
                #     f"[OPTIMIZED] First chunk after {first_chunk_time - start_time:.2f}s total")

            # Format each content chunk as JSON
            content_packet = {
                "type": "content",
                "content": chunk
            }
            yield content_packet  # Changed to return the dict directly instead of JSON string

        total_time = time.time() - start_time
        generation_time = time.time() - generation_start
        print(
            f"[OPTIMIZED] Completed - Total: {total_time:.2f}s, Generation: {generation_time:.2f}s, Chunks: {chunk_count}")

    async def _create_empty_context(self):
        """Helper to create empty context asynchronously"""
        return ""

    def _should_use_web_search(self, query: str, docs: list) -> bool:
        if not self.enable_web_search:
            return False

        if len(docs) < 2:
            return True

        temporal_indicators = ["latest", "recent", "current", "now", "today", "this year",
                               "this month", "update", "news"]

        fact_indicators = ["how many", "statistics", "data on", "numbers for",
                           "percentage", "figure", "population"]

        # Check for specific keywords
        for indicator in temporal_indicators + fact_indicators:
            if indicator.lower() in query.lower():
                print(f" Web search triggered by indicator: '{indicator}'")
                return True

        # Check if the relevance scores of the docs are low
        if docs and hasattr(docs[0], "metadata") and "score" in docs[0].metadata:
            avg_score = sum(doc.metadata.get("score", 0)
                            for doc in docs) / len(docs)
            if avg_score < 0.7:  # Arbitrary threshold - adjust based on your embedding model
                print(
                    f" Web search triggered by low relevance score: {avg_score:.2f}")
                return True

        return False

    def retrieve_context_optimized(self, sub_queries: list, is_direct: bool = False) -> str:
        """Optimized context retrieval with performance improvements"""
        print("[OPTIMIZED] Fast context retrieval...")
        retrieval_start = time.time()

        if not self.retriever or not sub_queries:
            return "No relevant context available."

        # Log retriever configuration
        retriever_config = getattr(self.retriever, 'search_kwargs', {})
        print(f"[RETRIEVAL] Retriever config: {retriever_config}")
        print(f"[RETRIEVAL] Retriever type: {type(self.retriever).__name__}")
        print(f"[RETRIEVAL] Retriever attributes: {dir(self.retriever)[:10]}...")  # Show first 10 attributes
        print(f"[RETRIEVAL] Max docs per query: {self.max_docs_per_query}, Max total docs: {self.max_total_docs}")

        if is_direct:
            print(f"[RETRIEVAL] Direct query: '{sub_queries[0]}'")
            print(f"[RETRIEVAL] Filters: {retriever_config.get('filter', 'None')}")
            docs = self.retriever.invoke(sub_queries[0])
            print(f"[RETRIEVAL] Retrieved {len(docs)} documents from vector store")

            # Print document metadata
            print(f"\n[DOCUMENTS RETRIEVED] Showing metadata for {len(docs)} documents:")
            for idx, doc in enumerate(docs[:5], 1):  # Show first 5 docs
                metadata = doc.metadata
                print(f"  Doc {idx}:")
                print(f"    - country: {metadata.get('country', 'N/A')}")
                print(f"    - isoAlpha2: {metadata.get('isoAlpha2', 'N/A')}")
                print(f"    - tab: {metadata.get('tab', 'N/A')}")
                print(f"    - title: {metadata.get('title', 'N/A')[:80]}...")
                print(f"    - content preview (first 300 chars):")
                print(f"      {doc.page_content[:300].replace(chr(10), ' ')}...")
            print()

            # CRITICAL: Verify content matches metadata
            print("[DATA QUALITY CHECK] Verifying document content matches metadata...")
            for idx, doc in enumerate(docs[:3], 1):
                content_lower = doc.page_content[:500].lower()
                claimed_country = doc.metadata.get('country', 'N/A')
                if claimed_country != 'N/A' and claimed_country.lower() not in content_lower:
                    print(f"  ⚠️  WARNING Doc {idx}: Metadata claims '{claimed_country}' but content doesn't mention it!")
                    # Check if content mentions other countries
                    for other_country in ['são tomé', 'sao tome', 'kenya', 'nigeria', 'south africa', 'ghana']:
                        if other_country in content_lower:
                            print(f"      Content actually mentions: '{other_country}'")
                            break
                else:
                    print(f"  ✓ Doc {idx}: Content matches metadata country '{claimed_country}'")
            print()

            # Check if we need web search to supplement vector store results
            if self._should_use_web_search(sub_queries[0], docs):
                web_results = self.perform_web_search(sub_queries[0])
                if web_results and "Error" not in web_results:
                    vector_context = "\n\n".join(
                        [doc.page_content for doc in docs[:self.max_docs_per_query]])
                    context = f"{vector_context}\n\n{web_results}"
                    print("Combined vector store and web search results")
                else:
                    context = "\n\n".join(
                        [doc.page_content for doc in docs[:self.max_total_docs]])
            else:
                context = "\n\n".join([doc.page_content for doc in docs[:self.max_total_docs]])
        else:
            limited_queries = sub_queries[:2]
            all_docs = []
            print(f"[RETRIEVAL] Processing {len(limited_queries)} sub-queries:")
            for idx, query in enumerate(limited_queries, 1):
                print(f"[RETRIEVAL] Sub-query {idx}/{len(limited_queries)}: '{query}'")
                print(f"[RETRIEVAL] Filters: {retriever_config.get('filter', 'None')}")
                docs = self.retriever.invoke(query)
                print(f"[RETRIEVAL] Retrieved {len(docs)} documents for sub-query {idx}")

                # Print document metadata for this sub-query
                print(f"\n[DOCUMENTS RETRIEVED] Sub-query {idx} - Showing metadata for {len(docs)} documents:")
                for doc_idx, doc in enumerate(docs[:3], 1):  # Show first 3 docs per sub-query
                    metadata = doc.metadata
                    print(f"  Doc {doc_idx}:")
                    print(f"    - country: {metadata.get('country', 'N/A')}")
                    print(f"    - isoAlpha2: {metadata.get('isoAlpha2', 'N/A')}")
                    print(f"    - tab: {metadata.get('tab', 'N/A')}")
                    print(f"    - title: {metadata.get('title', 'N/A')[:50]}...")
                    print(f"    - content preview: {doc.page_content[:200].replace(chr(10), ' ')}...")
                print()

                all_docs.extend(docs[:self.max_docs_per_query])

            unique_docs = []
            seen_content = set()
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(content_hash)
                    if len(unique_docs) >= self.max_total_docs:
                        break

            vector_context = "\n\n".join(
                [doc.page_content for doc in unique_docs])

            print(
                f"NEW -> Retrieved {len(unique_docs)} unique documents for {len(sub_queries)} sub-queries.")

            # Print final unique documents
            print(f"\n[FINAL UNIQUE DOCUMENTS] Showing {len(unique_docs)} unique documents after deduplication:")
            for idx, doc in enumerate(unique_docs, 1):
                metadata = doc.metadata
                print(f"  Doc {idx}:")
                print(f"    - country: {metadata.get('country', 'N/A')}")
                print(f"    - isoAlpha2: {metadata.get('isoAlpha2', 'N/A')}")
                print(f"    - tab: {metadata.get('tab', 'N/A')}")
                print(f"    - title: {metadata.get('title', 'N/A')[:50]}...")
            print()
            if unique_docs and len(unique_docs) < 3 and self.enable_web_search:

                best_query = max(sub_queries, key=len)
                web_results = self.perform_web_search(best_query)
                if web_results and "Error" not in web_results:
                    context = f"{vector_context}\n\n{web_results}"
                    print("Supplemented complex query with web search results")
                else:
                    context = vector_context
            else:
                context = vector_context

        retrieval_time = time.time() - retrieval_start
        print(f"[OPTIMIZED] Context retrieved in {retrieval_time:.2f}s")
        return context

    async def process_query_stream(self, user_query: str):
        """Main CoT processing pipeline (streaming)"""
        start_time = time.time()
        analysis_start = time.time()
        print(" Analyzing query...")
        analysis = self.analyze_and_decompose_query(user_query)

        if analysis["is_relevant"] and analysis["sub_queries"]:
            print(" Retrieving context...")
            context = self.retrieve_context_with_subqueries(
                analysis["sub_queries"],
                analysis.get('is_direct', False)
            )
            print(f"Context length: {len(context)} characters")
            print(f" Context preview: {context[:200]}..." if len(
                context) > 200 else f"Full context: {context}")
        else:
            context = ""
            print("No context to retrieve")
        history_start = time.time()
        history = self.get_session_history_text()
        print(" Generating response...")

        generation_start = time.time()
        chunk_count = 0
        first_chunk_time = None

        async for chunk in self.generate_response_stream(user_query, analysis, context, history):
            chunk_count += 1
            if first_chunk_time is None:
                first_chunk_time = time.time()
            yield chunk

        total_time = time.time() - start_time
        generation_time = time.time() - generation_start

