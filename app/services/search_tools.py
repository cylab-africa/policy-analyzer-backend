from app.core.config import settings
from typing import Dict, List, Any, Optional

class SearchToolFactory:
    @staticmethod
    def perform_search(query: str, tool_type: str = "tavily", max_results: int = 3) -> str:
        if tool_type == "tavily":
            return SearchToolFactory._tavily_search(query, max_results)
        elif tool_type == "serpapi":
            return SearchToolFactory._serpapi_search(query, max_results)
        elif tool_type == "duckduckgo":
            return SearchToolFactory._duckduckgo_search(query, max_results)
        else:
            return f"Error: Unknown search tool type '{tool_type}'"
    
    @staticmethod
    def _tavily_search(query: str, max_results: int = 3) -> str:
        try:
            api_key = settings.tavily_api_key
            if not api_key:
                return "Error: TAVILY_API_KEY environment variable not set"
            
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=api_key)
            search_result = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            if not search_result.get('results'):
                return "No results found."
            
            formatted_results = "**Web Search Results:**\n\n"
            for idx, result in enumerate(search_result['results'][:max_results]):
                formatted_results += f"**Source {idx+1}: {result.get('title', 'Unknown Title')}**\n"
                formatted_results += f"URL: {result.get('url', 'No URL')}\n"
                formatted_results += f"{result.get('content', 'No content available')}\n\n"
            
            return formatted_results
        
        except Exception as e:
            print(f"Tavily search error: {str(e)}")
            return f"Error performing Tavily search: {str(e)}"
    
    @staticmethod
    def _serpapi_search(query: str, max_results: int = 3) -> str:

        try:
            api_key = settings.serpapi_api_key
            if not api_key:
                return "Error: SERPAPI_API_KEY environment variable not set"
            
            from langchain_community.utilities import SerpAPIWrapper
            
            search = SerpAPIWrapper(serpapi_api_key=api_key)
            result = search.run(query)
            
            return f"**Web Search Results:**\n\n{result}"
            
        except Exception as e:
            print(f"SerpAPI search error: {str(e)}")
            return f"Error performing SerpAPI search: {str(e)}"
    
    @staticmethod
    def _duckduckgo_search(query: str, max_results: int = 3) -> str:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            
            search = DuckDuckGoSearchRun()
            result = search.run(query)
            
            if len(result) > 1500:
                result = result[:1500] + "...[truncated]"
            
            return f"**Web Search Results:**\n\n{result}"
            
        except Exception as e:
            print(f"DuckDuckGo search error: {str(e)}")
            return f"Error performing DuckDuckGo search: {str(e)}"
