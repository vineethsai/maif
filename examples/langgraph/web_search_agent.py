"""
Web Search Agent - Fallback when local KB doesn't have the answer.

Adds ability to search the web if the local knowledge base doesn't contain
relevant information for the user's question.
"""

import sys
import os
import requests
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def should_use_web_search(retrieved_chunks: List[Dict], threshold: float = 0.3) -> bool:
    """
    Determine if web search is needed based on retrieval quality.
    
    Args:
        retrieved_chunks: Chunks from local KB
        threshold: Minimum score threshold for relevance
        
    Returns:
        True if web search should be used
    """
    if not retrieved_chunks:
        return True
    
    # Check if best match is below threshold
    best_score = max(chunk.get('score', 0) for chunk in retrieved_chunks)
    
    return best_score < threshold


def search_web(query: str, num_results: int = 3) -> List[Dict]:
    """
    Search the web using a search API.
    
    NOTE: This is a simplified implementation. In production, use:
    - Google Custom Search API
    - Bing Search API
    - SerpAPI
    - Or other search providers
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of search results
    """
    print(f"   🌐 Searching the web for: '{query}'")
    print(f"   ℹ️  Using DuckDuckGo (no API key needed)")
    
    try:
        # Use DuckDuckGo's instant answer API (free, no key needed)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        
        # Get abstract if available
        if data.get('AbstractText'):
            results.append({
                "title": data.get('Heading', 'Web Result'),
                "snippet": data.get('AbstractText', ''),
                "url": data.get('AbstractURL', ''),
                "source": "DuckDuckGo"
            })
        
        # Get related topics
        for topic in data.get('RelatedTopics', [])[:num_results-1]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    "title": topic.get('Text', '')[:50],
                    "snippet": topic.get('Text', ''),
                    "url": topic.get('FirstURL', ''),
                    "source": "DuckDuckGo"
                })
        
        if results:
            print(f"   ✅ Found {len(results)} web results")
            return results
        else:
            print(f"   ⚠️  No web results found")
            return []
            
    except Exception as e:
        print(f"   ⚠️  Web search failed: {e}")
        return []


def format_web_results_as_chunks(web_results: List[Dict]) -> List[Dict]:
    """
    Format web search results as chunks for RAG pipeline.
    
    Args:
        web_results: Results from web search
        
    Returns:
        Formatted chunks compatible with RAG system
    """
    chunks = []
    
    for i, result in enumerate(web_results):
        chunk = {
            "doc_id": f"web_{i}",
            "chunk_index": 0,
            "text": f"{result.get('title', '')}\n\n{result.get('snippet', '')}",
            "score": 0.8,  # Give web results decent score
            "block_id": f"web_result_{i}",
            "metadata": {
                "source": "web_search",
                "url": result.get('url', ''),
                "search_engine": result.get('source', 'unknown')
            }
        }
        chunks.append(chunk)
    
    return chunks

