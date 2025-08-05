from mcp.server.fastmcp import FastMCP
from fastapi import HTTPException

import httpx
from ddgs import DDGS
from readability import Document
from bs4 import BeautifulSoup
import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

mcp = FastMCP(
    name="WebSearchMCP",
    prompt="This is a Web Search server. You can use it to find information on the internet by providing a search query.",
)


@mcp.tool()
async def web_search(query: str) -> dict:
    """
    Searches the web for a given query and returns the cleaned text information
    from the first search result. Useful for realtime/recent data.  Useful for information outside of basic LLM knowledge.

    Args:
        query (str): The search term or question.

    Returns:
        dict: A dictionary containing the original query, the source URL,
              and the cleaned content of the page.
    """
    if not query or not query.strip():
        logger.error("web_search called with an empty query.")
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'query' is required and cannot be empty.",
        )

    logger.info(f"Searching for query: '{query}'")
    with DDGS() as ddgs:
        results = list(ddgs.text(query.strip(), max_results=1))
        if not results:
            logger.error(f"No search results found for query: '{query}'")
            raise HTTPException(
                status_code=404, detail=f"No web results found for query: {query}"
            )

    source_url = results[0]["href"]
    logger.info(f"Found URL: {source_url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(source_url, headers=headers, timeout=20.0)
            response.raise_for_status()
            html_content = response.text
            logger.info(f"Successfully fetched content from {source_url}")

    except httpx.RequestError as e:
        logger.error(f"HTTP request failed for {source_url}: {e}")
        raise HTTPException(
            status_code=502, detail=f"Could not fetch content from source URL: {e}"
        )

    try:
        doc = Document(html_content)
        cleaned_html = doc.summary()
        soup = BeautifulSoup(cleaned_html, "lxml")
        content = soup.get_text(separator="\n", strip=True)
        if len(content) > 1000:
            content = content[0:1000]
        logger.info(f"Successfully parsed content for query: '{query}'")
    except Exception as e:
        logger.error(f"Failed to parse HTML content: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to parse content from source: {e}"
        )

    return {"query": query, "source_url": source_url, "content": content}


if __name__ == "__main__":
    mcp.run()
