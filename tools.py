import dspy

from tavily import TavilyClient
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging

#configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load environment variables from .env file
load_dotenv()

# openai_api_key=os.getenv("OPENAI_API_KEY")

# lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_api_key)
# dspy.configure(lm=lm)

tavily_client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    """
    Search the web using Tavily for current information.
    Use this for questions requiring real-time or external information.
    """

    logging.info(f"Searching web for: {query}")
    try: 
        results = tavily_client.search(query=query)
    except Exception as e:
        logging.error(f"Web search failed :{e}")
        return "Error occured while searching the web"

    if not results:
        logging.warning("No relevant inforation found on the web")
        return "No relevant information found on the web."

    #  if results contain a list of dicts, handle that case
    if isinstance(results, dict) and "results" in results:
        items = results["results"]
        logging.info(f"Retrieved {len(items)} web results.")
        top_results = "\n\n".join(
            [f"{item.get('title', 'No title')}: {item.get('content', '')}" for item in items[:3]]
        )
        logging.debug(f"Top result preview: {top_results[:200]}...")
        return top_results

    # if it's already a list of strings or summaries
    if isinstance(results, list):
        logging.info(f"Retrieved {len(results)} web results (list).")
        logging.debug(f"Top result preview: {results[0][:200]}...")
        return "\n\n".join(results[:3])

    logging.warning(f"Unexpected result format: {type(results)}")
    return "Unexpected format received from web search."