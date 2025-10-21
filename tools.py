import dspy

from tavily import TavilyClient
from openai import OpenAI
from dotenv import load_dotenv
import os


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
    results = fetch_relevant_info(query)
    if results:
        return "\n\n".join(results[:3])
    return "No relevant information found on the web."