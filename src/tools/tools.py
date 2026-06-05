import os
import warnings

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from langchain_core.documents import Document
from langchain_core.tools import InjectedToolCallId, tool
from tavily import TavilyClient
from typing import Annotated

from langgraph.prebuilt import InjectedState

from src.vector_store import vs_search 



@tool(response_format='content_and_artifact')
def vector_store_search(
    query : str,
    k : int = 4,
    session_id:Annotated[str,InjectedState('session_id')] = None
    ):
    """Search the vector store for relevant document passages.
    Adjust k (default 4 and retrieved max document must be less then 7) to retrieve more or fewer passages."""
    
    retriever = vs_search(session_id,k)
    retrieved_docs = retriever.invoke(query)
    
    context = "\n\n## Vector Store Results\n\n" + "\n\n".join(d.page_content for d in retrieved_docs)
    return context,retrieved_docs




@tool(response_format="content_and_artifact")
def web_search(query : str, k : int = 3):
    """Search the web for current or real-time information.
    Adjust max_results (default 3 and retrieved max result must be less then 7) to control how many results are returned."""
    client = TavilyClient(api_key= os.environ["TAVILY_API_KEY"])
    
    response = client.search(
        query=query,
        max_results=k
    )
    
    tavily_docs = []
    
    for resp in response["results"]:

        metadata = {
            "url": resp["url"],
            "title": resp["title"]
        }

        doc = Document(
            page_content=resp["content"],
            metadata=metadata
        )

        tavily_docs.append(doc)
        
    context = "\n\n## Web Search Results\n\n" + "\n\n".join(d.page_content for d in tavily_docs)
    return context,tavily_docs