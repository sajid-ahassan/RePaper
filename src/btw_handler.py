import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.models import btw_retrieval
from dotenv import load_dotenv

from tavily import TavilyClient
load_dotenv()
model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])

def handler(query):
    
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    
    dec = (route_prompt | model.with_structured_output(btw_retrieval)).invoke({'query':query}).need_web_retrieval
    
    if dec:
        client = TavilyClient(api_key= os.environ["TAVILY_API_KEY"])
        resp = client.search(
            query = query,
            max_results=3
        ).get('results',[])
        
        context = "\n\n".join(con['content'] for con in resp)
        sources = "\n".join(f"- {r['url']}" for r in resp)
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using the web search results below. Be concise.\n\n"
             "Results:\n{context}\n\nSources:\n{sources}"),
            ("human", "{query}"),
        ])
        kwargs = {'query':query,'context':context,'sources':sources}
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system","Answer the question from your general knowladge but never add any false information. Be concise.\n\n"),
            ("human", "{query}"),
        ])
        kwargs = {'query':query}
        
    for chunk in (answer_prompt | model).stream(kwargs):
        if chunk.content:
            yield chunk.content