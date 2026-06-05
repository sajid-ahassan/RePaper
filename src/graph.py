import os
import sqlite3
import warnings
from typing import Literal

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI,OpenAIEmbeddings


from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph,START,END
from langgraph.prebuilt import ToolNode


from src.prompts import AGENT_SYSTEM_PROMPT
from src.tools.tools import vector_store_search,web_search
from src.models import (
    RetrieveDecision,
    DecomposeDecision,
    DecomposeQueries,
    relevency_schema,
    claim_ev,
    claim_vardics
)
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])
agent_model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])
base_embedding = OpenAIEmbeddings(model = 'text-embedding-3-small',api_key= os.environ["OPENAI_API_KEY"])



class State(MessagesState):
    query : str
    session_id : str
    retrieved_docs : list[Document]
    good_docs : list[Document]
    
    context : str
    route : Literal['retrieval','direct_generate','verify_claim']
    
    need_decomposition : bool
    decomposed_query : str
    
    tool_retries : int
    query_retries : int
    
    claim_verification_result : Literal['superseded','partially superseded','not superseded']
    claim_evidance : list[claim_ev]
    
    fallback : bool
    
    answer : str
    
    
tools = [vector_store_search,web_search]
agent_model_with_tool = agent_model.bind_tools(tools)
tool_node = ToolNode(tools)



decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
        '''You are an intelligent routing agent for a Retrieval-Augmented Generation (RAG) system. Your primary task is to analyze incoming user queries and determine the optimal execution path. 

        Based on the nature of the query, you must classify it into exactly one of the following three routes:

        * **`direct_generate`**: Choose this route if your internal parametric knowledge is sufficient to answer the query accurately and completely. Use this for general knowledge, programming syntax, conversational greetings, or foundational concepts where there is zero risk of hallucination.
        * **`retrieval`**: Choose this route if the query requires pulling external context to answer accurately. Use this when the system needs to perform a vector database search or a general web search to gather missing facts, recent news, or specific document references You will have two tools to collect the context. One is vector search tool to pull context from documents and enother is web search tool to pull any recent info from the websites.
        * **`verify_claim`**: Choose this route if the user is presenting a specific factual, scientific, or academic claim that requires rigorous validation. Use this when the system must cross-reference the query against specialized academic repositories (like arXiv) alongside general web searches to prove, disprove, or add nuance to the claim.

        Analyze the user's input carefully and output ONLY the precise string literal that matches the required route.''',
        ),
        ("human", "Question: {question}"),
    ]
)

def retrieval_path(state:State):
    query = state['query']
    chain = decide_retrieval_prompt | model.with_structured_output(RetrieveDecision)
    resp = chain.invoke({'question':query})
    return {'route':resp.route}




CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

def verify_claim(state:State):
    client = TavilyClient()
    web_response = client.search(
        max_results= 2,
        query=f"recent research superseding : {state['query'][:300]}"
    ).get('results',[])
    
    arxiv_response = client.search(
        max_results= 2,
        query=f"site:arxiv.org : {state['query'][:300]}"
    ).get('results',[])
    
    context = ['===== General web search result ======\n']
    for resp in web_response:
        context.append(
            f"Title : {resp.get('title','')}\n"
            f"url : {resp.get('url','')}\n"
            f"Snippet : {resp.get('content','')[:1000]}\n"
        )
        
    context.append("==== Arxiv paper search result ====")
    for resp in arxiv_response:
        context.append(
            f"Title : {resp.get('title','')}\n"
            f"url : {resp.get('url','')}\n"
            f"Snippet : {resp.get('content','')[:1000]}\n"
        )

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{state['query']}\n\n"
        f"Search Results:\n{context}"
    )
    
    resp = model.with_structured_output(claim_vardics).invoke(prompt)
    return {'claim_verification_result':resp.claim_verification_result,'claim_evidance':resp.claim_evidance}

    
    
    
    
    
def check_decomposition(state:State):

    prompt = ChatPromptTemplate.from_messages([
        ('system',"You are a query analysis assistant. Determine if the user's question asks for "
            "multiple distinct pieces of information that each require separate retrieval. "
            "If yes, needs_decomposition should be True. "
            "If the question is simple or self-contained, needs_decomposition should be False."),
        ('human',"Query : {query}")
    ])
    chain = prompt | model.with_structured_output(DecomposeDecision)
    resp = chain.invoke({'query':state['query']})
    return {"need_decomposition": resp.need_decompose}


def decompose_query(state:State):
    query = state['query']
    decompose_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a query decomposition assistant. Break the user's question into focused, "
            "self-contained sub-queries — one per retrieval step. Each sub-query should target "
            "a single distinct piece of information."
        ),
        ("human", "{query}"),
    ])

    chain = decompose_prompt | model.with_structured_output(DecomposeQueries)
    resp = chain.invoke({'query':query})
    new_query = "\n".join(f"Step {i+1}: {q}" for i,q in enumerate(resp.queries))
    
    formatted = (
        f"You must call a retrieval tool for each of the following {len(resp.queries)} steps "
        f"before answering. Do not skip any step.\n\n{new_query}"
    )
    return {'decomposed_query': formatted}
    
    
    
    
def agent(state: State):
    history = state.get('messages', [])
    query = state.get("decomposed_query") or state['query']
    
    # 1. Dynamically build the system prompt with the current goal injected
    dynamic_system_prompt = f"""{AGENT_SYSTEM_PROMPT}

    CRITICAL INSTRUCTIONS:
    1. Your current retrieval target is: 
    {query}
    2. Review the tool outputs below. If the information needed is already present in the history, simply reply with 'All information retrieved.' and DO NOT call any more tools."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", dynamic_system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | agent_model_with_tool
    
    resp = chain.invoke({'messages': history})
    
    return {'messages': [resp]}
    
    
    

def tool_use_limit(state:State):
    cnt = state.get('tool_retries',0)
    return {'tool_retries':cnt+1}
    
    
    
    
def collect_retrieved_docs(state:State):
    good_docs = []
    for doc in reversed(state['messages']):
        if isinstance(doc,HumanMessage):
            break
        if isinstance(doc,ToolMessage):
            good_docs.extend(doc.artifact)
    # print(good_docs)
    good_docs.reverse()
    return {'retrieved_docs':good_docs}
        
        
        
    
is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true."
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)    
def fatch_relevent_docs(state:State):
    good_docs = []
    query = state.get('decomposed_query') or state['query']
    chain = is_relevant_prompt | model.with_structured_output(relevency_schema)
    
    for doc in state['retrieved_docs']:
        resp = chain.invoke({'question':query,'document':doc}).is_relevent
        if resp:
            good_docs.append(doc)
    # print(good_docs)
    context = '\n\n'.join(d.page_content for d in good_docs)

    return {'good_docs':good_docs,'context':context}





def modify_query(state:State):
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a query rewriting assistant. A retrieval system searched for information to answer the user's "
            "query but none of the retrieved documents were relevant. Your task is to rewrite the query to be more "
            "specific, use different terminology, or focus on a narrower aspect that is more likely to match "
            "content in the knowledge source. Return only the rewritten query string."
        ),
        ("human", "Original query: {query}\n\nRewritten query:"),
    ])
    
    chain = prompt_template | model
    response = chain.invoke({"query": state["query"]})
    
    new_query = response.content.strip()
    
    cnt = state.get('query_retries',0)
    return {'decomposed_query':new_query,'query_retries':cnt+1}



MAX_QUERY_RETRIES = 3


def fallback_node(state:State):
    if state.get('query_retries',0) > MAX_QUERY_RETRIES:
        return {'fallback':True}
    return {}





def generate(state: State) -> dict:
    if state.get("fallback"):
        
        response = AIMessage(
            content=(
                "I was unable to find relevant information to answer your query after multiple retrieval attempts. "
                "The knowledge source does not appear to contain content that addresses this question."
                )
            )
        response.response_metadata['is_final_answer'] = True
        return {
            "answer": response.content,
            "messages": [response]
        }

        
        
    direct_generate_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful, highly knowledgeable AI assistant. "
            "Answer the user's query clearly and accurately based on your internal knowledge. "
            "If the request involves coding, provide clean, well-documented solutions."
        )),
        MessagesPlaceholder(variable_name='messages'),
        ("human", "{query}"),
    ])
    
    retrieval_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            '''You are an AI assistant for question answering using retrieved context documents.

            Your task is to answer the user's question ONLY using the provided context. 
            Follow these rules strictly:

            1. Use only the information from the context.
            2. Do not use outside knowledge.
            3. If the answer is not contained in the context, say:
            "I could not find enough information in the provided context."
            4. Be concise but complete.
            5. When possible, cite which context chunk or source supports the answer.
            7. Do not hallucinate or invent facts.
            8. For research-related questions:
            - distinguish between claims, evidence, and conclusions
            - preserve technical accuracy
            9. You must provide data/info source(web,vector search) after finishing the answer to the user. If there any url or importent metadata, you may provide them as well for best user experiance'''
            
            "{context}"
        )),
        MessagesPlaceholder(variable_name='messages'),
        ("human", "Question: {query}"),
    ])
    
    
    verify_claim_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert fact-checker and scientific research assistant. "
            "Your task is to verify the user's claim using ONLY the provided search results. "
            "The context includes general web searches and academic papers (arXiv).\n\n"
            "Analyze the context carefully and structure your response as follows:\n"
            "Structure your response exactly as follows, using Markdown for readability:\n\n"
            "### 1. Verdict\n"
            "**[Superseded / Partially Superseded / Not Superseded]**\n\n"
            "### 2. Analysis\n"
            "Explain your reasoning clearly. Point out any consensus or contradictions "
            "between the academic papers and the general web results.\n\n"
            "### 3. Supporting Evidence\n"
            "If evidence was provided in the context, list 1 to 2 specific sources used "
            "to reach your verdict. Format each as a bullet point containing the Title and url(if any)"
            "between the academic papers and the general web results.\n\n"
            "Context:\n"
            "{context}\n"
            "Claim_verification_result:\n"
            "{result}"
        )),
        MessagesPlaceholder(variable_name='messages'),
        ("human", "Claim to verify: {query}"),
    ])
    
    if state['route'] == 'direct_generate':
        prompt_template = direct_generate_prompt
        invoke_kwargs = {'query':state['query'],'messages':state['messages']} 
    elif state['route'] == 'retrieval':
        prompt_template = retrieval_prompt
        invoke_kwargs = {'query':state['query'],'context':state.get('context',''),'messages':state['messages']}
    else:
        prompt_template = verify_claim_prompt
        invoke_kwargs = {'query':state['query'],'context':state.get("claim_evidance",[]),'result':
            state.get('claim_verification_result',''),'messages':state['messages']}
        
    
    response = (prompt_template | model).invoke(invoke_kwargs)
    
    response.response_metadata["is_final_answer"] = True
    return {"answer": response.content,'messages':[response]}








def need_retrieval_route(state:State) -> Literal['generate','check_decomposition','verify_claim']:
    if state['route'] == 'retrieval':
        return 'check_decomposition'
    elif state['route'] == "direct_generate":
        return 'generate'
    elif state['route'] == "verify_claim":
        return 'verify_claim'


def route_decompose(state:State)->Literal['agent','decompose_query']:
    if state['need_decomposition']:
        return 'decompose_query'
    return "agent"


def tool_call_route(state:State)->Literal['tool_node','collect_retrieved_docs']:
    return 'tool_node' if state['messages'][-1].tool_calls  else 'collect_retrieved_docs'


MAX_TOOL_CALL = 5

def tool_limit_route(state:State)->Literal['agent','collect_retrieved_docs']:
    return 'agent' if state['tool_retries'] < MAX_TOOL_CALL  else 'collect_retrieved_docs'



def check_relevence_route(state:State)->Literal['fallback_node','modify_query']:
    return 'fallback_node' if len(state.get('good_docs',[])) > 0 or state.get('query_retries', 0) > MAX_QUERY_RETRIES else 'modify_query'
    
    
    
    

    
def build_graph(db_path):
    con = sqlite3.connect(db_path,check_same_thread=False)
    checkpoint = SqliteSaver(con)
    graph = StateGraph(State)

    graph.add_node('retrieval_path',retrieval_path)
    graph.add_node('check_decomposition',check_decomposition)
    graph.add_node('verify_claim',verify_claim)
    graph.add_node('generate',generate)
    graph.add_node('agent',agent)
    graph.add_node('decompose_query',decompose_query)
    graph.add_node('tool_node',tool_node)
    graph.add_node('tool_use_limit',tool_use_limit)
    graph.add_node('collect_retrieved_docs',collect_retrieved_docs)
    graph.add_node('modify_query',modify_query)
    graph.add_node('fatch_relevent_docs',fatch_relevent_docs)
    graph.add_node('fallback_node',fallback_node)

    graph.add_edge(START,'retrieval_path')
    graph.add_conditional_edges('retrieval_path',need_retrieval_route)
    graph.add_conditional_edges('check_decomposition',route_decompose)
    graph.add_edge('decompose_query','agent') 
    graph.add_conditional_edges('agent',tool_call_route) 
    graph.add_edge('tool_node','tool_use_limit') 
    graph.add_conditional_edges('tool_use_limit',tool_limit_route) 
    graph.add_edge('collect_retrieved_docs','fatch_relevent_docs') 
    graph.add_conditional_edges('fatch_relevent_docs',check_relevence_route) 
    graph.add_edge('modify_query','agent')
    graph.add_edge('verify_claim','generate') 
    graph.add_edge('fallback_node','generate') 
    graph.add_edge('generate',END) 
    
    workflow = graph.compile(checkpointer = checkpoint)
    return workflow


