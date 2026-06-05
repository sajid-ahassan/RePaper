import os
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from langchain.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


import streamlit as st

from src.paper_loader import document_loader,paper_loader
from src.vector_store import add_paper,load_papers
from src.btw_handler import handler
from src.graph import build_graph

@st.cache_resource
def get_graph():
 return build_graph(r'C:\Users\sajid\OneDrive\Desktop\RePaper\src\DB\chatbot.db')

model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])
graph = get_graph()

SESSION_FILE = ('session.json')



# =========================================== Utilities ===============================================

def generate_thread_id():
    '''Generate new thread id and add it to the json'''
    thread_id = str(uuid.uuid4())
    st.session_state['session_meta'][thread_id] = {
        'thread' : thread_id,
        'thread_title':'New Conversation',
        'is_renamed':False,
        'created_at':datetime.now().isoformat()
    }
    save_session(st.session_state['session_meta'])
    return thread_id


def save_session(session_details):
    '''Helper to save session'''
    try:
        with open(SESSION_FILE, 'w', encoding="utf-8") as f:
            json.dump(session_details, f, indent=2)
    except Exception:
        raise Exception

def load_session() -> dict:
    """helpwe to load session from the session.json"""
    try:
        with open(SESSION_FILE, 'r', encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    

def reset_chat():
    '''generates new thread and clear the chat'''
    new_thread = generate_thread_id()
    st.session_state['active_thread'] = new_thread
    st.session_state['chats'] = []
    st.rerun()
    
    

def load_chat(thread):
    '''Fatches chat history from the state using current thread id and returns custome chat list'''
    messages =  graph.get_state(config = {'configurable':{'thread_id':thread}}).values.get('messages',[])
    try:
        chats = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                chats.append({'role':'user','content':msg.content})

            elif isinstance(msg, AIMessage):
                if msg.response_metadata.get('is_final_answer') == True:
                    chats.append({'role': 'assistant', 'content': msg.content})
        return chats
    except Exception:
        return []
    

def switch_session(thread):
    '''switch thread and load chats'''
    st.session_state['active_thread'] = thread
    chats = load_chat(st.session_state['active_thread']) 
    st.session_state['chats'] = chats
    st.rerun()
    
def delete_chat(thread_id):
    """Delete chat only from session.json."""

    if thread_id in st.session_state["session_meta"]:
        del st.session_state["session_meta"][thread_id]
        save_session(st.session_state["session_meta"])

    # If deleted chat was active, switch to another existing chat
    if st.session_state.get("active_thread") == thread_id:
        if st.session_state["session_meta"]:
            latest = max(
                st.session_state["session_meta"].values(),
                key=lambda s: s["created_at"]
            )
            st.session_state["active_thread"] = latest["thread"]
            st.session_state["chats"] = load_chat(latest["thread"])
        else:
            new_thread = generate_thread_id()
            st.session_state["active_thread"] = new_thread
            st.session_state["chats"] = []

    st.rerun()
    
    
    
# =================================== Session_state handling =========================================================
    
if 'session_meta' not in st.session_state:
    st.session_state['session_meta'] = load_session()
    

if 'active_thread' not in st.session_state:
    if st.session_state['session_meta']:
        latest = max(
            st.session_state.session_meta.values(),
            key=lambda s: s["created_at"],
        )
        st.session_state['active_thread'] = latest['thread']
    else:
        st.session_state['active_thread'] = generate_thread_id()
        
        
if 'chats' not in st.session_state:
    st.session_state['chats'] = load_chat(st.session_state['active_thread'])
    

for msg in st.session_state['chats']:
    if msg['role'] == 'user':
        with st.chat_message('user',avatar="👤"):
            st.markdown(msg['content'])
    else:
        with st.chat_message('assistant',avatar="🤖"):
            st.markdown(msg['content'])


# ============================================ Streamlit Frontend ================================================

st.sidebar.title('RePaper')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for key, values in reversed(st.session_state['session_meta'].items()):
    thread_id = values['thread']
    display_name = values['thread_title']
    is_active = (thread_id == st.session_state.get('active_thread'))
    button_type = "primary" if is_active else "secondary"

    title_col, delete_col = st.sidebar.columns([0.84, 0.16], gap="small")

    with title_col:
        if st.button(
            display_name,
            type=button_type,
            key=f"btn_{thread_id}",
            use_container_width=True,
        ):
            switch_session(thread_id)

    with delete_col:
        if st.button(
            "×",
            key=f"delete_{thread_id}",
            help=f"Delete {display_name}",
            use_container_width=True,
        ):
            delete_chat(thread_id)
        
        
        
        
# =========================================== File uploading section ==========================================

st.sidebar.divider()
st.sidebar.header('Add Context to Chat')

sidebar_files = st.sidebar.file_uploader(
    "Upload documents for this conversation", 
    accept_multiple_files=True,
    type=['pdf', 'txt'] 
)

papers = load_papers(st.session_state['active_thread'])

if sidebar_files:
    if st.sidebar.button("Process & Embed", type="primary", use_container_width=True):
        
        with st.sidebar.spinner("Reading documents..."):
            for file in sidebar_files:
                
                if file.name in papers:
                    continue
                suffix = Path(file.name).suffix
                st.sidebar.write(f"Processing {file.name}...")
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name
                    
                    
                    docs = document_loader(temp_file_path,file.name)
                    add_paper(session_id=st.session_state['active_thread'],docs=docs)
                    
                    os.remove(temp_file_path)
                except Exception:
                    st.error(f"Failed: {file.name} — {Exception}")
                    
            st.rerun()
            
st.sidebar.divider()

# ============================================== load from website, arxiv ================================================

st.sidebar.title("Universal Document Loader")


with st.sidebar:
    st.header("Input Settings")

    input_type = st.radio(
        "What are you loading?",
        ["Website URL", "ArXiv ID", "Paper Title"]
    )

    user_input = st.text_input(f"Enter the {input_type}:")

    submit = st.button("Fetch & Process")

if submit and user_input:
    with st.sidebar.spinner(f"Fetching {input_type}..."):
        try:
            if input_type == "Website URL":
                docs = document_loader(user_input, "website")

            elif input_type in ["ArXiv ID", "Paper Title"]:
                docs = paper_loader(user_input)

            add_paper(
                session_id=st.session_state["active_thread"],
                docs=docs
            )

            st.success("Done!")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.rerun()

st.sidebar.divider()


# ================================================== Displaying loadded Documents ======================================

st.sidebar.subheader("Active Knowledge Base")

if papers:
    for paper in papers:
        st.sidebar.caption(f"{paper}") 
else:
    st.sidebar.info("No documents uploaded in this session.")

st.sidebar.divider()
        
        
        
# ========================================= Chating section ============================================================
            
config = {'configurable':{'thread_id':st.session_state['active_thread']}}


user_input = st.chat_input('Type here')
if user_input:
    
    # ============ btw_handler ============
    
    is_btw = user_input.strip().lower().startswith("/btw")
    if is_btw:
        query = user_input.strip()[4:].strip()
        
        with st.expander(" **Side Channel Interaction** (Not saved to history)", expanded=True):
            
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
                
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = handler(query)
                st.info(response, icon="✨") 
    else :
    # ============= Use graph ===============
    
        input_state = {
            "messages": [HumanMessage(user_input)],
            "session_id": st.session_state['active_thread'],
            "query":user_input,
            "fallback": False,
            "query_retries": 0,
            "tool_retries": 0,
            "retrieved_docs": [],
            "good_docs": [],
            "context": "",
            "decomposed_query": None,
        }
        
        with st.chat_message('user',avatar="👤"):                       # user query display
            st.markdown(user_input)
            
        st.session_state['chats'].append({'role':'user','content':user_input})
        
        
        with st.chat_message('assistant',avatar="🤖"):                  # ai msg display

            def stream_answer():
                for chunk, meta in graph.stream(input_state, config=config, stream_mode='messages'):
                    if meta.get('langgraph_node') == 'answer' and chunk.content:
                        yield chunk.content

            ai_message = st.write_stream(stream_answer)

            if not ai_message:
                final_state = graph.get_state(config).values
                ai_message = final_state.get("answer", "No response generated.") 
                st.markdown(ai_message)
            
    
        st.session_state['chats'].append({'role':'assistant','content':ai_message})
        # print(final_state)
        
            
    # ================ Chat title renaming ========================
        
        meta = st.session_state['session_meta'][st.session_state['active_thread']]
        
        if not meta['is_renamed']:
            title = response = model.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "Generate a concise 3-5 word title for a research chat session "
                            "based on the user's first message. Return only the title, "
                            "no punctuation at the end, no quotes."
                        ),
                    },
                    {"role": "user", "content": user_input[:500]},
                ]
            ).content
            meta['thread_title'] = title   
            meta['is_renamed'] = True 
        
        with open(SESSION_FILE, 'w') as f:
            json.dump(st.session_state['session_meta'], f)
            
        st.rerun()
    