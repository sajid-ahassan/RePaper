import os
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

from dotenv import load_dotenv
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from src.paper_loader import document_loader, paper_loader
from src.vector_store import add_paper, load_papers
from src.btw_handler import handler
from src.graph import build_graph

# ========================================= LOAD ENV =========================================

load_dotenv()

# ========================================= PAGE CONFIG =========================================

st.set_page_config(
    page_title="RePaper",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================================= CUSTOM CSS =========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main App */
.stApp {
    background-color: #0f1117;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161a23;
    border-right: 1px solid #262b36;
}

/* Hide Streamlit menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main Header */
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.3rem;
}

.subtitle {
    color: #9ca3af;
    margin-bottom: 2rem;
    font-size: 1rem;
}

/* Chat Messages */
.user-msg {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 14px 18px;
    border-radius: 18px;
    color: white;
    margin-bottom: 12px;
    box-shadow: 0 4px 10px rgba(37,99,235,0.2);
}

.ai-msg {
    background: #1f2937;
    padding: 14px 18px;
    border-radius: 18px;
    border: 1px solid #374151;
    color: white;
    margin-bottom: 12px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

/* Cards */
.card {
    background: #1a1f2b;
    border: 1px solid #2b3240;
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Knowledge base item */
.kb-item {
    background: #1f2937;
    border: 1px solid #374151;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 0.92rem;
}

/* Section title */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f3f4f6;
    margin-bottom: 1rem;
}

/* Chat Input */
[data-testid="stChatInput"] {
    background-color: #1a1f2b;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)

# ========================================= GRAPH =========================================

@st.cache_resource
def get_graph():
 return build_graph(r'C:\Users\sajid\OneDrive\Desktop\RePaper\src\DB\chatbot.db')

graph = get_graph()

# ========================================= MODEL =========================================

model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])

# ========================================= SESSION FILE =========================================

SESSION_FILE = "session.json"

# ========================================= UTILITIES =========================================

def save_session(session_details):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_details, f, indent=2)

    except Exception:
        pass


def load_session():
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def generate_thread_id():

    thread_id = str(uuid.uuid4())

    st.session_state["session_meta"][thread_id] = {
        "thread": thread_id,
        "thread_title": "New Conversation",
        "is_renamed": False,
        "created_at": datetime.now().isoformat()
    }

    save_session(st.session_state["session_meta"])

    return thread_id


def reset_chat():

    new_thread = generate_thread_id()

    st.session_state["active_thread"] = new_thread
    st.session_state["chats"] = []

    st.rerun()


def load_chat(thread):

    messages = graph.get_state(
        config={"configurable": {"thread_id": thread}}
    ).values.get("messages", [])

    try:

        chats = []

        for msg in messages:

            if isinstance(msg, HumanMessage):

                chats.append({
                    "role": "user",
                    "content": msg.content
                })

            elif isinstance(msg, AIMessage):

                if msg.response_metadata.get("is_final_answer"):

                    chats.append({
                        "role": "assistant",
                        "content": msg.content
                    })

        return chats

    except Exception:
        return []


def switch_session(thread):

    st.session_state["active_thread"] = thread
    st.session_state["chats"] = load_chat(thread)

    st.rerun()

# ========================================= SESSION STATE =========================================

if "session_meta" not in st.session_state:
    st.session_state["session_meta"] = load_session()

if "active_thread" not in st.session_state:

    if st.session_state["session_meta"]:

        latest = max(
            st.session_state["session_meta"].values(),
            key=lambda s: s["created_at"]
        )

        st.session_state["active_thread"] = latest["thread"]

    else:
        st.session_state["active_thread"] = generate_thread_id()

if "chats" not in st.session_state:
    st.session_state["chats"] = load_chat(
        st.session_state["active_thread"]
    )

# ========================================= MAIN HEADER =========================================

st.markdown("""
<div class='main-title'>AI Research Assistant</div>
<div class='subtitle'>
Professional AI-powered document research and analysis workspace
</div>
""", unsafe_allow_html=True)

# ========================================= SIDEBAR =========================================

with st.sidebar:

    st.markdown("## 💬 Conversations")

    if st.button("➕ New Chat"):
        reset_chat()

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================= CONVERSATIONS =========================================

    for key, values in reversed(
        st.session_state["session_meta"].items()
    ):

        display_name = values["thread_title"]

        is_active = (
            values["thread"]
            == st.session_state.get("active_thread")
        )

        if is_active:
            label = f"🟢 {display_name}"
        else:
            label = f"⚪ {display_name}"

        if st.button(
            label,
            key=f"btn_{values['thread']}"
        ):
            switch_session(values["thread"])

    st.divider()

    # ========================================= FILE UPLOAD =========================================

    st.markdown("""
    <div class='card'>
        <div class='section-title'>📄 Upload Documents</div>
    """, unsafe_allow_html=True)

    sidebar_files = st.file_uploader(
        "Upload PDFs or TXT files",
        accept_multiple_files=True,
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    papers = load_papers(st.session_state["active_thread"])

    if sidebar_files:

        if st.button(
            "Process Documents",
            type="primary"
        ):

            with st.spinner("Processing documents..."):

                for file in sidebar_files:

                    if file.name in papers:
                        continue

                    suffix = Path(file.name).suffix

                    try:

                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=suffix
                        ) as temp_file:

                            temp_file.write(file.read())
                            temp_file_path = temp_file.name

                        docs = document_loader(
                            temp_file_path,
                            file.name
                        )

                        add_paper(
                            session_id=st.session_state["active_thread"],
                            docs=docs
                        )

                        os.remove(temp_file_path)

                    except Exception as e:
                        st.error(f"{file.name}: {e}")

            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ========================================= UNIVERSAL LOADER =========================================

    st.markdown("""
    <div class='card'>
        <div class='section-title'>🌐 Universal Loader</div>
    """, unsafe_allow_html=True)

    input_type = st.radio(
        "Select Input Type",
        [
            "Website URL",
            "ArXiv ID",
            "Paper Title"
        ]
    )

    user_input_loader = st.text_input(
        f"Enter {input_type}"
    )

    submit = st.button(
        "Fetch & Process"
    )

    if submit and user_input_loader:

        with st.spinner("Fetching content..."):

            try:

                if input_type == "Website URL":

                    docs = document_loader(
                        user_input_loader,
                        "website"
                    )

                else:
                    docs = paper_loader(user_input_loader)

                add_paper(
                    session_id=st.session_state["active_thread"],
                    docs=docs
                )

                st.success("Successfully added!")

            except Exception as e:
                st.error(str(e))

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ========================================= KNOWLEDGE BASE =========================================

    st.markdown("### 📚 Active Knowledge Base")

    if papers:

        for paper in papers:

            st.markdown(
                f"""
                <div class='kb-item'>
                    📄 {paper}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("No documents loaded.")

# ========================================= CHAT DISPLAY =========================================

for msg in st.session_state["chats"]:

    if msg["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(
                f"""
                <div class='user-msg'>
                    {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.markdown(
                f"""
                <div class='ai-msg'>
                    {msg['content']}
                </div>
                """,
                unsafe_allow_html=True
            )

# ========================================= CHAT INPUT =========================================

config = {
    "configurable": {
        "thread_id": st.session_state["active_thread"]
    }
}

user_input = st.chat_input(
    "Ask anything about your documents..."
)

if user_input:

    # ========================================= BTW HANDLER =========================================

    is_btw = user_input.strip().lower().startswith("/btw")

    if is_btw:

        query = user_input.strip()[4:].strip()

        with st.expander(
            "✨ Side Channel Interaction",
            expanded=True
        ):

            with st.chat_message(
                "user",
                avatar="👤"
            ):
                st.markdown(query)

            with st.chat_message(
                "assistant",
                avatar="🤖"
            ):

                with st.spinner("Thinking..."):
                    response = handler(query)

                st.info(response)

    else:

        input_state = {
            "messages": [HumanMessage(user_input)],
            "session_id": st.session_state["active_thread"],
            "query": user_input,
            "fallback": False,
            "query_retries": 0,
            "tool_retries": 0,
            "retrieved_docs": [],
            "good_docs": [],
            "context": "",
            "decomposed_query": None,
        }

        # ========================================= USER MESSAGE =========================================

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(
                f"""
                <div class='user-msg'>
                    {user_input}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.session_state["chats"].append({
            "role": "user",
            "content": user_input
        })

        # ========================================= AI RESPONSE =========================================

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            def stream_answer():

                for chunk, meta in graph.stream(
                    input_state,
                    config=config,
                    stream_mode="messages"
                ):

                    if (
                        meta.get("langgraph_node") == "answer"
                        and chunk.content
                    ):
                        yield chunk.content

            ai_message = st.write_stream(stream_answer)

            if not ai_message:

                final_state = graph.get_state(config).values

                ai_message = final_state.get(
                    "answer",
                    "No response generated."
                )

                st.markdown(
                    f"""
                    <div class='ai-msg'>
                        {ai_message}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.session_state["chats"].append({
            "role": "assistant",
            "content": ai_message
        })

        # ========================================= AUTO RENAME CHAT =========================================

        meta = st.session_state["session_meta"][
            st.session_state["active_thread"]
        ]

        if not meta["is_renamed"]:

            title = model.invoke([
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title "
                        "for a research chat session "
                        "based on the user's first message. "
                        "Return only the title."
                    ),
                },
                {
                    "role": "user",
                    "content": user_input[:500],
                },
            ]).content

            meta["thread_title"] = title
            meta["is_renamed"] = True

        save_session(st.session_state["session_meta"])

        st.rerun()