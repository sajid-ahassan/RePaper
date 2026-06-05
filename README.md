# RePaper

**RePaper** is an AI-powered research paper assistant built with an agentic RAG pipeline. It helps users upload research papers, load documents from web,arXiv sources, ask questions, retrieve relevant context, and verify research claims through a clean Streamlit chat interface.

The project combines **LangGraph**, **LangChain**, **Qdrant**, **OpenAI**, **Tavily**, and **Streamlit** to create a research-focused AI assistant for academic papers and technical documents.

---

## Features

- Chat-based research paper assistant
- Upload and query PDF/TXT/MD documents
- Load papers from arXiv using paper title or arXiv ID
- Load documents from website URLs
- Agentic RAG workflow using LangGraph
- Query routing between:
  - direct answer generation
  - vector database retrieval
  - web search
  - research claim verification
- Query decomposition for complex questions
- Document relevance filtering before answer generation
- Qdrant vector database integration
- Tavily web search integration
- OpenAI LLM and embedding support
- Cached embeddings using `CacheBackedEmbeddings`
- Streamlit frontend with conversation sidebar
- `/btw` side-channel mode for temporary questions
- Dockerized deployment
- Render deployment support

---

## What is RePaper?

RePaper is designed to make research paper reading and analysis easier.

Instead of manually searching through long PDFs, users can upload documents and ask natural-language questions. RePaper retrieves the most relevant context, filters the retrieved documents, and generates grounded answers based on the available information.

It can also use web search and claim verification workflows when the question requires external or updated information.

---

## `/btw` Side-Channel Mode

RePaper includes a special `/btw` command inspired by temporary side conversations.

Example:

```text
/btw explain what RAG means
```

This allows the user to ask a quick side question without saving it into the main chat history.

This is useful when the user wants clarification without disturbing the main research conversation.

---

## Cached Embeddings

RePaper uses LangChain's `CacheBackedEmbeddings` with a local file store:

```text
embedding_cache/
```

This helps avoid recalculating embeddings for the same content repeatedly.

Benefits:

- Faster repeated document processing
- Reduced OpenAI embedding API calls
- Lower cost during development and testing
- Better local performance


---

## Tech Stack

- **Python**
- **Streamlit** - frontend interface
- **LangChain** - LLM, tools, retrievers, and document loaders
- **LangGraph** - agentic workflow orchestration
- **Qdrant** - vector database
- **OpenAI** - LLM and embeddings
- **Tavily** - web search
- **arXiv** - research paper loading
- **SQLite** - LangGraph checkpointing
- **Docker** - containerization
- **Render** - cloud deployment

---

## Project Structure

```bash
RePaper/
│
├── .deepeval/
├── data/
├── embedding_cache/
├── re_env/
│
├── src/
│   ├── __pycache__/
│   ├── DB/
│   ├── tools/
│   ├── btw_handler.py
│   ├── graph.py
│   ├── models.py
│   ├── paper_loader.py
│   ├── prompts.py
│   └── vector_store.py
│
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── agentic_rag.ipynb
├── Dockerfile
├── eval_checkpoints.db
├── eval_results.json
├── evaluate.py
├── goldens.json
├── main.py
├── requirements.txt
├── session.json
```

---

## Main Files

### `main.py`

Main Streamlit application file.

It handles:

- Chat UI
- Sidebar conversation list
- File uploading
- arXiv, paper, and website loading
- `/btw` side-channel interaction
- LangGraph execution
- Session metadata

### `src/graph.py`

Contains the LangGraph workflow.

It handles:

- Query routing
- Retrieval decision
- Query decomposition
- Tool calling
- Relevance filtering
- Fallback handling
- Final answer generation
- Claim verification

### `src/vector_store.py`

Handles Qdrant vector database logic.

It includes:

- Qdrant client setup
- Collection creation
- Vector search
- Document insertion
- Loaded paper tracking
- Cached OpenAI embeddings

### `src/paper_loader.py`

Handles document loading from:

- PDFs
- text files
- markdown files
- websites
- arXiv papers

### `src/tools/`

Contains tools used by the agent, such as:

- Vector store search
- Web search

### `src/btw_handler.py`

Handles `/btw` side-channel interactions that are not saved to the main chat history.

### `evaluate.py`

Used for evaluating the RAG pipeline.

---

## How It Works

RePaper follows an agentic RAG workflow.

1. The user uploads a paper or loads a document.
2. The document is split into chunks.
3. Chunks are embedded using OpenAI embeddings.
4. Embeddings are cached locally with `CacheBackedEmbeddings`.
5. Vectors are stored in Qdrant.
6. The user asks a question.
7. LangGraph decides the best route:
   - direct answer
   - document retrieval
   - web search
   - claim verification
8. Relevant documents are retrieved and filtered.
9. The final answer is generated using the available context.

---

## Agentic Workflow

The system is designed to choose the best response path depending on the user query.

### Direct Generation

Used for general questions that can be answered safely without retrieval.

### Retrieval

Used when the user asks something that requires information from uploaded documents or other loaded sources.

### Query Decomposition

If the query is complex, the system can break it into smaller retrieval-focused sub-queries.

### Claim Verification

Used when the user provides a research claim that needs verification against web or academic sources.

### Fallback

If the system cannot find relevant information after multiple retrieval attempts, it returns a fallback response instead of hallucinating.

---

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=
TAVILY_API_KEY=
QDRANT_API_KEY=
QDRANT_URL=
```

Never commit your real `.env` file to GitHub.

Use `.env.example` to show required variables safely.


---

## Docker Hub

Pull and run:

```bash
docker pull sajid2680/repaper:latest
docker run --env-file .env -p 8501:8501 sajid2680/repaper:latest
```

---

## Future Improvements

- User authentication
- Better arXiv fallback handling
- Multi-paper comparison
- Citation-aware answer generation
- Export chat as PDF or Markdown
- Admin dashboard for uploaded papers
- Support for more file types
- Better document deduplication
- Production-ready storage for session metadata


---

## License

This project is open-source and available under the MIT License.

---

## Author

Developed by **Sajid**.

