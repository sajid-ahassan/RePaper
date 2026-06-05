import os
import warnings

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings


from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()
model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])
agent_model = ChatOpenAI(model = 'gpt-4o-mini',api_key= os.environ["OPENAI_API_KEY"])
base_embedding = OpenAIEmbeddings(model = 'text-embedding-3-small',api_key= os.environ["OPENAI_API_KEY"])


EMBEDDING_DIM = 1536
embedding_file_store = LocalFileStore("./embedding_cache")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embedding,
    embedding_file_store,
    namespace=base_embedding.model,
    query_embedding_cache=True,
    key_encoder='blake2b'
)



client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)



def get_vectorstore(session_id):
    collection_name = get_collection(session_id)
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
    except UnexpectedResponse as e:
        # Ignore only Qdrant 409 conflict: collection already exists
        if getattr(e, "status_code", None) != 409:
            raise
    return QdrantVectorStore(
        client=client,
        collection_name = collection_name,
        embedding=embeddings
    )


def get_collection(session_id):
    return f"paper_{session_id.replace('-','_')}"


def vs_search(session_id,k : int = 4):
    return get_vectorstore(session_id).as_retriever(search_kwargs = {'k':k})

def add_paper(session_id,docs):
    get_vectorstore(session_id).add_documents(docs)
    
    
def load_papers(session_id):
    collection_name = get_collection(session_id)
    if not client.collection_exists(collection_name):
        return []
    
    papers = set()
    offset = None
    while(True):
        points,offset = client.scroll(
            collection_name,
            offset = offset,
            with_payload = True,
            limit = 100
        )
        for point in points:
            title = point.payload.get('metadata').get('title')
            if title:
                papers.add(title)
        if offset is None:
            break
    return papers
    