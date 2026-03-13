import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME_BASE = "vnexpress"


def get_qdrant_client() -> QdrantClient:
    url     = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    timeout = 120
    
    if url and api_key:
        return QdrantClient(url=url, api_key=api_key, timeout=timeout)
    return QdrantClient(host="localhost", port=6333)


def _collection_name(category: str) -> str:
    return f"{COLLECTION_NAME_BASE}_{category}"


def qdrant_vectodb_setup(embeddings, category: str) -> QdrantVectorStore:
    client = get_qdrant_client()
    collection_name = _collection_name(category)
    embedding_dim = len(embeddings.embed_query("hello world"))

    existing = [col.name for col in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


