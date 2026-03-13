from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize

store = InMemoryStore()

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500
)

def create_retriever(
        vectorstore, 
        store=store, 
        child_splitter=child_splitter
        ):
    
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    return retriever

def create_bm25_retriever(chunked_documents):
    bm25_retriever = BM25Retriever.from_documents(
        chunked_documents,
        reprocess_func=word_tokenize,
        bm25_variant="plus",
        k=4
    )

    return bm25_retriever

def create_ensemble_retriever(retrievers: list):
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[0.7, 0.3],  # tuỳ chỉnh trọng số
    )
    return ensemble_retriever


