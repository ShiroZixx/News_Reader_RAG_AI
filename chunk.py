from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import json
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from models import gemini_embeddings, embeddings


def create_documents(json_path: str):

   with open(json_path, "r", encoding="utf-8") as f:
      json_data = json.load(f)

   documents = []

   for article in json_data:
      text = f"""Title: {article["title"]}

Description: {article["description"]}

Content:
{article["content"]}
"""
      documents.append(
         Document(
            page_content=text,
            metadata = {
               "url": article["url"],
               "title": article["title"],
               "description": article["description"],
               "date": article["date"],
               "category": article["category"]
            }
         )
      )
   return documents

def semantic_chunking(documents, embeddings):

   semantic_chunker = SemanticChunker(
      embeddings=embeddings,
      breakpoint_threshold_type="standard_deviation",  # hoặc "standard_deviation"
      breakpoint_threshold_amount=1.1,)

   texts = [doc.page_content for doc in documents]
   metadatas = [doc.metadata for doc in documents]

   semantic_chunks = semantic_chunker.create_documents(
      texts,
      metadatas=metadatas
   )

   clean_chunks = []

   for chunk in semantic_chunks:
      if chunk.page_content.strip():
         clean_chunks.append(chunk)

   for i, chunk in enumerate(clean_chunks):
      chunk.metadata["chunk_id"] = i

   return clean_chunks

def paragraphs_chunking(documents):
   for d in documents:
      print(d)


if __name__ == "__main__":
   data_json = "data\\articles.json"

   docs = create_documents(data_json)
   chunks = semantic_chunking(docs, embeddings)
   #paragraphs_chunking(docs)

   print(chunks[:10])