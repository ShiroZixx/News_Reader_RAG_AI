from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import json
from langchain_core.documents import Document
#from models import local_embeddings

from retrievers import create_retriever, create_bm25_retriever, create_ensemble_retriever

from vectoDB import qdrant_vectodb_setup,qdrant_collection_exists


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
      breakpoint_threshold_type="standard_deviation",  
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
   chunks = []

   texts = [doc.page_content for doc in documents]
   metadatas = [doc.metadata for doc in documents]

   for i in range(len(texts)):
      content_split = texts[i].split("\n\n")

      for j in range(len(content_split)-2):

         chunks.append(
            Document(
               page_content=content_split[j] + content_split[j+1] + content_split[j+2],
               metadata = metadatas[i]
            )
         )
   
   return chunks


if __name__ == "__main__":
   data_json = "data\\articles.json"

   docs = create_documents(data_json)
   #chunks = semantic_chunking(docs, embeddings)
   chunks = paragraphs_chunking(docs)

   qdrant_vectodb = qdrant_vectodb_setup(embeddings=local_embeddings)
   parent_retriever = create_retriever(vectorstore=qdrant_vectodb,
                                       chunked_documents=chunks)
   bm25_retriever = create_bm25_retriever(chunked_documents=chunks)

   already_exists = qdrant_collection_exists()

   if already_exists:
      parent_retriever = create_retriever(
            chunked_documents=chunks,
            vectorstore=qdrant_vectodb,
        )

   list_retrievers = [parent_retriever,bm25_retriever]

   ensemble_retriever = create_ensemble_retriever(list_retrievers)

   result = ensemble_retriever.invoke("điện thoại")
   print(result)