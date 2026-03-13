import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import torch
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

"""gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,  
    max_retries=2,
)

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
"""

local_embeddings = HuggingFaceEmbeddings(
    model_name="hiieu/halong_embedding",
    model_kwargs={"device": "cuda"}, 
    encode_kwargs={"batch_size": 128}  
)

local_qwen_4b = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3-4b-2507",  
    temperature=0,
)


