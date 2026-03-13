from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Literal
import torch
import gc
import re

from scraper import save_articles_json
from models import local_qwen_4b, local_embeddings
from documents_chunking import create_documents, paragraphs_chunking
from vectoDB import qdrant_vectodb_setup
from retrievers import create_retriever, create_bm25_retriever, create_ensemble_retriever
from promt import GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT


# Category mapping
CATEGORIES = {
    "tin-tuc-24h": " Tin tức 24h",
    "thoi-su": " Thời sự",
    "the-gioi": " Thế giới",
    "kinh-doanh": " Kinh doanh",
    "khoa-hoc-cong-nghe": " Khoa học & Công nghệ",
    "goc-nhin": " Góc nhìn",
    "bat-dong-san": " Bất động sản",
    "suc-khoe": " Sức khỏe",
    "giai-tri": " Giải trí",
    "the-thao": " Thể thao",
    "phap-luat": " Pháp luật",
    "giao-duc": " Giáo dục",
    "doi-song": " Đời sống",
    "xe": " Xe",
    "du-lich": " Du lịch",
    "tam-su": " Tâm sự",
    "cuoi": " Cười",
}


# Models
response_model = local_qwen_4b
grader_model = local_qwen_4b


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def _get_last_human_question(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return messages[-1].content

def _get_last_context(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content
    return messages[-1].content

def _make_generate_query_or_respond(retriever_tool):
    def generate_query_or_respond(state: MessagesState):
        bound_model = response_model.bind_tools([retriever_tool], tool_choice="auto")
        response = bound_model.invoke(state["messages"])
        return {"messages": [response]}
    return generate_query_or_respond

def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    messages = state["messages"]
    question = _get_last_human_question(messages)
    context = _get_last_context(messages)
    prompt = GRADE_PROMPT.format(question=question, context=context)
    result = (
        grader_model
        .with_structured_output(GradeDocuments, method="json_schema")
        .invoke([{"role": "user", "content": prompt}])
    )
    score = result.binary_score
    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

def rewrite_question(state: MessagesState):
    messages = state["messages"]
    question = _get_last_human_question(messages)
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

def generate_answer(state: MessagesState):
    messages = state["messages"]
    question = _get_last_human_question(messages)
    context = _get_last_context(messages)
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    cleaned_content = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
    cleaned_response = AIMessage(content=cleaned_content)
    return {"messages": [cleaned_response]}


# Build pipeline 
def build_pipeline(category: str):
    """Crawl → chunk → index → build LangGraph workflow."""
    embeddings = local_embeddings

    # 1. Crawl
    articles, data_path = save_articles_json(category=category)

    # 2. Build retriever
    documents = create_documents(data_path)
    chunked_documents = paragraphs_chunking(documents)
    print(f"[{category}] Total chunks: {len(chunked_documents)}")
    cat_name = chunked_documents[0].metadata["category"] if chunked_documents else category

    try:
        vectorstore = qdrant_vectodb_setup(embeddings=embeddings, category=cat_name)
        parent_retriever = create_retriever(vectorstore=vectorstore)
        parent_retriever.add_documents(chunked_documents, ids=None)  # <-- embedding xảy ra ở đây
    finally:
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()

    bm25_retriever = create_bm25_retriever(chunked_documents=chunked_documents)
    list_retrievers = [parent_retriever, bm25_retriever]
    ensemble_retriever = create_ensemble_retriever(list_retrievers)

    # 3. Create tool
    @tool
    def retriever_news(query: str) -> str:
        """Find and return information for user"""
        docs = ensemble_retriever.invoke(query)
        results = []
        for doc in docs:
            url = doc.metadata.get("url", "")
            title = doc.metadata.get("title", "")
            description = doc.metadata.get("description", "")
            text = f"""
Title: {title}
Description: {description}
URL: {url}

Content:
{doc.page_content}
"""
            results.append(text)
        return "\n\n".join(results)

    retriever_tool = retriever_news

    # 4. Build graph
    generate_query_or_respond = _make_generate_query_or_respond(retriever_tool)

    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    graph = workflow.compile(checkpointer=MemorySaver())

    return articles, graph
