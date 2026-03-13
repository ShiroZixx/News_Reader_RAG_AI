import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

from pydantic import BaseModel, Field
from typing import Literal

from promt import *
from scraper import save_articles_json
from models import local_qwen_4b


from documents_chunking import *
from models import local_embeddings

load_dotenv()

response_model = local_qwen_4b
grader_model = local_qwen_4b

categories = ["tin-tuc-24h", "thoi-su", "the-gioi", "kinh-doanh", "khoa-hoc-cong-nghe", 
              "goc-nhin", "spotlight", "bat-dong-san", "suc-khoe", "giai-tri", 
              "the-thao", "phap-luat", "giao-duc", "doi-song", "xe", "du-lich",
              "anh", "infographic", "y-kien", "tam-su", "cuoi"]

# ================================
# 1️⃣ Crawl dữ liệu
# ================================

_, DATA_PATH = save_articles_json()
#DATA_PATH = "data\\articles_tin_tuc_24h.json"

# ================================
# 2️⃣ Build retriever
# ================================

print("Building retriever...")

def build_retriever(json_path: str):
    """Load data → chunk → index → trả về ensemble retriever"""
    documents = create_documents(json_path)
    chunked_documents = paragraphs_chunking(documents)
    print("Total chunks:", len(chunked_documents))
    category = chunked_documents[0].metadata["category"]

    vectorstore    = qdrant_vectodb_setup(embeddings=local_embeddings, category=category)

    parent_retriever = create_retriever(vectorstore=vectorstore)

    parent_retriever.add_documents(chunked_documents, ids=None)

    bm25_retriever = create_bm25_retriever(chunked_documents=chunked_documents)
    list_retrievers = [parent_retriever,bm25_retriever]
    ensemble_retriever = create_ensemble_retriever(list_retrievers)

    return parent_retriever, bm25_retriever, ensemble_retriever


parent_retriever, bm25_retriever, ensemble_retriever = build_retriever(DATA_PATH)


# ================================
# 3️⃣ Tạo tool cho LangGraph
# ================================

@tool
def retriever_news(query: str) -> str:
    """Find and return information for user"""
    docs = ensemble_retriever.invoke(query)

    results = []

    for doc in docs:

        url = doc.metadata.get("url", "")
        title = doc.metadata.get("title", "")

        text = f"""
Title: {title}
URL: {url}

Content:
{doc.page_content}
"""

        results.append(text)

    return "\n\n".join(results)

retriever_tool = retriever_news

# ================================
# 4️⃣ Bind tool vào model
# ================================

def _get_last_human_question(messages) -> str:
    """Tìm câu hỏi HumanMessage cuối cùng trong lịch sử.
    Khi có memory, messages[0] là tin nhắn đầu tiên trong toàn bộ lịch sử,
    nên cần duyệt ngược để tìm câu hỏi hiện tại."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return messages[-1].content  # fallback


def _get_last_context(messages) -> str:
    """Tìm context từ ToolMessage cuối cùng (kết quả retrieve)."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content
    return messages[-1].content  # fallback


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool], tool_choice="auto")
        .invoke(state["messages"])
    )
    return {"messages": [response]}



class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    messages = state["messages"]
    question = _get_last_human_question(messages)
    context = _get_last_context(messages)

    prompt = GRADE_PROMPT.format(question=question, context=context)

    response = (
        grader_model
        .with_structured_output(GradeDocuments, method="json_schema").invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"



def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = _get_last_human_question(messages)
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState):
    """Generate an answer."""
    messages = state["messages"]
    question = _get_last_human_question(messages)
    context = _get_last_context(messages)
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ================================
# 5️⃣ Build LangGraph workflow
# ================================

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile with memory checkpointer
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# ================================
# 6️⃣ Chat loop
# ================================

def chat():
    print("\nAI News Assistant (VNExpress) — Memory Enabled 🧠")
    print("Type 'exit' to quit")
    print("Type 'reset' to start a new conversation\n")

    # Config với thread_id — cùng thread_id = cùng cuộc hội thoại
    thread_id = "chat-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        if question.lower() == "reset":
            thread_id = f"chat-session-{id(object())}"
            config = {"configurable": {"thread_id": thread_id}}
            print("\n🔄 Đã reset cuộc hội thoại!\n")
            continue

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config,
        )

        answer = result["messages"][-1].content

        print("\nBot:", answer)
        print()


# ================================
# 7️⃣ Run
# ================================

if __name__ == "__main__":
    chat()