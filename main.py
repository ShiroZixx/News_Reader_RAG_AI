from nodes import *
from scraper import save_articles_json
from vectoDB import *

from uuid import uuid4

categories = ["tin-tuc-24h", "thoi-su", "the-gioi", "kinh-doanh", "khoa-hoc-cong-nghe", 
              "goc-nhin", "spotlight", "bat-dong-san", "suc-khoe", "giai-tri", 
              "the-thao", "phap-luat", "giao-duc", "doi-song", "xe", "du-lich",
              "anh", "infographic", "y-kien", "tam-su", "cuoi"]


response_model = gemini_model.bind_tools([retriever_tool])

vdb_is_exist = qdrant_collection_exists()

def build_retriever(json_path: str):
    """Load data → chunk → index → trả về ensemble retriever"""
    vdb_is_exist = qdrant_collection_exists()

    documents = create_documents(json_path)
    chunked_documents = paragraphs_chunking(documents)
    print("Total chunks:", len(chunked_documents))
    category = chunked_documents[0].metadata["category"]

    vectorstore    = qdrant_vectodb_setup(embeddings=local_embeddings, category=category)
    parent_retriever = create_retriever(vectorstore=vectorstore)

    if not vdb_is_exist:

        
        uuids = [str(uuid4()) for _ in range(len(chunked_documents))]
        
        batch_size = 128
        for i in range(0, len(chunked_documents), batch_size):
            batch_docs = chunked_documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]

            parent_retriever.add_documents(batch_docs, ids=batch_ids)
    else:
        print("Vector DB already exists → skip indexing")


    bm25_retriever = create_bm25_retriever(chunked_documents=chunked_documents)
    list_retrievers = [parent_retriever,bm25_retriever]
    ensemble_retriever = create_ensemble_retriever(list_retrievers)

    return parent_retriever, bm25_retriever, ensemble_retriever

# ================================
# 5️⃣ Build LangGraph workflow
# ================================

workflow = StateGraph(MessagesState)

workflow.add_node(
    "generate_query_or_respond",
    lambda state: generate_query_or_respond(state, response_model, retriever_tool),
)

workflow.add_node("retrieve", ToolNode([retriever_tool]))

workflow.add_node("rewrite_question", rewrite_question)

workflow.add_node(
    "generate_answer",
    lambda state: generate_answer(state, response_model),
)

workflow.add_edge(START, "generate_query_or_respond")

# quyết định gọi tool hay không
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# kiểm tra document
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()


# ================================
# 6️⃣ Chat loop
# ================================

def chat():
    print("\nAI News Assistant (VNExpress)")
    print("Type 'exit' to quit\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            }
        )

        answer = result["messages"][-1].content

        print("\nBot:", answer)
        print()


