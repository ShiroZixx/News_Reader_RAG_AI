import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

from pydantic import BaseModel, Field
from typing import Literal

from scraper import save_articles_json
from models import local_qwen_4b, local_embeddings
from documents_chunking import create_documents, paragraphs_chunking
from vectoDB import qdrant_vectodb_setup
from retrievers import create_retriever, create_bm25_retriever, create_ensemble_retriever
from promt import GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT

load_dotenv()

# ─────────────────────────────────────────────
# Model config (matching flow.py)
# ─────────────────────────────────────────────
response_model = local_qwen_4b
grader_model = local_qwen_4b

# ─────────────────────────────────────────────
# Category mapping (slug → display name)
# ─────────────────────────────────────────────
CATEGORIES = {
    "tin-tuc-24h": "📰 Tin tức 24h",
    "thoi-su": "🏛️ Thời sự",
    "the-gioi": "🌍 Thế giới",
    "kinh-doanh": "💼 Kinh doanh",
    "khoa-hoc-cong-nghe": "🔬 Khoa học & Công nghệ",
    "goc-nhin": "👁️ Góc nhìn",
    "bat-dong-san": "🏠 Bất động sản",
    "suc-khoe": "🏥 Sức khỏe",
    "giai-tri": "🎬 Giải trí",
    "the-thao": "⚽ Thể thao",
    "phap-luat": "⚖️ Pháp luật",
    "giao-duc": "🎓 Giáo dục",
    "doi-song": "🌸 Đời sống",
    "xe": "🚗 Xe",
    "du-lich": "✈️ Du lịch",
    "tam-su": "💬 Tâm sự",
    "cuoi": "😂 Cười",
}


# ─────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VNExpress AI Reader",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] label {
        color: #c0c0c0 !important;
    }

    /* Article cards */
    .article-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .article-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    .article-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        line-height: 1.4;
    }
    .article-card h4 a {
        color: #90caf9;
        text-decoration: none;
    }
    .article-card h4 a:hover {
        color: #bbdefb;
        text-decoration: underline;
    }
    .article-card p {
        color: #aaa;
        font-size: 0.82rem;
        margin: 0;
        line-height: 1.45;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-ready {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #1a1a1a;
    }
    .status-crawling {
        background: linear-gradient(135deg, #ff9800, #ffc107);
        color: #1a1a1a;
    }

    /* Chat area */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* Spinner area */
    .crawl-progress {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
if "articles" not in st.session_state:
    st.session_state.articles = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_category" not in st.session_state:
    st.session_state.current_category = None
if "crawl_done" not in st.session_state:
    st.session_state.crawl_done = False
if "last_updated" not in st.session_state:
    st.session_state.last_updated = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-chat-1"
if "graph_config" not in st.session_state:
    st.session_state.graph_config = {"configurable": {"thread_id": "streamlit-chat-1"}}


# ─────────────────────────────────────────────
# Graph node functions (matching flow.py)
# ─────────────────────────────────────────────

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


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


def _make_generate_query_or_respond(retriever_tool):
    """Create the generate_query_or_respond node with the given retriever tool."""
    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response based on the current state.
        Given the question, it will decide to retrieve using the retriever tool,
        or simply respond to the user.
        """
        bound_model = response_model.bind_tools([retriever_tool], tool_choice="auto")
        response = bound_model.invoke(state["messages"])
        return {"messages": [response]}
    return generate_query_or_respond


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
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


# ─────────────────────────────────────────────
# Helper: build the LangGraph pipeline
# ─────────────────────────────────────────────
def build_pipeline(category: str):
    """Crawl → chunk → index → build LangGraph workflow (matching flow.py architecture)."""

    # 1. Crawl
    articles, data_path = save_articles_json(category=category)

    # 2. Build retriever (same as flow.py build_retriever)
    documents = create_documents(data_path)
    chunked_documents = paragraphs_chunking(documents)
    print("Total chunks:", len(chunked_documents))
    cat_name = chunked_documents[0].metadata["category"] if chunked_documents else category

    vectorstore = qdrant_vectodb_setup(embeddings=local_embeddings, category=cat_name)
    parent_retriever = create_retriever(vectorstore=vectorstore)
    parent_retriever.add_documents(chunked_documents, ids=None)

    bm25_retriever = create_bm25_retriever(chunked_documents=chunked_documents)
    list_retrievers = [parent_retriever, bm25_retriever]
    ensemble_retriever = create_ensemble_retriever(list_retrievers)

    # 3. Create tool (same as flow.py)
    @tool
    def retriever_news(query: str) -> str:
        """Find and return information for user"""
        docs = ensemble_retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    retriever_tool = retriever_news

    # 4. Build graph (same structure as flow.py)
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
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    graph = workflow.compile(checkpointer=MemorySaver())

    return articles, graph


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Chọn Chuyên Mục")
    st.markdown("---")

    # Category selector
    display_names = list(CATEGORIES.values())
    slugs = list(CATEGORIES.keys())

    selected_display = st.selectbox(
        "Chuyên mục VNExpress",
        display_names,
        index=0,
        help="Chọn chuyên mục tin tức bạn muốn đọc & hỏi đáp",
    )
    selected_slug = slugs[display_names.index(selected_display)]

    st.markdown(f"""
    <div style="margin:0.8rem 0; padding:0.6rem 1rem; background:rgba(102,126,234,0.12); 
                border-radius:10px; border-left:3px solid #667eea;">
        <span style="font-size:0.85rem; color:#b0b0b0;">🔗 URL nguồn:</span><br>
        <a href="https://vnexpress.net/{selected_slug}" target="_blank" 
           style="color:#90caf9; font-size:0.82rem;">
            vnexpress.net/{selected_slug}
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Crawl button
    crawl_btn = st.button(
        "🚀 Bắt đầu Crawl & Phân tích",
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")

    # Status info
    if st.session_state.crawl_done and st.session_state.current_category:
        cat_display = CATEGORIES.get(st.session_state.current_category, st.session_state.current_category)
        last_time = st.session_state.last_updated
        time_str = last_time.strftime("%H:%M:%S — %d/%m/%Y") if last_time else "N/A"
        st.markdown(f"""
        <div style="text-align:center; margin-top:0.5rem;">
            <span class="status-badge status-ready">✅ Sẵn sàng</span>
            <p style="color:#aaa; font-size:0.8rem; margin-top:0.5rem;">
                Đang phục vụ: <strong>{cat_display}</strong><br>
                Số bài viết: <strong>{len(st.session_state.articles)}</strong>
            </p>
            <div style="margin-top:0.6rem; padding:0.5rem 0.8rem; 
                        background:rgba(102,126,234,0.1); border-radius:8px;
                        border:1px solid rgba(102,126,234,0.2);">
                <span style="font-size:0.75rem; color:#90caf9;">🕐 Cập nhật lần cuối</span><br>
                <span style="font-size:0.82rem; color:#e0e0e0; font-weight:500;">{time_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; margin-top:0.5rem;">
            <p style="color:#888; font-size:0.82rem;">
                👈 Chọn chuyên mục và nhấn <strong>Crawl</strong> để bắt đầu
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#555; font-size:0.72rem; padding:0.5rem;">
        Powered by <strong>LangGraph</strong> · <strong>Qwen (Local)</strong> · <strong>Qdrant</strong>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
if st.session_state.last_updated:
    _ts = st.session_state.last_updated.strftime("%H:%M:%S — %d/%m/%Y")
    _header_time = f'<span style="font-size:0.82rem; opacity:0.9;">🕐 Cập nhật lần cuối: {_ts}</span>'
else:
    _header_time = ''

st.markdown(f"""
<div class="main-header">
    <h1>📰 VNExpress AI News Reader</h1>
    <p>Crawl tin tức theo chuyên mục từ VNExpress · Hỏi đáp thông minh với AI (Qwen Local)</p>
    {_header_time}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Crawl handler
# ─────────────────────────────────────────────
if crawl_btn:
    # Reset state
    st.session_state.chat_history = []
    st.session_state.crawl_done = False
    st.session_state.current_category = selected_slug

    with st.status(f"🔄 Đang crawl chuyên mục **{selected_display}**...", expanded=True) as status:
        st.write("⏳ Đang lấy danh sách bài viết từ VNExpress...")
        st.write(f"🔗 URL: `https://vnexpress.net/{selected_slug}`")

        try:
            articles, graph = build_pipeline(selected_slug)

            st.session_state.articles = articles
            st.session_state.graph = graph
            st.session_state.crawl_done = True
            st.session_state.last_updated = datetime.now()

            status.update(
                label=f"✅ Hoàn tất! Đã crawl {len(articles)} bài viết từ **{selected_display}**",
                state="complete",
                expanded=False,
            )
            st.toast(f"🎉 Crawl thành công {len(articles)} bài viết!", icon="✅")

        except Exception as e:
            status.update(label="❌ Lỗi khi crawl!", state="error")
            st.error(f"Có lỗi xảy ra: {e}")


# ─────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────
if st.session_state.crawl_done and st.session_state.articles:
    tab_chat, tab_articles = st.tabs(["💬 Hỏi đáp AI", "📋 Danh sách bài viết"])

    # ── Tab: Chat ──
    with tab_chat:
        st.markdown("#### 🤖 Trò chuyện với AI về tin tức")
        st.caption("🧠 AI có bộ nhớ hội thoại — Hỏi tiếp tục dựa trên câu trả lời trước!")

        # Reset conversation button
        col_info, col_reset = st.columns([3, 1])
        with col_reset:
            if st.button("🔄 Reset hội thoại", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.thread_id = f"streamlit-chat-{id(object())}"
                st.session_state.graph_config = {
                    "configurable": {"thread_id": st.session_state.thread_id}
                }
                st.rerun()

        # Display chat history
        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Nhập câu hỏi của bạn về tin tức..."):
            # Show user message
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Get AI response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Đang tìm kiếm & phân tích..."):
                    try:
                        result = st.session_state.graph.invoke(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=st.session_state.graph_config,
                        )
                        answer = result["messages"][-1].content
                    except Exception as e:
                        answer = f"⚠️ Có lỗi xảy ra khi xử lý câu hỏi: {e}"

                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # ── Tab: Articles list ──
    with tab_articles:
        st.markdown(f"#### 📋 Bài viết đã crawl — {CATEGORIES.get(st.session_state.current_category, '')}")
        st.caption(f"Tổng cộng **{len(st.session_state.articles)}** bài viết")

        for i, article in enumerate(st.session_state.articles):
            title = article.get("title", "Không có tiêu đề")
            desc = article.get("description", "")
            url = article.get("url", "#")
            date = article.get("date", "")
            author = article.get("author", "")

            st.markdown(f"""
            <div class="article-card">
                <h4><a href="{url}" target="_blank">{title}</a></h4>
                <p>{desc[:200]}{'...' if len(desc) > 200 else ''}</p>
                <p style="margin-top:0.4rem; font-size:0.75rem; color:#777;">
                    📅 {date} {'· ✍️ ' + author if author else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align:center; padding:2rem 1rem; 
                    background:linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)); 
                    border-radius:16px; border:1px solid rgba(102,126,234,0.15);">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">📂</div>
            <h3 style="font-size:1rem; margin:0 0 0.5rem 0;">1. Chọn chuyên mục</h3>
            <p style="font-size:0.82rem; color:#999;">
                Chọn chuyên mục tin tức bạn quan tâm từ sidebar bên trái
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align:center; padding:2rem 1rem; 
                    background:linear-gradient(135deg, rgba(0,200,83,0.1), rgba(0,230,118,0.1)); 
                    border-radius:16px; border:1px solid rgba(0,200,83,0.15);">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">🚀</div>
            <h3 style="font-size:1rem; margin:0 0 0.5rem 0;">2. Crawl dữ liệu</h3>
            <p style="font-size:0.82rem; color:#999;">
                Nhấn nút Crawl để thu thập bài viết mới nhất từ VNExpress
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align:center; padding:2rem 1rem; 
                    background:linear-gradient(135deg, rgba(255,152,0,0.1), rgba(255,193,7,0.1)); 
                    border-radius:16px; border:1px solid rgba(255,152,0,0.15);">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">💬</div>
            <h3 style="font-size:1rem; margin:0 0 0.5rem 0;">3. Hỏi đáp AI</h3>
            <p style="font-size:0.82rem; color:#999;">
                Chat với AI để hỏi bất kỳ thông tin nào về các bài viết đã crawl
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show available categories preview
    st.markdown("#### 🗂️ Các chuyên mục có sẵn")
    cols = st.columns(4)
    for idx, (slug, display) in enumerate(CATEGORIES.items()):
        with cols[idx % 4]:
            st.markdown(f"""
            <div style="padding:0.5rem 0.8rem; margin:0.3rem 0; 
                        background:rgba(255,255,255,0.03); border-radius:8px;
                        border:1px solid rgba(255,255,255,0.06); font-size:0.85rem;">
                {display}
            </div>
            """, unsafe_allow_html=True)
