import discord
import asyncio
import os
from langchain_core.messages import HumanMessage
from agents.agent_graph import build_pipeline, CATEGORIES

TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

user_sessions = {}


def format_category_list() -> str:
    lines = ["**📂 Danh sách chuyên mục VNExpress:**\n"]
    for slug, display in CATEGORIES.items():
        lines.append(f"  `{slug}` → {display}")
    lines.append(f"\n**Cách dùng:** `!category <tên-chuyên-mục>`")
    lines.append(f"**Ví dụ:** `!category the-thao` → Crawl tin thể thao từ `vnexpress.net/the-thao`")
    return "\n".join(lines)

def get_session(user_id: str) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "thread_id": f"discord-session-{user_id}",
            "graph": None,
            "category": None,
            "articles_count": 0,
        }
    return user_sessions[user_id]


# Discord events
@client.event
async def on_ready():
    print(f"Bot logged in as {client.user}")
    print(f"Available categories: {list(CATEGORIES.keys())}")
    print(f"Waiting discord command ...")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    content = message.content.strip()
    user_id = str(message.author.id)
    session = get_session(user_id)

    # !help — Hiển thị hướng dẫn
    if content.lower() in ["!help", "!h"]:
        help_text = (
            "**🤖 VNExpress AI News Bot — Hướng dẫn sử dụng**\n\n"
            " **Các lệnh:**\n"
            "  `!list` — Xem danh sách chuyên mục\n"
            "  `!category <tên>` — Chọn chuyên mục & crawl tin tức\n"
            "  `!status` — Xem trạng thái hiện tại\n"
            "  `!reset` — Reset cuộc hội thoại\n"
            "  `!help` — Hiển thị hướng dẫn này\n\n"
            "**Hỏi đáp:** Sau khi chọn chuyên mục, gõ câu hỏi bình thường để hỏi AI về tin tức.\n\n"
            "**Ví dụ:**\n"
            "```\n"
            "!category the-thao\n"
            "Hôm nay có tin gì về bóng đá?\n"
            "```"
        )
        await message.channel.send(help_text)
        return

    # !list — Danh sách chuyên mục
    if content.lower() in ["!list", "!categories", "!ds"]:
        await message.channel.send(format_category_list())
        return

    # !status — Trạng thái hiện tại
    if content.lower() in ["!status", "!info"]:
        if session["category"] and session["graph"]:
            cat_display = CATEGORIES.get(session["category"], session["category"])
            status_text = (
                f"**Trạng thái hiện tại:**\n"
                f"  Chuyên mục: **{cat_display}**\n"
                f"  URL: `vnexpress.net/{session['category']}`\n"
                f"  Số bài viết: **{session['articles_count']}**"
            )
        else:
            status_text = (
                "**Trạng thái hiện tại:**\n"
                "  Chưa chọn chuyên mục nào.\n"
                "  Dùng `!category <tên>` hoặc `!list` để xem danh sách."
            )
        await message.channel.send(status_text)
        return

    # !reset — Reset hội thoại
    if content.lower() in ["!reset"]:
        session["thread_id"] = f"discord-session-{user_id}-{id(object())}"
        await message.channel.send("🔄 Đã reset cuộc hội thoại! Bộ nhớ đã được xoá.\n"
                                   f"Chuyên mục vẫn giữ: **{CATEGORIES.get(session['category'], 'Chưa chọn')}**")
        return

    # !category <slug> — Chọn chuyên mục
    if content.lower().startswith("!category") or content.lower().startswith("!cat"):
        parts = content.split(maxsplit=1)
        if len(parts) < 2:
            await message.channel.send(
                "⚠️ Vui lòng chỉ định chuyên mục!\n"
                "**Cách dùng:** `!category <tên-chuyên-mục>`\n"
                "**Ví dụ:** `!category the-thao`\n\n"
                "Gõ `!list` để xem danh sách chuyên mục."
            )
            return

        selected_category = parts[1].strip().lower()

        # Validate category
        if selected_category not in CATEGORIES:
            await message.channel.send(
                f"❌ Chuyên mục `{selected_category}` không tồn tại!\n\n"
                f"{format_category_list()}"
            )
            return

        cat_display = CATEGORIES[selected_category]
        status_msg = await message.channel.send(
            f"Đang lấy dữ liệu chuyên mục **{cat_display}**...\n"
            f"Nguồn: `vnexpress.net/{selected_category}`\n"
            f"Vui lòng đợi trong giây lát..."
        )

        try:
            loop = asyncio.get_event_loop()
            articles, graph = await loop.run_in_executor(
                None,
                lambda: build_pipeline(selected_category)
            )

            # Update session
            session["graph"] = graph
            session["category"] = selected_category
            session["articles_count"] = len(articles)
            # Reset thread_id khi đổi category
            session["thread_id"] = f"discord-session-{user_id}-{selected_category}"

            # article preview
            preview_lines = []
            for i, article in enumerate(articles[:30]):
                title = article.get("title", "Không có tiêu đề")
                url = article.get("url", "")
                preview_lines.append(f"**{i+1}.** {title}")

            more_text = f"\n  ... và {len(articles) - 30} bài viết khác" if len(articles) > 30 else ""

            await status_msg.edit(content=(
                f"✅ Đã lấy dữ liệu thành công chuyên mục **{cat_display}**!\n\n"
                f" **{len(articles)} bài viết** từ `vnexpress.net/{selected_category}`\n\n"
                f"*Bài viết mới nháta:**\n"
                + "\n".join(preview_lines)
                + more_text
                + "\n\nBây giờ bạn có thể hỏi về tin tức!"
            ))

        except Exception as e:
            await status_msg.edit(content=f"❌ Lỗi khi crawl chuyên mục `{selected_category}`: {e}")

        return

    # Regular chat — Hỏi đáp AI
    if session["graph"] is None:
        await message.channel.send(
            "⚠️ Bạn chưa chọn chuyên mục tin tức!\n\n"
            "**Hãy chọn chuyên mục trước khi hỏi:**\n"
            "  `!category the-thao` — Thể thao\n"
            "  `!category kinh-doanh` — Kinh doanh\n"
            "  `!category khoa-hoc-cong-nghe` — Khoa học & Công nghệ\n"
            "  ... và nhiều chuyên mục khác!\n\n"
            "Gõ `!list` để xem đầy đủ danh sách."
        )
        return

    thinking_msg = await message.channel.send("Đang suy nghĩ...")

    config = {
        "configurable": {
            "thread_id": session["thread_id"]
        }
    }

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: session["graph"].invoke(
                {"messages": [HumanMessage(content=content)]},
                config=config
            )
        )

        answer = result["messages"][-1].content

        # Chia tin nhắn thành các phần nhỏ hơn (discord limit 2000 chars)
        if len(answer) > 1900:
            chunks = [answer[i:i+1900] for i in range(0, len(answer), 1900)]
            await thinking_msg.edit(content=chunks[0])
            for chunk in chunks[1:]:
                await message.channel.send(chunk)
        else:
            await thinking_msg.edit(content=answer)

    except Exception as e:
        await thinking_msg.edit(content=f"⚠️ Có lỗi khi xử lý câu hỏi: {e}")

# Run bot
if __name__ == "__main__":
    print("Starting VNExpress AI News Discord Bot...")
    print(f"Available categories: {list(CATEGORIES.keys())}")
    client.run(TOKEN)
