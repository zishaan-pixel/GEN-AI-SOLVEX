"""
SOLVEX - High-Accuracy ChatGPT-Style AI Agent
Powered by LangChain, OpenRouter, and Streamlit.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional runtime safeguard
    PdfReader = None  # type: ignore[assignment]

# Load environment variables from .env for local runs without overriding host env vars.
load_dotenv(override=False)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PLACEHOLDER_KEYS = {"", "your_openrouter_api_key_here", "sk-or-v1-your-key-here"}

CURATED_MODELS: Dict[str, str] = {
    "OpenRouter Auto (Best Available)": "openrouter/auto",
    "GPT-4o Mini (Fast & Accurate)": "openai/gpt-4o-mini",
    "GPT-4o (Flagship Reasoning)": "openai/gpt-4o",
    "Claude 3.5 Sonnet (Coding & Analysis)": "anthropic/claude-3.5-sonnet",
    "Gemini 2.0 Flash (Fast & Grounded)": "google/gemini-2.0-flash-001",
    "DeepSeek Chat (Strong Logic & Math)": "deepseek/deepseek-chat",
    "Llama 3.3 70B Instruct (Free Tier)": "meta-llama/llama-3.3-70b-instruct:free",
    "Custom Model ID...": "__custom__",
}

FALLBACK_MODELS: List[str] = [
    "openrouter/auto",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
]

STARTER_PROMPTS: List[Tuple[str, str]] = [
    (
        "Explain a Concept",
        "Explain how transformer attention works with a clear, intuitive analogy and minimal math.",
    ),
    (
        "Write & Debug Code",
        "Write a clean Python function to validate and parse JSON safely with unit tests.",
    ),
    (
        "Solve Step-by-Step",
        "A train leaves at 9:15 AM traveling at 72 km/h and another leaves 45 minutes later at 90 km/h on the same track. When do they meet?",
    ),
    (
        "Structured Analysis",
        "Compare REST, GraphQL, and gRPC across latency, caching, typing, and best use cases in a markdown table.",
    ),
]

CORE_SYSTEM_PROMPT = """You are SOLVEX, an accurate, reliable, and user-friendly AI assistant.

Follow these accuracy and communication rules strictly:
1. Verify reasoning step by step before giving a final answer for math, code, or logic questions.
2. Never fabricate facts, statistics, URLs, library APIs, or citations. If something is uncertain or outside available context, state clearly what you know and what you cannot verify.
3. When web search results or uploaded document excerpts are provided in the system context, prioritize them, cross-check carefully, and cite source numbers like [1], [2] where applicable.
4. Keep answers well-structured using clear Markdown headings, bullet points, tables, and language-tagged code blocks when helpful.
5. Be concise when a short answer is best, and thorough when solving complex problems.
Current UTC date: {current_date}
"""


def get_config(key: str, default: str = "") -> str:
    """Resolve configuration from environment variables or Streamlit secrets safely."""
    env_val = os.getenv(key, "").strip()
    if env_val and env_val not in PLACEHOLDER_KEYS:
        return env_val

    try:
        if key in st.secrets:
            secret_val = str(st.secrets[key]).strip()
            if secret_val and secret_val not in PLACEHOLDER_KEYS:
                return secret_val
    except Exception:
        # st.secrets raises if no secrets.toml exists locally; ignore cleanly.
        pass

    return default


def estimate_message_tokens(messages: List[BaseMessage]) -> int:
    """Approximate token count (~4 chars/token) without external tokenizer binaries."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += max(4, len(content) // 4 + 4)
    return total


def create_openrouter_llm(
    api_key: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    streaming: bool = True,
) -> ChatOpenAI:
    """Create a configured LangChain ChatOpenAI client targeting OpenRouter."""
    site_url = get_config("OPENROUTER_SITE_URL", "https://bitm-solvex.streamlit.app")
    app_name = get_config("OPENROUTER_APP_NAME", "SOLVEX")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=2,
        streaming=streaming,
        default_headers={
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        },
    )


def search_web_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch lightweight web search snippets via DuckDuckGo HTML endpoint."""
    results: List[Dict[str, str]] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for item in soup.select(".result"):
            title_el = item.select_one(".result__title a")
            snippet_el = item.select_one(".result__snippet")
            if not title_el:
                continue

            raw_href = title_el.get("href", "")
            parsed = urlparse(raw_href)
            if "uddg" in parse_qs(parsed.query):
                url = unquote(parse_qs(parsed.query)["uddg"][0])
            else:
                url = raw_href

            title = title_el.get_text(" ", strip=True)
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
    except Exception:
        return []

    return results


def extract_uploaded_text(uploaded_file: Any, max_chars: int = 16000) -> Tuple[str, Optional[str]]:
    """Extract readable text from an uploaded document safely."""
    if uploaded_file is None:
        return "", None

    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    raw_bytes = uploaded_file.getvalue()

    try:
        if ext == ".pdf":
            if PdfReader is None:
                return "", "PDF support requires `pypdf`. Please install requirements."
            import io

            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages[:25]]
            text = "\n\n".join(pages).strip()
        else:
            text = raw_bytes.decode("utf-8", errors="replace").strip()

        if not text:
            return "", f"`{filename}` appears to be empty or contains no extractable text."

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[...Truncated to fit model context safely...]"

        return text, None
    except Exception as exc:
        return "", f"Could not read `{filename}`: {exc}"


def format_error_message(exc: Exception, model: str) -> str:
    """Convert raw API/network exceptions into clear, actionable user messages."""
    err_text = str(exc)
    lowered = err_text.lower()

    if "401" in err_text or "unauthorized" in lowered or "invalid api key" in lowered:
        return (
            "Authentication failed (401). Please verify that your `OPENROUTER_API_KEY` "
            "is valid and active at https://openrouter.ai/keys."
        )
    if "402" in err_text or "insufficient" in lowered or "credits" in lowered:
        return (
            "OpenRouter account credits are insufficient for this model (402). "
            "Try selecting a `:free` model, `openrouter/auto`, or top up credits on OpenRouter."
        )
    if "404" in err_text or "not found" in lowered or "no endpoints" in lowered:
        return (
            f"The selected model `{model}` is currently unavailable on OpenRouter (404). "
            "Try enabling **Auto-Fallback** or selecting another model."
        )
    if "429" in err_text or "rate limit" in lowered:
        return (
            f"Rate limit reached on `{model}` (429). Wait a few seconds or enable "
            "**Auto-Fallback** to switch to an available model automatically."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "The request timed out while waiting for OpenRouter. Please try again."
    if "connection" in lowered or "dns" in lowered or "name resolution" in lowered:
        return "Network connection error reaching OpenRouter. Check internet connectivity."

    return f"An error occurred while contacting `{model}`: {err_text}"


def build_langchain_messages(
    history: List[Dict[str, str]],
    user_prompt: str,
    web_results: List[Dict[str, str]],
    doc_name: Optional[str],
    doc_text: str,
    custom_instructions: str,
) -> List[BaseMessage]:
    """Assemble trimmed, grounded LangChain message history for the model."""
    system_parts = [
        CORE_SYSTEM_PROMPT.format(current_date=datetime.utcnow().strftime("%Y-%m-%d"))
    ]

    if custom_instructions.strip():
        system_parts.append(f"Additional user instructions:\n{custom_instructions.strip()}")

    if doc_text:
        system_parts.append(
            f"Uploaded reference file ({doc_name}):\n"
            f"Use this content as primary evidence when relevant.\n"
            f"```\n{doc_text}\n```"
        )

    if web_results:
        formatted_sources = "\n".join(
            f"[{idx}] {r['title']} ({r['url']})\nSnippet: {r['snippet']}"
            for idx, r in enumerate(web_results, start=1)
        )
        system_parts.append(
            "Live web search results (cite as [1], [2], etc. when used):\n"
            + formatted_sources
        )

    messages: List[BaseMessage] = [SystemMessage(content="\n\n".join(system_parts))]

    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_prompt))

    # Keep total context bounded safely so long chats do not exceed context limits.
    trimmed = trim_messages(
        messages,
        max_tokens=12000,
        strategy="last",
        token_counter=estimate_message_tokens,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    return trimmed


def stream_with_fallback(
    api_key: str,
    primary_model: str,
    messages: List[BaseMessage],
    temperature: float,
    max_tokens: int,
    auto_fallback: bool,
) -> Tuple[Generator[str, None, None], str]:
    """Stream response chunks using LangChain, falling back gracefully on model errors."""
    candidate_models = [primary_model]
    if auto_fallback:
        for fb in FALLBACK_MODELS:
            if fb not in candidate_models:
                candidate_models.append(fb)

    last_error: Optional[Exception] = None
    last_model = primary_model

    for idx, model_name in enumerate(candidate_models):
        last_model = model_name
        try:
            llm = create_openrouter_llm(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
            )
            prompt = ChatPromptTemplate.from_messages(
                [MessagesPlaceholder(variable_name="messages")]
            )
            chain = prompt | llm

            stream_iter = chain.stream({"messages": messages})
            first_chunk = next(stream_iter, None)

            def _gen(first: Any, rest: Any) -> Generator[str, None, None]:
                if first is not None and getattr(first, "content", ""):
                    yield str(first.content)
                for chunk in rest:
                    text = getattr(chunk, "content", "")
                    if text:
                        yield str(text)

            return _gen(first_chunk, stream_iter), model_name
        except Exception as exc:
            last_error = exc
            # Do not retry other models for invalid API keys or billing errors.
            msg = str(exc).lower()
            if "401" in msg or "unauthorized" in msg or "402" in msg:
                raise RuntimeError(format_error_message(exc, model_name)) from exc

            # Try non-streaming once on the same model if streaming itself failed.
            try:
                non_stream_llm = create_openrouter_llm(
                    api_key=api_key,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=False,
                )
                result = non_stream_llm.invoke(messages)
                content = str(getattr(result, "content", "") or "")

                def _single(text: str) -> Generator[str, None, None]:
                    if text:
                        yield text

                return _single(content), model_name
            except Exception as invoke_exc:
                last_error = invoke_exc
                if idx == len(candidate_models) - 1:
                    raise RuntimeError(
                        format_error_message(last_error, last_model)
                    ) from last_error

    raise RuntimeError(
        format_error_message(last_error or RuntimeError("Unknown error"), last_model)
    )


def verify_and_refine_answer(
    api_key: str,
    model: str,
    user_prompt: str,
    draft_answer: str,
    web_results: List[Dict[str, str]],
    doc_text: str,
) -> Tuple[str, str]:
    """Run a second LangChain verification pass to catch errors and improve accuracy."""
    verifier_llm = create_openrouter_llm(
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=2200,
        streaming=False,
    )

    context_notes = []
    if doc_text:
        context_notes.append(f"Uploaded file context:\n{doc_text[:6000]}")
    if web_results:
        context_notes.append(
            "Web sources:\n"
            + "\n".join(
                f"[{i}] {r['title']} ({r['url']}): {r['snippet']}"
                for i, r in enumerate(web_results, 1)
            )
        )
    joined_context = "\n\n".join(context_notes) if context_notes else "None provided."

    verify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the accuracy verifier for SOLVEX. Review the draft answer "
                "for arithmetic errors, coding bugs, logical flaws, missing edge cases, or "
                "unsupported claims. Return TWO XML-like sections:\n"
                "<verification_notes>\nBrief bullet list of what you verified or corrected.\n"
                "</verification_notes>\n"
                "<final_answer>\nThe polished, accurate final response for the user.\n"
                "</final_answer>",
            ),
            (
                "human",
                "User Question:\n{question}\n\n"
                "Available Grounding Context:\n{context}\n\n"
                "Draft Answer:\n{draft}",
            ),
        ]
    )

    chain = verify_prompt | verifier_llm | StrOutputParser()
    raw = chain.invoke(
        {"question": user_prompt, "context": joined_context, "draft": draft_answer}
    )

    notes_match = re.search(
        r"<verification_notes>(.*?)</verification_notes>", raw, flags=re.DOTALL
    )
    ans_match = re.search(
        r"<final_answer>(.*?)</final_answer>", raw, flags=re.DOTALL
    )

    notes = notes_match.group(1).strip() if notes_match else "Verified draft response."
    final_ans = ans_match.group(1).strip() if ans_match else draft_answer
    return final_ans, notes


def init_session_state() -> None:
    """Initialize multi-chat session state."""
    if "conversations" not in st.session_state:
        first_id = str(uuid.uuid4())[:8]
        st.session_state.conversations = {
            first_id: {
                "title": "New Chat",
                "created_at": datetime.now().strftime("%H:%M"),
                "messages": [],
            }
        }
        st.session_state.active_chat_id = first_id

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def create_new_chat() -> None:
    """Create and switch to a fresh conversation."""
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[chat_id] = {
        "title": "New Chat",
        "created_at": datetime.now().strftime("%H:%M"),
        "messages": [],
    }
    st.session_state.active_chat_id = chat_id


def export_chat_markdown(chat: Dict[str, Any]) -> str:
    """Export the active conversation as clean Markdown."""
    lines = [
        f"# SOLVEX - {chat['title']}",
        f"_Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]
    for msg in chat["messages"]:
        speaker = "User" if msg["role"] == "user" else "SOLVEX"
        lines.append(f"### {speaker}")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def inject_custom_css() -> None:
    """Apply clean, ChatGPT-inspired UI styling for SOLVEX."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 920px;
        }
        .solvex-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.1rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(16, 163, 127, 0.14), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(128, 128, 128, 0.22);
            margin-bottom: 1.1rem;
        }
        .solvex-title {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 0;
        }
        .solvex-subtitle {
            font-size: 0.85rem;
            opacity: 0.82;
            margin: 0.15rem 0 0 0;
        }
        .solvex-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(16, 163, 127, 0.18);
            border: 1px solid rgba(16, 163, 127, 0.4);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="SOLVEX | AI Assistant",
        page_icon="âš¡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()

    # Sidebar: Conversations, Model Settings, Accuracy & Grounding Controls
    with st.sidebar:
        st.markdown("## âš¡ SOLVEX")
        st.caption("LangChain + OpenRouter AI Agent")

        if st.button("ï¼‹ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()

        st.markdown("### Conversations")
        chat_ids = list(st.session_state.conversations.keys())
        for cid in reversed(chat_ids):
            convo = st.session_state.conversations[cid]
            is_active = cid == st.session_state.active_chat_id
            label = f"{'â–¶ ' if is_active else ''}{convo['title']}"
            if st.button(label, key=f"switch_{cid}", use_container_width=True):
                st.session_state.active_chat_id = cid
                st.rerun()

        st.divider()
        st.markdown("### Model & Accuracy")

        default_env_model = get_config("OPENROUTER_DEFAULT_MODEL", "openrouter/auto")
        model_labels = list(CURATED_MODELS.keys())
        default_index = 0
        for idx, lbl in enumerate(model_labels):
            if CURATED_MODELS[lbl] == default_env_model:
                default_index = idx
                break

        selected_label = st.selectbox("OpenRouter Model", model_labels, index=default_index)
        if CURATED_MODELS[selected_label] == "__custom__":
            selected_model = st.text_input(
                "Custom Model ID",
                value=default_env_model,
                placeholder="e.g. qwen/qwen-2.5-72b-instruct",
            ).strip() or "openrouter/auto"
        else:
            selected_model = CURATED_MODELS[selected_label]

        accuracy_mode = st.radio(
            "Accuracy Mode",
            options=["Balanced (Fast Streaming)", "High Accuracy (Self-Verified)"],
            index=0,
            help="High Accuracy runs a second LangChain verification pass to catch math, code, or logic errors.",
        )

        use_web_search = st.toggle(
            "Live Web Search Grounding",
            value=False,
            help="Searches DuckDuckGo before answering so responses stay grounded in current info.",
        )

        auto_fallback = st.toggle(
            "Auto-Fallback on Model Errors",
            value=True,
            help="Automatically switches to backup OpenRouter models if the selected model rate-limits or fails.",
        )

        with st.expander("Advanced Parameters", expanded=False):
            temperature = st.slider(
                "Temperature (Lower = More Accurate)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
            max_tokens = st.slider(
                "Max Response Tokens",
                min_value=256,
                max_value=4096,
                value=2048,
                step=256,
            )
            custom_instructions = st.text_area(
                "Custom System Instructions",
                placeholder="Optional: e.g., Always explain answers for a beginner.",
                height=80,
            )

        with st.expander("Attach File for Grounded Q&A", expanded=False):
            uploaded_file = st.file_uploader(
                "Upload code, text, CSV, JSON, or PDF",
                type=["txt", "md", "py", "js", "ts", "json", "csv", "log", "pdf"],
            )

        st.divider()
        st.markdown("### API Configuration")
        configured_key = get_config("OPENROUTER_API_KEY", "")
        manual_key = st.text_input(
            "OpenRouter API Key (optional override)",
            type="password",
            placeholder="sk-or-v1-..." if not configured_key else "Loaded from .env / secrets",
            help="Set OPENROUTER_API_KEY in .env or Streamlit Secrets, or paste one here for this session.",
        ).strip()
        active_api_key = manual_key or configured_key

        if active_api_key:
            st.success("OpenRouter API key connected", icon="âœ…")
        else:
            st.warning("Add `OPENROUTER_API_KEY` in `.env`, Secrets, or above.", icon="ðŸ”‘")

        active_chat = st.session_state.conversations[st.session_state.active_chat_id]
        col1, col2 = st.columns(2)
        with col1:
            if active_chat["messages"]:
                st.download_button(
                    "Export .md",
                    data=export_chat_markdown(active_chat),
                    file_name=f"bitm-solvex-{st.session_state.active_chat_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                active_chat["messages"] = []
                active_chat["title"] = "New Chat"
                st.rerun()

    # Extract uploaded document text once per render if present
    doc_text, doc_error = extract_uploaded_text(uploaded_file)
    if doc_error:
        st.warning(doc_error)

    # Header banner
    mode_badge = "Verified Mode" if accuracy_mode.startswith("High") else "Streaming Mode"
    st.markdown(
        f"""
        <div class="solvex-header">
            <div>
                <p class="solvex-title">âš¡ SOLVEX</p>
                <p class="solvex-subtitle">Accurate AI Assistant â€¢ LangChain + OpenRouter ({selected_model})</p>
            </div>
            <span class="solvex-badge">{mode_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_chat = st.session_state.conversations[st.session_state.active_chat_id]

    # Empty-state welcome screen with quick starter prompts (ChatGPT-style)
    if not active_chat["messages"]:
        st.markdown("#### How can SOLVEX help you today?")
        cols = st.columns(2)
        for idx, (label, prompt_text) in enumerate(STARTER_PROMPTS):
            with cols[idx % 2]:
                if st.button(f"**{label}**\n\n{prompt_text}", key=f"starter_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = prompt_text
                    st.rerun()

    # Render existing conversation messages
    for msg in active_chat["messages"]:
        avatar = "ðŸ§‘" if msg["role"] == "user" else "âš¡"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
            if msg.get("sources"):
                with st.expander("Sources used", expanded=False):
                    for idx, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**[{idx}] [{src['title']}]({src['url']})** â€” {src['snippet']}")
            if msg.get("verification_notes"):
                with st.expander("Verification Notes", expanded=False):
                    st.markdown(msg["verification_notes"])

    # Chat input
    typed_prompt = st.chat_input("Ask SOLVEX anything...")
    user_prompt = typed_prompt or st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if not user_prompt:
        return

    if not active_api_key:
        st.error(
            "OpenRouter API key is missing. Add `OPENROUTER_API_KEY` to your `.env` file, "
            "Streamlit Secrets, or paste it into the sidebar input."
        )
        return

    # Append and render user prompt immediately
    if active_chat["title"] == "New Chat":
        clean_title = " ".join(user_prompt.split())
        active_chat["title"] = (clean_title[:32] + "...") if len(clean_title) > 35 else clean_title

    with st.chat_message("user", avatar="ðŸ§‘"):
        st.markdown(user_prompt)

    history_before = list(active_chat["messages"])
    active_chat["messages"].append({"role": "user", "content": user_prompt})

    # Generate assistant response
    with st.chat_message("assistant", avatar="âš¡"):
        web_results: List[Dict[str, str]] = []
        if use_web_search:
            with st.status("Searching the web for up-to-date facts...", expanded=False) as status:
                web_results = search_web_duckduckgo(user_prompt, max_results=5)
                if web_results:
                    status.update(
                        label=f"Found {len(web_results)} web sources",
                        state="complete",
                    )
                else:
                    status.update(
                        label="No web results retrieved; answering from model knowledge",
                        state="complete",
                    )

        lc_messages = build_langchain_messages(
            history=history_before,
            user_prompt=user_prompt,
            web_results=web_results,
            doc_name=uploaded_file.name if uploaded_file else None,
            doc_text=doc_text,
            custom_instructions=custom_instructions,
        )

        verification_notes: Optional[str] = None
        used_model = selected_model

        try:
            if accuracy_mode.startswith("High"):
                with st.status("Drafting and verifying answer for maximum accuracy...", expanded=False) as status:
                    stream_gen, used_model = stream_with_fallback(
                        api_key=active_api_key,
                        primary_model=selected_model,
                        messages=lc_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        auto_fallback=auto_fallback,
                    )
                    draft_answer = "".join(stream_gen)
                    final_answer, verification_notes = verify_and_refine_answer(
                        api_key=active_api_key,
                        model=used_model,
                        user_prompt=user_prompt,
                        draft_answer=draft_answer,
                        web_results=web_results,
                        doc_text=doc_text,
                    )
                    status.update(label="Verification complete", state="complete")
                st.markdown(final_answer)
            else:
                stream_gen, used_model = stream_with_fallback(
                    api_key=active_api_key,
                    primary_model=selected_model,
                    messages=lc_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    auto_fallback=auto_fallback,
                )
                final_answer = st.write_stream(stream_gen)
                if not isinstance(final_answer, str):
                    final_answer = str(final_answer)

            meta_note = f"Model: `{used_model}`"
            if used_model != selected_model:
                meta_note += f" (auto-fallback from `{selected_model}`)"
            if web_results:
                meta_note += f" â€¢ {len(web_results)} web sources"
            if uploaded_file and doc_text:
                meta_note += f" â€¢ File: `{uploaded_file.name}`"
            st.caption(meta_note)

            if web_results:
                with st.expander("Sources used", expanded=False):
                    for idx, src in enumerate(web_results, start=1):
                        st.markdown(f"**[{idx}] [{src['title']}]({src['url']})** â€” {src['snippet']}")

            if verification_notes:
                with st.expander("Verification Notes", expanded=False):
                    st.markdown(verification_notes)

            active_chat["messages"].append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "meta": meta_note,
                    "sources": web_results,
                    "verification_notes": verification_notes,
                }
            )
        except Exception as exc:
            error_msg = str(exc)
            st.error(error_msg)
            active_chat["messages"].append(
                {
                    "role": "assistant",
                    "content": f"âš ï¸ {error_msg}",
                }
            )


if __name__ == "__main__":
    main()

