import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import uuid
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent.graph import app
from src.agent.tools import compute_amortization_schedule
from src.memory import list_thread_ids

st.set_page_config(
    page_title="AI Loan Advisory Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    .stChatInput { padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)
NODE_LABELS = {
    "planner": "🧭 Planning — deciding what's needed",
    "researcher": "📚 Researching policy documents",
    "calculator": "🧮 Running EMI calculation",
    "credit": "🪪 Checking credit score",
    "critic": "🔍 Checking evidence quality",
    "synthesizer": "✍️ Writing final answer",
}

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}


def _load_thread(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(config)
    values = snapshot.values if snapshot else {}
    st.session_state.thread_id = thread_id
    st.session_state.messages = values.get("messages", [])
    st.session_state.user_profile = values.get("user_profile", {})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("System Architecture")
    st.markdown("""
    **Local Multi-Agent RAG**
    * **LLM:** `Qwen-2.5-Coder:7b`
    * **Embeddings:** `Nomic-Embed-Text`
    * **Vector DB:** `ChromaDB (k=5)`
    * **Orchestrator:** `LangGraph`
      Planner → Researcher → Calculator → Credit → Critic → Synthesizer
    * **Memory:** `SqliteSaver` (persists across sessions)
    """)
    st.divider()

    st.subheader("Session")
    threads = list_thread_ids()
    options = ["New session"] + threads
    current_index = options.index(st.session_state.thread_id) if st.session_state.thread_id in options else 0
    choice = st.selectbox("Conversation", options, index=current_index)

    if choice != "New session" and choice != st.session_state.thread_id:
        _load_thread(choice)
        st.rerun()

    st.caption(f"Thread ID: `{st.session_state.thread_id}`")

    if st.button("Start New Session", type="primary", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.user_profile = {}
        st.rerun()

    st.divider()
    st.subheader("What I know about you")
    profile = {k: v for k, v in st.session_state.user_profile.items() if v is not None}
    if profile:
        for key, value in profile.items():
            st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")
    else:
        st.caption("Nothing yet — ask a question and I'll build a profile as we go.")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🏦 AI-Powered Loan Advisory Agent")
st.markdown(
    "Ask about loan eligibility, policies, interest rates, or EMI calculations. "
    "Policy answers are retrieved from internal documents; math is computed deterministically, never guessed by the LLM."
)

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    if msg.content.strip():
        with st.chat_message(role):
            st.write(msg.content)

prompt = st.chat_input("E.g., What is the minimum income for a Home Loan?", key="chat_input")

if not st.session_state.messages:
    st.info("👋 Welcome! Try asking one of the common queries below:")
    suggestion = st.pills(
        "Starter queries:",
        [
            "What is the eligibility for a home loan?",
            "Calculate EMI for a 20 Lakh loan at 8.5% for 5 years",
            "Can a college student get a personal loan?",
        ],
        label_visibility="collapsed",
    )
    if suggestion and not prompt:
        prompt = suggestion

# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        status_box = st.status("Agent is working...", expanded=True)
        try:
            # IMPORTANT: only send the NEW message, not the full history. The
            # SqliteSaver checkpointer already holds prior turns for this thread_id;
            # `messages` uses add_messages, which APPENDS. Sending the full
            # st.session_state.messages list here would re-append everything that's
            # already checkpointed, duplicating the conversation on every turn.
            for update in app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="updates",
            ):
                for node_name in update:
                    status_box.write(NODE_LABELS.get(node_name, node_name))
        except Exception as e:
            status_box.update(label="Error", state="error")
            st.error(
                f"Agent error: {e}\n\n"
                "If this is a connection error, make sure Ollama is running "
                "(`ollama serve`) and the required models are pulled."
            )
            st.stop()

        status_box.update(label="Done", state="complete", expanded=False)

        full_state = app.get_state(config).values
        final_response = full_state["messages"][-1]
        st.markdown(final_response.content)

        # Live agent trace, replacing the old post-hoc "thought process" expander
        with st.expander("👁️ View Agent Trace", expanded=False):
            st.markdown(f"**Needs research:** {full_state.get('needs_research')}")
            if full_state.get("needs_research"):
                st.markdown(f"- Search query: `{full_state.get('search_query')}`")
                st.markdown(f"- Retries used: {full_state.get('retry_count', 0)}")
            st.markdown(f"**Needs calculation:** {full_state.get('needs_calculation')}")
            st.markdown(f"**Needs credit check:** {full_state.get('needs_credit_check')}")

        # Structured amortization chart, built from the same params the calculator used —
        # not re-parsed from the markdown table, so it can't drift out of sync.
        calc_params = full_state.get("calc_params")
        if full_state.get("needs_calculation") and calc_params:
            try:
                schedule = compute_amortization_schedule(
                    float(calc_params["principal"]),
                    float(calc_params["rate_pa"]),
                    int(calc_params["tenure_months"]),
                )
                if schedule:
                    df = pd.DataFrame(schedule)
                    with st.expander("📊 Amortization chart", expanded=False):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.line_chart(df.set_index("year")[["principal_paid", "interest_paid"]])
            except (KeyError, ValueError, TypeError):
                pass

    st.session_state.messages = full_state["messages"]
    st.session_state.user_profile = full_state.get("user_profile", st.session_state.user_profile)