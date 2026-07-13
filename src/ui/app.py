import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import uuid
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent.graph import app
from src.agent.tools import compute_amortization_schedule
from src.memory import list_thread_ids, delete_thread

st.set_page_config(
    page_title="AI Loan Advisory Agent",
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
    "planner": "Planning — deciding what's needed",
    "researcher": "Researching policy documents",
    "calculator": "Running EMI calculation",
    "credit": "Checking credit score",
    "critic": "Checking evidence quality",
    "synthesizer": "Writing final answer",
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
    
    # Determine selectbox index safely
    if st.session_state.thread_id in threads:
        current_index = options.index(st.session_state.thread_id)
    else:
        current_index = 0

    def on_session_change():
        choice = st.session_state.session_selector
        if choice == "New session":
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.user_profile = {}
            st.session_state.uploaded_doc_text = ""
            st.session_state.uploaded_doc_name = ""
        else:
            _load_thread(choice)

    choice = st.selectbox(
        "Conversation",
        options,
        index=current_index,
        key="session_selector",
        on_change=on_session_change
    )

    st.caption(f"Thread ID: `{st.session_state.thread_id}`")

    # Start and delete session buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New", use_container_width=True, help="Start a new session"):
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.user_profile = {}
            st.session_state.uploaded_doc_text = ""
            st.session_state.uploaded_doc_name = ""
            st.rerun()
    with col2:
        if st.session_state.thread_id in threads:
            if st.button("Delete", use_container_width=True, help="Delete the current session"):
                delete_thread(st.session_state.thread_id)
                st.session_state.thread_id = str(uuid.uuid4())[:8]
                st.session_state.messages = []
                st.session_state.user_profile = {}
                st.session_state.uploaded_doc_text = ""
                st.session_state.uploaded_doc_name = ""
                st.toast("Session deleted successfully!")
                st.rerun()

    # Document upload section
    st.divider()
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "PDF or TXT file",
        type=["pdf", "txt"],
        help="The agent will ground its responses in the contents of this document."
    )

    # Initialize uploaded document session state
    if "uploaded_doc_text" not in st.session_state:
        st.session_state.uploaded_doc_text = ""
    if "uploaded_doc_name" not in st.session_state:
        st.session_state.uploaded_doc_name = ""

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_doc_name:
            with st.spinner("Processing document..."):
                try:
                    file_contents = ""
                    if uploaded_file.type == "application/pdf":
                        import fitz  # PyMuPDF
                        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        pages_text = []
                        for page in doc:
                            pages_text.append(page.get_text())
                        file_contents = "\n".join(pages_text)
                    else:
                        file_contents = uploaded_file.read().decode("utf-8")
                    
                    import re
                    file_contents = re.sub(r'\n+', '\n', file_contents)
                    file_contents = re.sub(r'\s{2,}', ' ', file_contents)
                    
                    st.session_state.uploaded_doc_text = file_contents.strip()
                    st.session_state.uploaded_doc_name = uploaded_file.name
                    st.toast(f"Successfully processed {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
    else:
        st.session_state.uploaded_doc_text = ""
        st.session_state.uploaded_doc_name = ""

    if st.session_state.uploaded_doc_name:
        st.success(f"Loaded: `{st.session_state.uploaded_doc_name}`")

    st.divider()
    st.subheader("User Profile")
    profile = {k: v for k, v in st.session_state.user_profile.items() if v is not None}
    if profile:
        st.markdown(
            '<div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 15px;">',
            unsafe_allow_html=True
        )
        for key, value in profile.items():
            field_name = key.replace('_', ' ').title()
            if isinstance(value, float) or isinstance(value, int):
                if "income" in key.lower():
                    val_str = f"₹ {value:,.2f}"
                else:
                    val_str = str(value)
            else:
                val_str = str(value)
            st.markdown(f"**{field_name}:** `{val_str}`")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("No user profile details extracted yet.")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("AI-Powered Loan Advisory Agent")
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
    st.info("Welcome! Try asking one of the common queries below:")
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

    # Prepare input state
    input_state = {
        "messages": [HumanMessage(content=prompt)]
    }
    if st.session_state.get("uploaded_doc_text"):
        input_state["uploaded_doc_text"] = st.session_state.uploaded_doc_text
        input_state["uploaded_doc_name"] = st.session_state.uploaded_doc_name
    else:
        input_state["uploaded_doc_text"] = ""
        input_state["uploaded_doc_name"] = ""

    with st.chat_message("assistant"):
        status_box = st.status("Agent is working...", expanded=True)
        try:
            # We track whether calculation/credit steps are needed from the planner node's output
            needs_calc = False
            needs_credit = False
            for update in app.stream(
                input_state,
                config=config,
                stream_mode="updates",
            ):
                for node_name, values in update.items():
                    if node_name == "planner":
                        needs_calc = bool(values.get("needs_calculation", False))
                        needs_credit = bool(values.get("needs_credit_check", False))
                    
                    # Conditionally skip displaying calculation/credit nodes if not needed
                    if node_name == "calculator" and not needs_calc:
                        continue
                    if node_name == "credit" and not needs_credit:
                        continue
                    
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
        with st.expander("View Agent Trace", expanded=False):
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
                    with st.expander("Amortization chart", expanded=False):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.line_chart(df.set_index("year")[["principal_paid", "interest_paid"]])
            except (KeyError, ValueError, TypeError):
                pass

    st.session_state.messages = full_state["messages"]
    st.session_state.user_profile = full_state.get("user_profile", st.session_state.user_profile)