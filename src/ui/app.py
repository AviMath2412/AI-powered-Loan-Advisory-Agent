import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import uuid
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import app
from src.agent.tools import compute_amortization_schedule
from src.memory import list_thread_ids, delete_thread
import importlib
import src.config as config
import src.memory as memory_module
importlib.reload(config)
importlib.reload(memory_module)
import src.observability as observability_module
importlib.reload(observability_module)
tracer = observability_module.tracer
metrics_exporter = observability_module.metrics_exporter
import src.auth as auth
importlib.reload(auth)

import src.agent.utils as agent_utils
importlib.reload(agent_utils)

import src.agent.schemas as schemas_module
importlib.reload(schemas_module)

import src.agent.state as state_module
importlib.reload(state_module)

import src.agent.graph as graph_module
importlib.reload(graph_module)
app = graph_module.app

LLM_PROVIDER = getattr(config, "LLM_PROVIDER", os.getenv("LLM_PROVIDER", "ollama"))
LLM_MODEL = getattr(config, "LLM_MODEL", os.getenv("LLM_MODEL", "qwen2.5-coder:7b"))
from src.cache import response_cache

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
    "validator": "Validating constraints and conflicts",
    "synthesizer": "Drafting response",
    "constraint_checker": "Verifying user constraints",
    "hallucination_guard": "Verifying factual grounding",
    "commit": "Finalizing answer",
}

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "user" not in st.session_state:
    st.session_state.user = None
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None

# --- AUTHENTICATION SCREEN ---
if not st.session_state.user:
    st.title("AI Loan Advisory Agent")
    st.subheader("Login to access your advisory sessions")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True):
            user_info = auth.authenticate_user(login_username, login_password)
            if user_info:
                st.session_state.user = user_info
                st.session_state.jwt_token = auth.generate_jwt(user_info["username"], user_info["role"])
                st.session_state.thread_id = f"{user_info['username']}_{uuid.uuid4().hex[:8]}"
                st.rerun()
            else:
                st.error("Invalid username or password.")
                
    with tab2:
        reg_username = st.text_input("New Username", key="reg_username")
        reg_password = st.text_input("New Password", type="password", key="reg_password")
        if st.button("Register", use_container_width=True):
            if auth.register_user(reg_username, reg_password):
                st.success("Registered successfully! Please login.")
            else:
                st.error("Username already exists or registration failed.")
                
    st.stop() # Stop execution if not logged in
    
if st.session_state.user and not st.session_state.thread_id:
    st.session_state.thread_id = f"{st.session_state.user['username']}_{uuid.uuid4().hex[:8]}"


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
    if st.session_state.user:
        st.header(f"Welcome, {st.session_state.user['username']}!")
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.jwt_token = None
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
    
    st.header("System Architecture")
    st.markdown("""
    **Multi-Agent RAG System**
    * **Vector DB:** `ChromaDB (k=5)`
    * **Orchestrator:** `LangGraph`
      Planner → Researcher → Calculator → Credit → Critic → Synthesizer
    * **Memory:** `SqliteSaver` (persists across sessions)
    """)
    st.divider()

    st.subheader("LLM Configuration")
    provider_options = ["ollama", "openai", "bedrock"]
    prov_idx = provider_options.index(LLM_PROVIDER.lower()) if LLM_PROVIDER.lower() in provider_options else 0
    selected_provider = st.selectbox("LLM Provider", provider_options, index=prov_idx, key="provider_select")

    model_presets = ["qwen2.5-coder:7b", "gemma4:12b", "gemma4:31b-cloud", "gpt-4o-mini", "Custom..."]
    curr_model = LLM_MODEL if LLM_MODEL in model_presets else "qwen2.5-coder:7b"
    model_idx = model_presets.index(curr_model) if curr_model in model_presets else 0
    selected_model_choice = st.selectbox("LLM Model", model_presets, index=model_idx, key="model_choice_select")
    
    if selected_model_choice == "Custom...":
        selected_model = st.text_input("Enter Model Name", value=LLM_MODEL, key="custom_model_input")
    else:
        selected_model = selected_model_choice

    st.session_state.selected_provider = selected_provider
    st.session_state.selected_model = selected_model

    if st.button("⚡ Clear Query Cache", use_container_width=True, help="Clear saved query responses"):
        response_cache.clear()
        st.toast("Query response cache cleared!")

    st.divider()

    st.subheader("Session")
    if st.session_state.user:
        current_username = st.session_state.user["username"]
        threads = memory_module.list_thread_ids(current_username)
        options = ["New session"] + threads
        
        # Determine selectbox index safely
        if st.session_state.thread_id in threads:
            current_index = options.index(st.session_state.thread_id)
        else:
            current_index = 0
            
        def on_session_change():
            choice = st.session_state.session_selector
            if choice == "New session":
                st.session_state.thread_id = f"{st.session_state.user['username']}_{uuid.uuid4().hex[:8]}"
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
    else:
        threads = []
        current_username = ""

    # Start and delete session buttons
    if st.session_state.user:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New", use_container_width=True, help="Start a new session"):
                st.session_state.thread_id = f"{st.session_state.user['username']}_{uuid.uuid4().hex[:8]}"
                st.session_state.messages = []
                st.session_state.user_profile = {}
                st.session_state.uploaded_doc_text = ""
                st.session_state.uploaded_doc_name = ""
                st.rerun()
        with col2:
            if st.session_state.thread_id in threads:
                if st.button(f"🗑️ Delete Session", key=f"del_{st.session_state.thread_id}"):
                    memory_module.delete_thread(st.session_state.thread_id, current_username)
                    st.session_state.thread_id = f"{st.session_state.user['username']}_{uuid.uuid4().hex[:8]}"
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
    # Check Rate Limiter
    if not auth.check_rate_limit(st.session_state.user["username"], limit=30, window_minutes=60):
        st.error("Rate limit exceeded. Please wait before sending more requests.")
        st.stop()
        
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Prepare input state
    input_state = {
        "messages": [HumanMessage(content=prompt)],
        "llm_provider": st.session_state.get("selected_provider", LLM_PROVIDER),
        "llm_model": st.session_state.get("selected_model", LLM_MODEL),
    }
    if st.session_state.get("uploaded_doc_text"):
        input_state["uploaded_doc_text"] = st.session_state.uploaded_doc_text
        input_state["uploaded_doc_name"] = st.session_state.uploaded_doc_name
    else:
        input_state["uploaded_doc_text"] = ""
        input_state["uploaded_doc_name"] = ""

    with st.chat_message("assistant"):
        current_prov = st.session_state.get("selected_provider", LLM_PROVIDER)
        current_mdl = st.session_state.get("selected_model", LLM_MODEL)
        cache_key = response_cache.generate_key(
            prompt=prompt,
            model=current_mdl,
            provider=current_prov,
            uploaded_doc_text=st.session_state.get("uploaded_doc_text", ""),
            user_profile=st.session_state.get("user_profile", {}),
        )
        cached_entry = response_cache.get(cache_key)

        if cached_entry:
            st.caption("⚡ *Served instantly from Response Cache (< 0.05s)*")
            st.markdown(cached_entry["response_content"])
            st.session_state.messages.append(AIMessage(content=cached_entry["response_content"]))
            full_state = {
                "messages": st.session_state.messages,
                "user_profile": st.session_state.get("user_profile", {})
            }
        else:
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

            # Store response in persistent cache
            response_cache.set(cache_key, {"response_content": final_response.content})

        # Live agent trace, replacing the old post-hoc "thought process" expander
        with st.expander("View Agent Trace & Observability", expanded=False):
            st.markdown(f"**Needs research:** {full_state.get('needs_research')}")
            if full_state.get('needs_research'):
                st.markdown(f"- Search query: `{full_state.get('search_query')}`")
                st.markdown(f"- Retries used: {full_state.get('retry_count', 0)}")
            st.markdown(f"**Needs calculation:** {full_state.get('needs_calculation')}")
            st.markdown(f"**Needs credit check:** {full_state.get('needs_credit_check')}")
            
            st.divider()
            st.markdown("#### System Telemetry Metrics")
            metrics = metrics_exporter.get_summary()
            if metrics.get("token_usage"):
                total_cost = sum(m.get("estimated_cost_usd", 0) for m in metrics["token_usage"].values())
                st.markdown(f"**Estimated LLM Cost:** `${total_cost:.4f}`")
                st.markdown("**Token Usage:**")
                st.json(metrics["token_usage"])
            if metrics.get("latencies"):
                st.markdown("**Node Latencies:**")
                st.json(metrics["latencies"])
            if metrics.get("errors"):
                st.markdown("**Errors Recorded:**")
                st.json(metrics["errors"])
            
            st.divider()
            st.markdown(f"**Active Trace Spans:** `{len(tracer.spans)}`")

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