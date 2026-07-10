import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import our compiled LangGraph agent
from src.agent.graph import app

# 1. Page Configuration
st.set_page_config(
    page_title="AI Loan Advisory Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sidebar for System Architecture Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100) # Placeholder bank icon
    st.header("System Architecture")
    st.markdown("""
    **Local Agentic RAG**
    * **LLM:** `Qwen-2.5-Coder:7b`
    * **Embeddings:** `Nomic-Embed-Text`
    * **Vector DB:** `ChromaDB`
    * **Orchestrator:** `LangGraph`
    
    **Available Capabilities:**
    1. 📚 Read Policy Documents
    2. 🧮 Calculate EMI Math
    3. 💬 General Conversational Fallback
    """)
    st.divider()
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        st.rerun()

# 3. Main UI Header
st.title("🏦 AI-Powered Loan Advisory Agent")
st.markdown("Ask me anything about loan eligibility, policies, interest rates, or EMI calculations based on internal bank documents.")

# 4. Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display existing chat messages
for msg in st.session_state.messages:
    # Determine the role for the UI icon
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    
    # We only display actual conversational messages (skip tool invocation metadata)
    if msg.content.strip():
        with st.chat_message(role):
            st.write(msg.content)

# --- UPGRADE: Quick Starter Prompts ---
prompt = st.chat_input("E.g., What is the minimum income for an SBI Personal Loan?")

if not st.session_state.messages:
    st.info("👋 Welcome! You can ask me to calculate EMIs, check eligibility, or explain banking fees.")
    suggestion = st.pills(
        "Try asking:", 
        [
            "What is the eligibility for a home loan?", 
            "Calculate EMI for a 20 Lakh loan at 8.5% for 5 years", 
            "What are the pre-closure charges for personal loans?"
        ]
    )
    if suggestion:
        prompt = suggestion

# 6. Handle New User Input
if prompt:
    
    # Render user message
    with st.chat_message("user"):
        st.write(prompt)
        
    # Append user message to internal state
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Generate and render AI response
    with st.chat_message("assistant"):
        # A temporary spinner while the graph executes
        with st.spinner("Agent is thinking..."):
            
            state = {"messages": st.session_state.messages}
            thought_process = []
            
            # Stream the LangGraph execution
            for event in app.stream(state, stream_mode="values"):
                latest_msg = event["messages"][-1]
                
                # Capture tool calls to display in the UI as the "thought process"
                if latest_msg.type == "ai" and latest_msg.tool_calls:
                    for tool in latest_msg.tool_calls:
                        thought_process.append(
                            f"🛠️ **Agent routed to:** `{tool['name']}`\n"
                            f"**Extracted Parameters:** `{tool['args']}`"
                        )
            
            # The final conversational response from the LLM
            final_response = event["messages"][-1]
            
            # 1. Display the Thought Process (Expander)
            if thought_process:
                with st.expander("👁️ View Agent Thought Process", expanded=False):
                    for thought in thought_process:
                        st.info(thought)
                        
            # 2. Display the Final Answer
            st.markdown(final_response.content)
            
    # Save the updated conversation history (including tool responses) back to session state
    st.session_state.messages = event["messages"]