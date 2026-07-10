import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import our compiled LangGraph agent
from src.agent.graph import app

st.set_page_config(
    page_title="AI Loan Advisory Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a slightly polished look
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    .stChatInput { padding-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100) # Placeholder bank icon
    st.header("System Architecture")
    st.markdown("""
    **Local Agentic RAG**
    * **LLM:** `Qwen-2.5-Coder:7b`
    * **Embeddings:** `Nomic-Embed-Text`
    * **Vector DB:** `ChromaDB (k=5)`
    * **Orchestrator:** `LangGraph`
    
    **Available Capabilities:**
    1. 📚 Read Policy Documents
    2. 🧮 Calculate EMI Math
    3. 🧠 Contextual Deductions
    """)
    st.divider()
    if st.button("Clear Conversation History", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🏦 AI-Powered Loan Advisory Agent")
st.markdown("Ask me anything about loan eligibility, policies, interest rates, or EMI calculations. I fetch verified data from our internal policies.")

# Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for msg in st.session_state.messages:
    # Determine the role for the UI icon
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    
    # We only display actual conversational messages (skip tool invocation metadata)
    if msg.content.strip():
        with st.chat_message(role):
            st.write(msg.content)

prompt = st.chat_input("E.g., What is the minimum income for a Home Loan?", key="chat_input")

# Quick Starter Prompts (Only show if conversation is empty)
if not st.session_state.messages:
    st.info("👋 Welcome! Try asking one of the common queries below:")
    suggestion = st.pills(
        "Starter queries:", 
        [
            "What is the eligibility for a home loan?", 
            "Calculate EMI for a 20 Lakh loan at 8.5% for 5 years", 
            "Can a college student get a personal loan?"
        ],
        label_visibility="collapsed"
    )
    # If the user clicks a pill and hasn't typed anything else, use the pill text
    if suggestion and not prompt:
        prompt = suggestion

if prompt:
    # Render user message
    with st.chat_message("user"):
        st.write(prompt)
        
    # Append user message to internal state
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Generate and render AI response
    with st.chat_message("assistant"):
        # A temporary spinner while the graph executes
        with st.spinner("Agent is retrieving documents and reasoning..."):
            
            state = {"messages": st.session_state.messages}
            thought_process = []
            
            # Stream the LangGraph execution
            for event in app.stream(state, stream_mode="values"):
                latest_msg = event["messages"][-1]
                
                # Capture tool calls to display in the UI as the "thought process"
                if latest_msg.type == "ai" and getattr(latest_msg, "tool_calls", None):
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