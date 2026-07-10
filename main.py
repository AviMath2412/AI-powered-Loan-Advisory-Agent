from src.agent.graph import app
from langchain_core.messages import HumanMessage

def main():
    """
    A simple terminal interface to interact with the LangGraph Agent.
    This allows us to test the routing and tools before attaching a web UI.
    """
    print("\n" + "="*50)
    print("🏦 AI Loan Advisory Agent (Local RAG) Initialized")
    print("="*50)
    print("Type 'exit' or 'quit' to close the terminal.")
    print("Try asking:")
    print("  1. 'What is the eligibility for an SBI Personal Loan?'")
    print("  2. 'Calculate the EMI for a 15 Lakh loan at 9.5% for 60 months.'\n")
    
    # We maintain the thread of messages in this list
    chat_history = []
    
    while True:
        user_input = input("\n👤 You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Append user message to history
        chat_history.append(HumanMessage(content=user_input))
        
        print("\n🤖 Agent is thinking...")
        
        # Invoke the LangGraph app with the current history
        state = {"messages": chat_history}
        
        # Stream the graph execution to show the thought process
        for event in app.stream(state, stream_mode="values"):
            # Get the latest message generated in the graph
            latest_msg = event["messages"][-1]
            
            # If the AI generated a tool call, let the user know what it's doing
            if latest_msg.type == "ai" and latest_msg.tool_calls:
                for tool_call in latest_msg.tool_calls:
                    print(f"   [System: Calling Tool '{tool_call['name']}' with args: {tool_call['args']}]")
            
        # After the loop finishes, the last message is the final AI response
        final_response = event["messages"][-1]
        print(f"\n💡 Answer:\n{final_response.content}")
        
        # Update our running history with the newly generated messages 
        # (including tool invocations and the final answer) so it remembers context
        chat_history = event["messages"]

if __name__ == "__main__":
    main()