import uuid

from langchain_core.messages import HumanMessage

from src.agent.graph import app


def main():
    """
    A simple terminal interface to interact with the LangGraph Agent.
    Each run gets its own thread_id, so conversation memory persists in the
    SqliteSaver checkpoint if you restart with the same thread_id (printed below).
    """
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 50)
    print("🏦 AI Loan Advisory Agent (Local Multi-Agent RAG) Initialized")
    print(f"   Thread ID: {thread_id}")
    print("=" * 50)
    print("Type 'exit' or 'quit' to close the terminal.")
    print("Try asking:")
    print("  1. 'What is the eligibility for an SBI Personal Loan?'")
    print("  2. 'Calculate the EMI for a 15 Lakh loan at 9.5% for 60 months.'\n")

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        print("\n🤖 Agent is thinking...")

        # Only the new message is sent — the checkpointer supplies prior turns
        # for this thread_id automatically.
        for update in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name in update:
                print(f"   [Agent: running node '{node_name}']")

        final_state = app.get_state(config).values
        final_response = final_state["messages"][-1]
        print(f"\n💡 Answer:\n{final_response.content}")


if __name__ == "__main__":
    main()