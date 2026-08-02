# AI-Powered Loan Advisory Agent - Project Context for Report Generation

## Project Overview
The "AI-Powered Loan Advisory Agent" is a local, production-grade Multi-Agent RAG (Retrieval-Augmented Generation) system built entirely from scratch. It handles everything from raw banking policy document collection and text preprocessing to vector space indexing, state-machine orchestration, and interactive web dashboard execution.

Its primary goal is to provide reliable, grounded, and mathematically accurate loan advisory services, avoiding the common pitfalls of LLMs (hallucinations, incorrect math, missing context).

## Technology Stack
- **Orchestrator**: LangGraph (for stateful, multi-agent workflow orchestration)
- **Language Models**: Ollama (`qwen2.5-coder:7b` for reasoning), OpenAI, Bedrock
- **Embedding Models**: `nomic-embed-text:latest`
- **Vector Database**: ChromaDB (local) and Pinecone (managed)
- **Database / Persistence**: PostgreSQL (via `langgraph-checkpoint-postgres`) and SQLite for local development. Used for multi-tenant checkpointer and memory persistence.
- **Frontend / UI**: Streamlit (with JWT Authentication, bcrypt hashing, and sliding-window rate limiting)
- **State Management**: Pydantic (Strict typing for inter-node communication payloads)

## Core Architectural Features
1. **Strict Pydantic Payloads**: Replaces unstructured dictionary states with strictly typed `pydantic.BaseModel` objects across all 12 sequential nodes to prevent downstream state corruption.
2. **Evidence-First Retrieval**: Uses Reciprocal Rank Fusion (RRF) for document retrieval. Each chunk retains explicit metadata (`Source`, `Retrieval Score`, `Timestamp`, `Trust Score`).
3. **Deterministic Numeric Engine**: Firewalls the LLM from doing mental math. Validations, APR comparisons, and compound interest calculations are offloaded to a deterministic Python utility.
4. **Adversarial Hallucination Guard**: A factual editor node verifies that every generated sentence explicitly traces back to retrieved chunks or deterministic tool outputs.
5. **Missing Information Detection**: Cross-domain utility that gracefully refuses requests if required variables are omitted, preventing hallucinated assumptions.
6. **Robust Observability**: Custom JSON-structured logging and real-time telemetry metrics (`Tokens`, `Latency`, `Est. LLM Cost`). Response caching is also implemented for sub-millisecond retrieval of repeated queries.

## Agent Workflow (LangGraph 12-Stage Pipeline)
The system executes a query through a strict state machine consisting of 12 nodes:

1. **Intent Detection**: Analyzes the user query to detect if it needs research or calculation.
2. **Constraint Extraction**: Identifies any strict user constraints (e.g., maximum budget).
3. **Memory Retrieval**: Fetches past conversational history and user profile data.
4. **Document Retrieval (RRF)**: Queries the Vector DB to find relevant policy documents.
5. **Conflict Detection**: Checks for contradictory information within the retrieved documents or user profile.
6. **Missing Info Detection**: Ensures all required fields for calculations or policy evaluation are present.
7. **Confidence Estimation**: Assigns a confidence score to the retrieved context.
8. **Policy Validator**: Strictly validates if the user request meets hard banking policy requirements.
9. **Reasoning Agent**: The core LLM invocation that drafts an initial response based on all aggregated state data.
10. **Grounding Verification**: Verifies the reasoning draft against the retrieved documents to strip out hallucinations.
11. **Response Formatter**: Formats the verified draft into a clean, markdown-friendly response.
12. **Final Response**: Outputs the response back to the user via the UI.

## Testing and CI/CD
The project includes a comprehensive set of adversarial `pytest` suites to verify robustness against hallucinations, contradictions, missing variables, and mathematical impossibilities. The CI/CD pipeline validates dependencies, types (`mypy`), and test correctness on every commit.

## Usage Guide for Claude
*When generating the project report based on this context, please structure it into standard sections: Introduction, System Architecture, Core Components, Technical Innovations, Implementation Details, and Conclusion. Highlight the shift from a basic RAG to a deterministic, multi-agent pipeline using LangGraph, and emphasize the safety mechanisms (Hallucination Guard, Deterministic Math) that make this financial agent production-ready.*
