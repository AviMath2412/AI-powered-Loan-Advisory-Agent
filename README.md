# 🏦 AI-Powered Loan Advisory Agent (Local Agentic RAG)

An intelligent, local agentic RAG application built to assist users with bank loan policies, interest rates, eligibility criteria, and monthly loan payment calculations. It leverages local open-source LLMs and embeddings via **Ollama**, orchestrates tasks using **LangGraph**, and offers both a terminal interface and a modern **Streamlit Web UI**.

---

## 🌟 Key Features

* **📚 Contextual Policy Ingestion & Search:** Automatically parses raw PDF documents (interest rates, terms and conditions, eligibility guidelines) into text chunks, embeds them, and queries the local Chroma vector database with top-k retrieval ($k=5$).
* **🧮 Smart EMI Math Tool:** Dynamically calculates Equated Monthly Installments (EMI), total interest, and total payable amount. Generates a formatted **Yearly Amortization Schedule** table.
* **🧠 Agentic Reasoning with LangGraph:** Uses state-machine routing to intelligently select between search, calculation, or conversational responses. Includes contextual deduction capabilities to reason about profiles not explicitly listed (e.g. students or freelancers) using general eligibility limits.
* **🛡️ Production Guardrails:** Features a custom local LLM JSON fallback parser that catches and processes raw JSON tool outputs to ensure reliable execution on consumer-grade local LLMs.
* **💻 Dual Interface:**
  * **Interactive CLI:** Run queries directly from your shell terminal.
  * **Beautiful Web UI:** A sleek chat interface with interactive starter prompts, a visual agent thought-process inspector, and side-by-side architecture stats.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | `Qwen-2.5-Coder:7b` | Running locally via Ollama (configured with Temperature = 0 for strict calculations) |
| **Embeddings** | `Nomic-Embed-Text` | Local embedding model generating high-quality context vectors |
| **Vector Database** | `ChromaDB` | Embedded vector database with metadata-aware document storage |
| **Orchestration** | `LangGraph` & `LangChain` | Handles agent state, conditional edges, and structured tool routing |
| **Web Dashboard** | `Streamlit` | Frontend web interface with chat interface and custom themes |

---

## 📂 Project Structure

```text
final_project/
├── data/
│   ├── raw_pdfs/            # Put your raw loan policy PDF documents here
│   ├── processed_text/      # Extracted clean text files
│   └── chroma_db/           # Local ChromaDB vector database index files
├── src/
│   ├── agent/
│   │   ├── state.py         # Defines LangGraph state schema
│   │   ├── tools.py         # Retrieval and EMI calculation tools
│   │   └── graph.py         # Agent node definitions and graph compilation
│   ├── ingestion/
│   │   ├── pdf_extractor.py # Extracts & cleans text from raw PDFs
│   │   └── vector_builder.py# Chunks text, generates embeddings, builds vector DB
│   ├── rag/
│   │   └── retriever.py     # Database retrieval and context-formatting logic
│   ├── ui/
│   │   └── app.py           # Streamlit web dashboard application
│   └── config.py            # Global project configurations
├── requirements.txt         # Project dependencies
├── main.py                  # CLI Terminal interface for the agent
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Install & Set Up Ollama
1. Download and install **Ollama** from [ollama.com](https://ollama.ai/).
2. Pull the required models:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text:latest
   ```

### 2. Set Up Virtual Environment & Dependencies
Create a Python virtual environment and install the required libraries:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Ingest PDF Documents & Build Vector DB
1. Place any loan policy PDFs you want to use into the `data/raw_pdfs/` directory.
2. Extract text from the PDFs:
   ```bash
   python src/ingestion/pdf_extractor.py
   ```
3. Generate embeddings and build the Chroma database:
   ```bash
   python src/ingestion/vector_builder.py
   ```

---

## 🖥️ Running the Application

### Option A: Interactive CLI Terminal Interface
Execute the following command to talk to the agent directly in your command line:
```bash
python main.py
```

### Option B: Streamlit Web Dashboard
Run the web application as a module (to automatically resolve dependencies):
```bash
python -m streamlit run src/ui/app.py
```
Open your browser and navigate to **`http://localhost:8501`** (or the port shown in your terminal).

---

## 📝 Example Queries to Try

1. **Policy Queries:**
   * *"What is the eligibility for an SBI Personal Loan?"*
   * *"What are the pre-closure charges for home loans?"*
2. **EMI Calculations:**
   * *"Calculate the EMI for a 15 Lakh loan at 9.5% for 60 months."*
   * *"Show me the yearly amortization schedule for 20 Lakhs at 8.5% for 5 years."*
3. **Contextual Inference:**
   * *"Can a college student get a personal loan?"*
   * *"Can a freelancer apply for a car loan?"*
