import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, RETRIEVER_K

def get_retriever():
    """Initializes the connection to our existing local ChromaDB."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # search_kwargs={"k": RETRIEVER_K} ensures it returns the top N most relevant chunks
    return vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

def retrieve_loan_context(query: str) -> str:
    """
    Takes a natural language query, searches the vector database, 
    and formats the resulting chunks into a single readable string.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant policy documents found in the database."
        
    # Format the retrieved chunks clearly so the LLM knows where the data came from
    formatted_context = "\n\n---\n\n".join(
        [f"Source Document: {doc.metadata.get('source', 'Unknown')}\nExcerpt:\n{doc.page_content}" for doc in docs]
    )
    return formatted_context