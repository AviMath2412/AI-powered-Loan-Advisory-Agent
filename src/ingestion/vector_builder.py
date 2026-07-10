import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import (
    PROCESSED_TEXT_DIR, 
    CHROMA_DB_DIR, 
    EMBEDDING_MODEL, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    OLLAMA_BASE_URL
)

def build_vector_database():
    """
    Reads processed text files, chunks them, embeds them using Ollama, 
    and saves to a local ChromaDB instance.
    """
    print("Starting Vector DB Build Process...")
    
    # 1. Load documents
    documents = []
    if not os.path.exists(PROCESSED_TEXT_DIR) or not os.listdir(PROCESSED_TEXT_DIR):
        print(f"❌ Directory {PROCESSED_TEXT_DIR} is empty. Run pdf_extractor.py first.")
        return

    for filename in os.listdir(PROCESSED_TEXT_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(PROCESSED_TEXT_DIR, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded: {filename}")

    if not documents:
        print("❌ No processed text documents found.")
        return

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split {len(documents)} documents into {len(chunks)} chunks.")

    # 3. Initialize Embeddings via Local Ollama
    print(f"🧠 Initializing {EMBEDDING_MODEL} via Ollama...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # 4. Create and persist ChromaDB
    print(f"💾 Storing vectors in ChromaDB at {CHROMA_DB_DIR}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("✅ Vector database successfully built!")
    return vector_db

def test_retrieval(query: str):
    """
    Tests the created ChromaDB to ensure embeddings and retrieval work correctly.
    """
    print(f"\n🔍 Testing Search Query: '{query}'")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # Retrieve top 3 most similar chunks
    results = vector_db.similarity_search(query, k=3)
    
    print("-" * 60)
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
        print(f"Result {i+1} | Source: {source}")
        print(f"Excerpt: {doc.page_content}\n")
    print("-" * 60)

if __name__ == "__main__":
    build_vector_database()
    
    # Run a quick test query based on the files you've uploaded
    test_retrieval("What is the interest rate for a personal loan?")