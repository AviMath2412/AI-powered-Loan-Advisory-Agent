import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.config import (
    PROCESSED_TEXT_DIR, 
    CHROMA_DB_DIR, 
    EMBEDDING_MODEL, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
)
from src.llm_factory import get_embeddings

def build_vector_database():
    """
    Reads processed text files, chunks them, embeds them using configured embeddings provider, 
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

    # 3. Initialize Embeddings via LLM Factory
    print(f"🧠 Initializing embeddings ({EMBEDDING_MODEL})...")
    embeddings = get_embeddings()

    # 4. Create and persist VectorDB
    db_provider = os.getenv("VECTOR_DB_PROVIDER", "chroma").lower()
    
    if db_provider == "pinecone":
        print("💾 Storing vectors in Pinecone...")
        try:
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone, ServerlessSpec
            import time
            
            index_name = os.getenv("PINECONE_INDEX_NAME", "loan-policies")
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # Check if index exists, create if not
            if index_name not in pc.list_indexes().names():
                print(f"Creating new Pinecone index '{index_name}' (dimension 768)...")
                pc.create_index(
                    name=index_name,
                    dimension=768, # Dimension for nomic-embed-text
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be initialized
                while not pc.describe_index(index_name).status["ready"]:
                    time.sleep(1)
            
            vector_db = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name
            )
        except ImportError as e:
            raise ImportError("Please install langchain-pinecone and pinecone: pip install langchain-pinecone pinecone") from e
    else:
        print(f"💾 Storing vectors in ChromaDB at {CHROMA_DB_DIR}...")
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
            
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
    
    from src.rag.retriever import get_retriever
    retriever = get_retriever()
    results = retriever.invoke(query)
    
    # Only show top 3 results for test output
    results = results[:3]
    
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