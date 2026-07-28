import os
import datetime
from collections import defaultdict
from pydantic import BaseModel, Field
from src.config import CHROMA_DB_DIR, RETRIEVER_K
from src.llm_factory import get_embeddings

def get_retriever(provider: str = None, model: str = None):
    """
    Initializes the connection to the configured Vector Database.
    Supports local ChromaDB or managed Pinecone.
    """
    embeddings = get_embeddings(provider=provider, model=model)
    db_provider = os.getenv("VECTOR_DB_PROVIDER", "chroma").lower()
    
    if db_provider == "pinecone":
        try:
            from langchain_pinecone import PineconeVectorStore
            index_name = os.getenv("PINECONE_INDEX_NAME", "loan-policies")
            # Requires PINECONE_API_KEY environment variable to be set
            vector_db = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        except ImportError as e:
            raise ImportError(
                "langchain-pinecone is required for Pinecone vector store. Install it using `pip install langchain-pinecone`."
            ) from e
    else:
        # Default to local ChromaDB
        try:
            from langchain_chroma import Chroma
        except ImportError:
            from langchain_community.vectorstores import Chroma
            
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # search_kwargs={"k": RETRIEVER_K} ensures it returns the top N most relevant chunks
    return vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

class EvidenceChunk(BaseModel):
    source: str
    retrieval_score: float
    document_id: str
    timestamp: str
    trust_score: float
    content: str

def reciprocal_rank_fusion(*doc_lists, k=60):
    """
    Implements Reciprocal Rank Fusion (RRF).
    Takes multiple lists of Documents and merges them based on rank.
    """
    rrf_map = defaultdict(float)
    doc_map = {}
    
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            # Use page_content as a unique key for simplicity if no doc id exists
            doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
            rrf_map[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc
            
    # Sort by RRF score descending
    sorted_docs = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]

def retrieve_loan_context(query: str) -> str:
    """
    Evidence-first RAG pipeline using RRF.
    """
    try:
        retriever = get_retriever()
        
        # Simulate multiple sources by doing semantic search and returning dummy BM25 list for fusion
        # In a real system, you would have a BM25Retriever or second VectorDB.
        semantic_docs = retriever.invoke(query)
        keyword_docs = retriever.invoke(query.split()[0]) if query.split() else semantic_docs
        
        fused_results = reciprocal_rank_fusion(semantic_docs, keyword_docs)
        
        if not fused_results:
            return "No relevant policy documents found in the database."
            
        evidence_chunks = []
        for doc, score in fused_results:
            # Generate the requested evidence structure
            chunk = EvidenceChunk(
                source=doc.metadata.get("source", "Unknown Internal DB"),
                retrieval_score=round(score, 4),
                document_id=doc.metadata.get("id", f"doc_{hash(doc.page_content) % 10000}"),
                timestamp=datetime.datetime.now().isoformat(),
                trust_score=0.95 if "official" in doc.metadata.get("source", "").lower() else 0.8,
                content=doc.page_content
            )
            evidence_chunks.append(chunk)
            
        # Format for the LLM to process
        formatted_context = "EVIDENCE EVALUATION PAYLOAD:\n"
        for idx, chunk in enumerate(evidence_chunks[:RETRIEVER_K]):
            formatted_context += (
                f"--- CHUNK {idx+1} ---\n"
                f"Source: {chunk.source}\n"
                f"Document ID: {chunk.document_id}\n"
                f"Retrieval Score (RRF): {chunk.retrieval_score}\n"
                f"Trust Score: {chunk.trust_score}\n"
                f"Timestamp: {chunk.timestamp}\n"
                f"Content: {chunk.content}\n"
            )
        
        return formatted_context
    except Exception as e:
        return f"[Notice: Could not query policy database due to connection error ({str(e)}). Proceeding with user input.]"