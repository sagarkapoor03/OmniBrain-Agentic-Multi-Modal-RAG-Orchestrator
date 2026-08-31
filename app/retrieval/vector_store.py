import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.ingestion.embedder import embedder

class VectorStore:
    """Manages Qdrant vector database storage and retrieval."""
    
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        db_path = settings.QDRANT_PATH
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize local persistent Qdrant client with process lock fallback
        try:
            self.client = QdrantClient(path=db_path)
        except Exception as e:
            print(f"[Qdrant Warning] Path {db_path} is locked by another process ({e}). Falling back to in-memory Qdrant instance.")
            self.client = QdrantClient(":memory:")
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            # We determine dimension dynamically by embedding a test string
            test_emb = embedder.embed_text("test dimension check")
            vector_dim = len(test_emb)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"[Qdrant] Created collection '{self.collection_name}' with vector dimension {vector_dim}.")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds and upserts text chunks into Qdrant."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_documents(texts)

        points = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            point_id = hash(chunk["chunk_id"]) & 0x7FFFFFFFFFFFFFFF  # 64-bit integer ID
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "doc_name": chunk["doc_name"],
                        "page_number": chunk["page_number"],
                        "text": chunk["text"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Performs vector search in Qdrant given a natural language query."""
        query_vector = embedder.embed_text(query)

        if hasattr(self.client, "query_points"):
            res_obj = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            )
            results = res_obj.points
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )

        retrieved = []
        for res in results:
            retrieved.append({
                "score": getattr(res, "score", 0.0),
                "doc_name": res.payload.get("doc_name", ""),
                "page_number": res.payload.get("page_number", 1),
                "chunk_id": res.payload.get("chunk_id", ""),
                "text": res.payload.get("text", "")
            })
        return retrieved

vector_store = VectorStore()
