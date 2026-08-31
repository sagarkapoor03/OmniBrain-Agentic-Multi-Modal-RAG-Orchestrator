import os
from typing import List
from app.config import settings

class Embedder:
    """
    Embedding generation supporting:
    1. Gemini API (text-embedding-004 via google-genai) by default
    2. Local SentenceTransformer fallback for offline/testing mode
    """
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY", "")
        self.model_name = settings.GEMINI_EMBEDDING_MODEL
        self._fallback_model = None

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        if not texts:
            return []

        # Try Gemini API via google.genai SDK
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                
                embeddings = []
                for text in texts:
                    res = client.models.embed_content(
                        model=self.model_name,
                        contents=text
                    )
                    # Extract vector float list
                    vec = res.embedding.values if hasattr(res.embedding, "values") else res.embeddings[0].values
                    embeddings.append(vec)
                return embeddings
            except Exception as e:
                # Fallback to google.generativeai if genai fails
                try:
                    import google.generativeai as ggenai
                    ggenai.configure(api_key=self.api_key)
                    embeddings = []
                    for text in texts:
                        res = ggenai.embed_content(
                            model=f"models/{self.model_name}",
                            content=text
                        )
                        embeddings.append(res["embedding"])
                    return embeddings
                except Exception as ex:
                    print(f"[Embedder Warning] Gemini embedding API call failed ({ex}). Using local fallback model.")

        # Fallback to SentenceTransformer
        return self._local_fallback(texts)

    def _local_fallback(self, texts: List[str]) -> List[List[float]]:
        if self._fallback_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[Embedder] Loading local SentenceTransformer model (all-MiniLM-L6-v2)...")
                self._fallback_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[Embedder Error] Could not load SentenceTransformer ({e}). Using hash fallback.")
                return [self._hash_embedding(t) for t in texts]

        embeddings = self._fallback_model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    def _hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Simple pseudo-embedding for testing when no models are available."""
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw_vec = [(float(b) / 255.0) - 0.5 for b in h]
        vec = (raw_vec * (dim // len(raw_vec) + 1))[:dim]
        return vec

embedder = Embedder()
