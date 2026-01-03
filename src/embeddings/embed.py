"""Embedding generation using HuggingFace sentence-transformers.

Primary: all-mpnet-base-v2 (768 dim, good quality)
Alternative: all-MiniLM-L6-v2 (384 dim, faster)
Fallback: Gemini text-embedding-004 (API)
"""

from typing import Optional
import numpy as np
from tqdm import tqdm

from src.chunking.strategies import Chunk


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers or Gemini API."""
    
    # Available models
    MODELS = {
        "mpnet": {
            "name": "all-mpnet-base-v2",
            "dimension": 768,
            "description": "High quality, balanced speed",
        },
        "minilm": {
            "name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "description": "Faster, smaller dimension",
        },
    }
    
    def __init__(
        self,
        model_key: str = "mpnet",
        use_gpu: bool = True,
        batch_size: int = 32,
        google_api_key: Optional[str] = None,
    ):
        """Initialize embedding generator.
        
        Args:
            model_key: Key from MODELS dict ("mpnet" or "minilm")
            use_gpu: Whether to use GPU if available
            batch_size: Batch size for encoding
            google_api_key: API key for Gemini fallback
        """
        self.model_key = model_key
        self.batch_size = batch_size
        self.google_api_key = google_api_key
        self._model = None
        self._use_local = True
        
        # Set device
        if use_gpu:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        # Get model info
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from {list(self.MODELS.keys())}")
        
        self.model_info = self.MODELS[model_key]
        self.dimension = self.model_info["dimension"]
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading model: {self.model_info['name']}...")
            self._model = SentenceTransformer(
                self.model_info["name"],
                device=self.device,
            )
            print(f"Model loaded on {self.device}")
            self._use_local = True
            
        except Exception as e:
            print(f"Failed to load local model: {e}")
            if self.google_api_key:
                print("Falling back to Gemini API...")
                self._use_local = False
                self.dimension = 768  # Gemini embedding dimension
            else:
                raise RuntimeError(
                    "Could not load local model and no Gemini API key provided"
                )
    
    def _embed_with_gemini(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Gemini API."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.google_api_key)
        
        embeddings = []
        for text in tqdm(texts, desc="Embedding with Gemini"):
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        
        return embeddings
    
    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        # Ensure model is loaded
        _ = self.model if self._use_local else None
        
        if self._use_local:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        else:
            embeddings = self._embed_with_gemini(texts)
            embeddings = np.array(embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query for search.
        
        Uses different task type for query vs document if using Gemini.
        """
        if self._use_local:
            # For local models, no difference between query and document
            return self.model.encode([query], convert_to_numpy=True)[0]
        else:
            # Gemini has specific task type for queries
            import google.generativeai as genai
            
            genai.configure(api_key=self.google_api_key)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query",
            )
            return np.array(result["embedding"])
    
    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> tuple[list[Chunk], np.ndarray]:
        """Generate embeddings for chunks.
        
        Args:
            chunks: List of Chunk objects
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (chunks, embeddings array)
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        return chunks, embeddings
    
    def get_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_info["name"],
            "dimension": self.dimension,
            "device": self.device,
            "using_local": self._use_local,
            "description": self.model_info["description"],
        }

