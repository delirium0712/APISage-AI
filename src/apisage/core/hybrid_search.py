"""
Hybrid Search Engine combining ChromaDB vector search with BM25 keyword search
Provides best-in-class search capabilities for RAG systems
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import tempfile

# Disable ChromaDB telemetry at environment level
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_TELEMETRY"] = "False"

try:
    import chromadb
    from chromadb.config import Settings

    # Patch ChromaDB telemetry system to prevent errors
    try:
        from chromadb.telemetry import Telemetry
        # Override the capture method to do nothing
        original_capture = Telemetry.capture
        def silent_capture(self, *args, **kwargs):
            pass
        Telemetry.capture = silent_capture
    except ImportError:
        pass  # Telemetry module doesn't exist in this version

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    import numpy as np

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""

    content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    chunk_id: str = ""
    source_type: str = ""


class HybridSearchEngine:
    """
    Advanced hybrid search engine combining vector similarity and keyword matching.
    Supports both ChromaDB vector search and BM25 keyword search.
    """

    def __init__(
        self,
        collection_name: str = "api_specs",
        embedding_model: str = "BAAI/bge-m3",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the hybrid search engine.

        Args:
            collection_name: Name for the ChromaDB collection
            embedding_model: Model name for sentence transformers
            vector_weight: Weight for vector search scores (0.0-1.0)
            bm25_weight: Weight for BM25 keyword search scores (0.0-1.0)
            persist_directory: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Initialize components
        self.chroma_client = None
        self.collection = None
        self.sentence_transformer = None
        self.bm25_index = None
        self.documents = []
        self.document_metadata = []

        # Setup persistence
        if persist_directory is None:
            persist_directory = tempfile.mkdtemp(prefix="apisage_search_")
        self.persist_directory = persist_directory

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all search components."""

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            try:
                # Disable all ChromaDB telemetry
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                )

                self.chroma_client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=settings,
                )

                # Get or create collection
                try:
                    self.collection = self.chroma_client.get_collection(
                        self.collection_name
                    )
                except Exception:
                    # Create new collection with BGE-M3 embeddings
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name, metadata={"hnsw:space": "cosine"}
                    )

            except Exception as e:
                raise RuntimeError(
                    f"ChromaDB initialization failed: {e}. Please install chromadb and ensure proper configuration."
                )

        # Initialize sentence transformer for embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer(self.embedding_model)
            except Exception as e:
                raise RuntimeError(
                    f"SentenceTransformer initialization failed: {e}. Please install sentence-transformers."
                )

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to both vector and keyword search indices.

        Args:
            documents: List of documents with 'content', 'metadata', and optional 'id'

        Returns:
            True if successful, False otherwise
        """
        if not documents:
            return False

        try:
            # Prepare documents
            texts = []
            metadatas = []
            ids = []

            for i, doc in enumerate(documents):
                content = doc.get("text", doc.get("content", ""))
                metadata = doc.get("metadata", {})
                doc_id = doc.get(
                    "id", f"doc_{i}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
                )

                # Flatten metadata for ChromaDB compatibility (only supports str, int, float, bool)
                flattened_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        flattened_metadata[key] = value
                    else:
                        # Convert complex types to strings
                        flattened_metadata[key] = str(value)

                texts.append(content)
                metadatas.append(flattened_metadata)
                ids.append(doc_id)

            # Add to vector store (ChromaDB)
            if self.collection is not None and self.sentence_transformer is not None:
                try:
                    # Generate embeddings
                    embeddings = self.sentence_transformer.encode(texts).tolist()

                    # Add to ChromaDB
                    self.collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings,
                    )
                except Exception as e:
                    raise RuntimeError(f"Vector store addition failed: {e}")

            # Add to BM25 index
            if BM25_AVAILABLE:
                try:
                    # Tokenize documents for BM25
                    tokenized_docs = [doc.lower().split() for doc in texts]
                    self.bm25_index = BM25Okapi(tokenized_docs)
                    self.documents = texts
                    self.document_metadata = metadatas
                except Exception as e:
                    raise RuntimeError(f"BM25 index creation failed: {e}")

            return True

        except Exception as e:
            raise RuntimeError(f"Error adding documents: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_search: bool = True,
        keyword_search: bool = True,
        min_score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            vector_search: Whether to perform vector search
            keyword_search: Whether to perform keyword search
            min_score_threshold: Minimum score threshold for results

        Returns:
            List of SearchResult objects ranked by hybrid score
        """
        results = []

        try:
            # Perform vector search
            vector_results = []
            if (
                vector_search
                and self.collection is not None
                and self.sentence_transformer is not None
            ):
                vector_results = self._vector_search(
                    query, top_k * 2
                )  # Get more for reranking

            # Perform keyword search
            bm25_results = []
            if keyword_search and self.bm25_index is not None:
                bm25_results = self._bm25_search(
                    query, top_k * 2
                )  # Get more for reranking

            # Combine and rank results
            combined_results = self._combine_results(
                vector_results, bm25_results, top_k
            )

            # Filter by minimum score threshold
            filtered_results = [
                r for r in combined_results if r.hybrid_score >= min_score_threshold
            ]

            return filtered_results[:top_k]

        except Exception as e:
            raise RuntimeError(f"Search error: {e}")

    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform vector similarity search using ChromaDB."""
        try:
            # Generate query embedding
            query_embedding = self.sentence_transformer.encode([query]).tolist()

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self.collection.count()),
            )

            search_results = []
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # Convert distance to similarity score (higher is better)
                similarity_score = max(0.0, 1.0 - distance)

                search_results.append(
                    SearchResult(
                        content=results["documents"][0][i],
                        metadata=(
                            results["metadatas"][0][i] if results["metadatas"] else {}
                        ),
                        vector_score=similarity_score,
                        chunk_id=(
                            results["ids"][0][i] if results["ids"] else f"chunk_{i}"
                        ),
                        source_type="vector_search",
                    )
                )

            return search_results

        except Exception as e:
            raise RuntimeError(f"Vector search error: {e}")

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform BM25 keyword search."""
        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top results
            top_indices = np.argsort(scores)[::-1][:top_k]

            search_results = []
            for idx in top_indices:
                if idx < len(self.documents) and scores[idx] > 0:
                    # Normalize BM25 score (simple normalization)
                    normalized_score = min(1.0, scores[idx] / 10.0)

                    search_results.append(
                        SearchResult(
                            content=self.documents[idx],
                            metadata=(
                                self.document_metadata[idx]
                                if idx < len(self.document_metadata)
                                else {}
                            ),
                            bm25_score=normalized_score,
                            chunk_id=f"bm25_chunk_{idx}",
                            source_type="bm25_search",
                        )
                    )

            return search_results

        except Exception as e:
            raise RuntimeError(f"BM25 search error: {e}")

    def _combine_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Combine and rank results from vector and BM25 search."""

        # Create a mapping of content to results for deduplication
        result_map = {}

        # Add vector results
        for result in vector_results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in result_map:
                result_map[content_hash] = result
            else:
                # Update existing result with vector score
                result_map[content_hash].vector_score = max(
                    result_map[content_hash].vector_score, result.vector_score
                )

        # Add BM25 results
        for result in bm25_results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in result_map:
                result_map[content_hash] = result
            else:
                # Update existing result with BM25 score
                result_map[content_hash].bm25_score = max(
                    result_map[content_hash].bm25_score, result.bm25_score
                )

        # Calculate hybrid scores
        combined_results = []
        for result in result_map.values():
            hybrid_score = (
                self.vector_weight * result.vector_score
                + self.bm25_weight * result.bm25_score
            )
            result.hybrid_score = hybrid_score
            combined_results.append(result)

        # Sort by hybrid score (descending)
        combined_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return combined_results[:top_k]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the search collection."""
        info = {
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "chromadb_available": CHROMADB_AVAILABLE,
            "bm25_available": BM25_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "persist_directory": self.persist_directory,
        }

        if self.collection is not None:
            try:
                info["document_count"] = self.collection.count()
                # Get collection metadata including API spec hash
                collection_metadata = self.collection.metadata or {}
                info["metadata"] = collection_metadata
            except:
                info["document_count"] = 0
                info["metadata"] = {}
        else:
            info["document_count"] = len(self.documents) if self.documents else 0
            info["metadata"] = {}

        return info

    def set_collection_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Set metadata for the collection."""
        try:
            if self.collection is not None:
                # ChromaDB doesn't support modifying collection metadata after creation
                # So we need to store it in a different way or recreate the collection
                # For now, we'll recreate the collection with new metadata
                document_count = self.collection.count()
                if document_count > 0:
                    # If there are documents, we don't want to lose them
                    # Just store in memory for now
                    print(f"Warning: Cannot update collection metadata with {document_count} documents present")
                    return False
                else:
                    # Safe to recreate empty collection with new metadata
                    self.chroma_client.delete_collection(self.collection_name)
                    collection_metadata = {"hnsw:space": "cosine"}
                    # Add creation timestamp for TTL tracking
                    from datetime import datetime
                    collection_metadata["created_timestamp"] = datetime.now().isoformat()
                    collection_metadata.update(metadata)
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name, metadata=collection_metadata
                    )
                    return True
            return False
        except Exception as e:
            print(f"Error setting collection metadata: {e}")
            return False

    def check_collection_ttl(self, ttl_hours: int = 24) -> bool:
        """Check if collection has expired based on TTL and clear if needed."""
        try:
            if self.collection is not None:
                collection_metadata = self.collection.metadata or {}
                created_timestamp = collection_metadata.get("created_timestamp")
                
                if created_timestamp:
                    from datetime import datetime, timedelta
                    created_time = datetime.fromisoformat(created_timestamp)
                    expiry_time = created_time + timedelta(hours=ttl_hours)
                    
                    if datetime.now() > expiry_time:
                        print(f"Collection expired after {ttl_hours} hours, clearing...")
                        self.reset_collection()
                        return True  # Was expired and cleared
                        
            return False  # Not expired
        except Exception as e:
            print(f"Error checking collection TTL: {e}")
            return False

    def reset_collection(self) -> bool:
        """Reset/clear the search collection."""
        try:
            if self.collection is not None:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )

            # Reset BM25 index
            self.bm25_index = None
            self.documents = []
            self.document_metadata = []

            return True
        except Exception as e:
            raise RuntimeError(f"Error resetting collection: {e}")


# Factory function for easy instantiation
def create_hybrid_search(
    collection_name: str = "api_specs",
    embedding_model: str = "BAAI/bge-m3",
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    persist_directory: Optional[str] = None,
) -> HybridSearchEngine:
    """
    Factory function to create a configured HybridSearchEngine.

    Args:
        collection_name: Name for the search collection
        embedding_model: Model for generating embeddings
        vector_weight: Weight for vector search (0.0-1.0)
        bm25_weight: Weight for keyword search (0.0-1.0)
        persist_directory: Directory to persist search data

    Returns:
        Configured HybridSearchEngine instance
    """
    return HybridSearchEngine(
        collection_name=collection_name,
        embedding_model=embedding_model,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        persist_directory=persist_directory,
    )


# Global instance for the main API
_hybrid_search_engine = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    """
    Get or create the global hybrid search engine instance.

    Returns:
        Global HybridSearchEngine instance
    """
    global _hybrid_search_engine
    if _hybrid_search_engine is None:
        _hybrid_search_engine = create_hybrid_search()
    return _hybrid_search_engine
