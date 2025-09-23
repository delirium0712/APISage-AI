"""
Semantic Cache for RAG systems
High-performance semantic caching using sentence transformers and similarity matching
"""

import json
import hashlib
import pickle
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import tempfile
import threading

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    np = None


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""

    query: str
    query_embedding: List[float]
    response: str
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    ttl_seconds: Optional[int] = None
    cache_key: str = ""


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    total_size_mb: float
    average_similarity_threshold: float


class SemanticCache:
    """
    Advanced semantic cache using sentence transformers for similarity matching.
    Provides intelligent caching with semantic understanding and TTL support.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
        default_ttl_hours: int = 24,
        cache_dir: Optional[str] = None,
        enable_persistence: bool = True,
    ):
        """
        Initialize the semantic cache.

        Args:
            embedding_model: Model name for sentence transformers
            similarity_threshold: Minimum similarity score for cache hits
            max_cache_size: Maximum number of cached entries
            default_ttl_hours: Default TTL in hours for cache entries
            cache_dir: Directory to persist cache data
            enable_persistence: Whether to persist cache to disk
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.default_ttl_seconds = default_ttl_hours * 3600
        self.enable_persistence = enable_persistence

        # Initialize cache storage
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.embedding_keys: List[str] = []

        # Statistics
        self.hit_count = 0
        self.miss_count = 0

        # Thread safety
        self._lock = threading.RLock()

        # Setup cache directory
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="apisage_cache_")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "semantic_cache.pkl")

        # Initialize sentence transformer
        self.sentence_transformer = None
        self.semantic_available = SEMANTIC_CACHE_AVAILABLE
        if self.semantic_available:
            try:
                self.sentence_transformer = SentenceTransformer(self.embedding_model)
                print(f"Semantic cache initialized with model: {self.embedding_model}")
            except Exception as e:
                print(f"Failed to initialize sentence transformer: {e}")
                self.semantic_available = False

        # Load existing cache if persistence is enabled
        if self.enable_persistence:
            self._load_cache()

    def get(
        self, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """
        Retrieve a cached response for the given query.

        Args:
            query: The query to search for
            metadata: Optional metadata for filtering

        Returns:
            Tuple of (response, metadata, similarity_score) if found, None otherwise
        """
        if not self.semantic_available or not self.sentence_transformer:
            return None

        with self._lock:
            try:
                # Clean expired entries
                self._clean_expired_entries()

                if not self.cache_entries:
                    self.miss_count += 1
                    return None

                # Generate query embedding
                query_embedding = self.sentence_transformer.encode([query])[0]

                # Find most similar entry
                best_match = self._find_best_match(query_embedding, metadata)

                if best_match is None:
                    self.miss_count += 1
                    return None

                cache_key, similarity_score = best_match
                entry = self.cache_entries[cache_key]

                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self.hit_count += 1

                return entry.response, entry.metadata, similarity_score

            except Exception as e:
                print(f"Error retrieving from cache: {e}")
                self.miss_count += 1
                return None

    def put(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
    ) -> bool:
        """
        Store a query-response pair in the cache.

        Args:
            query: The query string
            response: The response to cache
            metadata: Optional metadata to associate with the entry
            ttl_hours: Time to live in hours (overrides default)

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.semantic_available or not self.sentence_transformer:
            return False

        with self._lock:
            try:
                # Generate cache key and embedding
                cache_key = self._generate_cache_key(query, metadata)
                query_embedding = self.sentence_transformer.encode([query])[0].tolist()

                # Determine TTL
                ttl_seconds = None
                if ttl_hours is not None:
                    ttl_seconds = ttl_hours * 3600
                elif self.default_ttl_seconds > 0:
                    ttl_seconds = self.default_ttl_seconds

                # Create cache entry
                entry = CacheEntry(
                    query=query,
                    query_embedding=query_embedding,
                    response=response,
                    metadata=metadata or {},
                    timestamp=datetime.now(),
                    ttl_seconds=ttl_seconds,
                    cache_key=cache_key,
                )

                # Check if we need to evict entries
                if len(self.cache_entries) >= self.max_cache_size:
                    self._evict_entries()

                # Store the entry
                self.cache_entries[cache_key] = entry
                self._rebuild_embeddings_matrix()

                # Persist if enabled
                if self.enable_persistence:
                    self._save_cache()

                return True

            except Exception as e:
                print(f"Error storing in cache: {e}")
                return False

    def _find_best_match(
        self,
        query_embedding: np.ndarray,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, float]]:
        """Find the best matching cache entry."""
        if self.embeddings_matrix is None or len(self.embedding_keys) == 0:
            return None

        try:
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[
                0
            ]

            # Find entries above threshold
            valid_indices = []
            for i, similarity in enumerate(similarities):
                if similarity >= self.similarity_threshold:
                    cache_key = self.embedding_keys[i]
                    entry = self.cache_entries.get(cache_key)

                    if entry and self._matches_metadata_filter(entry, metadata_filter):
                        valid_indices.append((i, cache_key, similarity))

            if not valid_indices:
                return None

            # Return the best match
            best_idx, best_key, best_similarity = max(valid_indices, key=lambda x: x[2])
            return best_key, best_similarity

        except Exception as e:
            print(f"Error finding best match: {e}")
            return None

    def _matches_metadata_filter(
        self, entry: CacheEntry, metadata_filter: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if entry matches metadata filter."""
        if metadata_filter is None:
            return True

        for key, value in metadata_filter.items():
            if entry.metadata.get(key) != value:
                return False

        return True

    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix for similarity calculations."""
        try:
            if not self.cache_entries:
                self.embeddings_matrix = None
                self.embedding_keys = []
                return

            embeddings = []
            keys = []

            for cache_key, entry in self.cache_entries.items():
                embeddings.append(entry.query_embedding)
                keys.append(cache_key)

            self.embeddings_matrix = np.array(embeddings)
            self.embedding_keys = keys

        except Exception as e:
            print(f"Error rebuilding embeddings matrix: {e}")

    def _generate_cache_key(
        self, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique cache key for the query and metadata."""
        key_data = {"query": query, "metadata": metadata or {}}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _clean_expired_entries(self):
        """Remove expired cache entries."""
        try:
            current_time = datetime.now()
            expired_keys = []

            for cache_key, entry in self.cache_entries.items():
                if entry.ttl_seconds is not None:
                    expiry_time = entry.timestamp + timedelta(seconds=entry.ttl_seconds)
                    if current_time > expiry_time:
                        expired_keys.append(cache_key)

            for key in expired_keys:
                del self.cache_entries[key]

            if expired_keys:
                self._rebuild_embeddings_matrix()

        except Exception as e:
            print(f"Error cleaning expired entries: {e}")

    def _evict_entries(self):
        """Evict entries using LRU policy."""
        try:
            if len(self.cache_entries) < self.max_cache_size:
                return

            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.cache_entries.items(),
                key=lambda x: x[1].last_accessed or x[1].timestamp,
            )

            # Remove oldest 10% of entries
            entries_to_remove = max(1, len(sorted_entries) // 10)

            for i in range(entries_to_remove):
                cache_key = sorted_entries[i][0]
                del self.cache_entries[cache_key]

            self._rebuild_embeddings_matrix()

        except Exception as e:
            print(f"Error evicting entries: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        try:
            cache_data = {
                "entries": {k: asdict(v) for k, v in self.cache_entries.items()},
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "config": {
                    "embedding_model": self.embedding_model,
                    "similarity_threshold": self.similarity_threshold,
                    "max_cache_size": self.max_cache_size,
                    "default_ttl_seconds": self.default_ttl_seconds,
                },
            }

            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            print(f"Error saving cache: {e}")

    def _load_cache(self):
        """Load cache from disk."""
        try:
            if not os.path.exists(self.cache_file):
                return

            with open(self.cache_file, "rb") as f:
                cache_data = pickle.load(f)

            # Restore cache entries
            for cache_key, entry_data in cache_data.get("entries", {}).items():
                # Convert timestamp strings back to datetime objects
                entry_data["timestamp"] = datetime.fromisoformat(
                    entry_data["timestamp"]
                )
                if entry_data.get("last_accessed"):
                    entry_data["last_accessed"] = datetime.fromisoformat(
                        entry_data["last_accessed"]
                    )

                self.cache_entries[cache_key] = CacheEntry(**entry_data)

            # Restore statistics
            self.hit_count = cache_data.get("hit_count", 0)
            self.miss_count = cache_data.get("miss_count", 0)

            # Rebuild embeddings matrix
            self._rebuild_embeddings_matrix()

            print(f"Loaded {len(self.cache_entries)} entries from cache")

        except Exception as e:
            print(f"Error loading cache: {e}")
            # Reset cache on load error
            self.cache_entries = {}
            self.embeddings_matrix = None
            self.embedding_keys = []

    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            try:
                self.cache_entries.clear()
                self.embeddings_matrix = None
                self.embedding_keys = []
                self.hit_count = 0
                self.miss_count = 0

                if self.enable_persistence and os.path.exists(self.cache_file):
                    os.remove(self.cache_file)

                return True
            except Exception as e:
                print(f"Error clearing cache: {e}")
                return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0

            # Estimate cache size
            cache_size_mb = 0.0
            try:
                if os.path.exists(self.cache_file):
                    cache_size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
            except:
                pass

            return CacheStats(
                total_entries=len(self.cache_entries),
                hit_count=self.hit_count,
                miss_count=self.miss_count,
                hit_rate=hit_rate,
                total_size_mb=cache_size_mb,
                average_similarity_threshold=self.similarity_threshold,
            )

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        stats = self.get_stats()

        return {
            "semantic_cache_available": self.semantic_available,
            "embedding_model": self.embedding_model,
            "similarity_threshold": self.similarity_threshold,
            "max_cache_size": self.max_cache_size,
            "default_ttl_seconds": self.default_ttl_seconds,
            "cache_dir": self.cache_dir,
            "enable_persistence": self.enable_persistence,
            "stats": asdict(stats),
        }

    def update_similarity_threshold(self, new_threshold: float) -> bool:
        """Update the similarity threshold for cache hits."""
        if 0.0 <= new_threshold <= 1.0:
            with self._lock:
                self.similarity_threshold = new_threshold
                return True
        return False


# Factory function for easy instantiation
def create_semantic_cache(
    embedding_model: str = "BAAI/bge-m3",
    similarity_threshold: float = 0.85,
    max_cache_size: int = 1000,
    default_ttl_hours: int = 24,
    cache_dir: Optional[str] = None,
) -> SemanticCache:
    """
    Factory function to create a configured SemanticCache.

    Args:
        embedding_model: Model for generating embeddings
        similarity_threshold: Minimum similarity for cache hits
        max_cache_size: Maximum number of cached entries
        default_ttl_hours: Default TTL for cache entries
        cache_dir: Directory to persist cache data

    Returns:
        Configured SemanticCache instance
    """
    return SemanticCache(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        max_cache_size=max_cache_size,
        default_ttl_hours=default_ttl_hours,
        cache_dir=cache_dir,
    )


# Global instance for singleton pattern
_semantic_cache_instance = None


def get_semantic_cache() -> SemanticCache:
    """
    Get or create the global semantic cache instance.
    
    Returns:
        Global SemanticCache instance
    """
    global _semantic_cache_instance
    if _semantic_cache_instance is None:
        _semantic_cache_instance = create_semantic_cache()
    return _semantic_cache_instance
