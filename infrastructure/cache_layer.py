"""
Cache Layer for APISage RAG system
High-performance caching with MD5 hash-based keys and optional Redis backend
"""

import json
import hashlib
import pickle
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    size_bytes: int = 0
    hit_rate: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = None


class CacheLayer:
    """
    High-performance cache layer with MD5 hash-based keys.
    Supports both in-memory and Redis backends with TTL and LRU eviction.
    """

    def __init__(
        self,
        backend: str = "memory",  # "memory" or "redis"
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour default TTL
        max_memory_entries: int = 10000,
        enable_compression: bool = True,
        cache_prefix: str = "apisage",
    ):
        """
        Initialize the cache layer.

        Args:
            backend: Cache backend ("memory" or "redis")
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password
            default_ttl: Default TTL in seconds
            max_memory_entries: Maximum entries for in-memory cache
            enable_compression: Enable value compression
            cache_prefix: Prefix for cache keys
        """
        self.backend = backend
        self.default_ttl = default_ttl
        self.max_memory_entries = max_memory_entries
        self.enable_compression = enable_compression
        self.cache_prefix = cache_prefix

        # Redis configuration
        self.redis_config = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "password": redis_password,
        }

        # Initialize backend
        self.redis_client = None
        self.memory_cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # Metrics
        self.metrics = CacheMetrics()

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the chosen cache backend."""
        if self.backend == "redis" and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.redis_config["host"],
                    port=self.redis_config["port"],
                    db=self.redis_config["db"],
                    password=self.redis_config["password"],
                    decode_responses=False,  # We handle binary data
                )
                # Test connection
                self.redis_client.ping()
                print(
                    f"Redis cache backend initialized: {self.redis_config['host']}:{self.redis_config['port']}"
                )
            except Exception as e:
                print(f"Failed to initialize Redis: {e}. Falling back to memory cache.")
                self.backend = "memory"
                self.redis_client = None
        else:
            if self.backend == "redis" and not REDIS_AVAILABLE:
                print("Redis not available. Using memory cache backend.")
            self.backend = "memory"

    def _generate_cache_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """Generate MD5 hash-based cache key."""
        if isinstance(key, dict):
            key_string = json.dumps(key, sort_keys=True, default=str)
        else:
            key_string = str(key)

        hash_key = hashlib.md5(key_string.encode()).hexdigest()
        return f"{self.cache_prefix}:{hash_key}"

    def get(self, key: Union[str, Dict[str, Any]], default: Any = None) -> Any:
        """
        Retrieve value from cache.

        Args:
            key: Cache key (string or dict)
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        cache_key = self._generate_cache_key(key)

        try:
            if self.backend == "redis" and self.redis_client:
                return self._redis_get(cache_key, default)
            else:
                return self._memory_get(cache_key, default)
        except Exception as e:
            print(f"Cache get error: {e}")
            self.metrics.misses += 1
            return default

    def set(
        self,
        key: Union[str, Dict[str, Any]],
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key (string or dict)
            value: Value to cache
            ttl: Time to live in seconds
            metadata: Optional metadata

        Returns:
            True if successful
        """
        cache_key = self._generate_cache_key(key)
        ttl = ttl or self.default_ttl

        try:
            if self.backend == "redis" and self.redis_client:
                return self._redis_set(cache_key, value, ttl, metadata)
            else:
                return self._memory_set(cache_key, value, ttl, metadata)
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted
        """
        cache_key = self._generate_cache_key(key)

        try:
            if self.backend == "redis" and self.redis_client:
                return self._redis_delete(cache_key)
            else:
                return self._memory_delete(cache_key)
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            if self.backend == "redis" and self.redis_client:
                # Delete keys with our prefix
                pattern = f"{self.cache_prefix}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                with self._lock:
                    self.memory_cache.clear()

            # Reset metrics
            self.metrics = CacheMetrics()
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False

    def _redis_get(self, cache_key: str, default: Any) -> Any:
        """Get value from Redis."""
        raw_value = self.redis_client.get(cache_key)
        if raw_value is None:
            self.metrics.misses += 1
            return default

        # Deserialize value
        if self.enable_compression:
            value = pickle.loads(raw_value)
        else:
            value = json.loads(raw_value.decode())

        self.metrics.hits += 1
        return value

    def _redis_set(
        self, cache_key: str, value: Any, ttl: int, metadata: Optional[Dict]
    ) -> bool:
        """Set value in Redis."""
        # Serialize value
        if self.enable_compression:
            serialized = pickle.dumps(value)
        else:
            serialized = json.dumps(value, default=str).encode()

        # Set with TTL
        success = self.redis_client.setex(cache_key, ttl, serialized)

        if success:
            self.metrics.sets += 1
            # Store metadata separately if provided
            if metadata:
                metadata_key = f"{cache_key}:meta"
                meta_serialized = json.dumps(metadata, default=str).encode()
                self.redis_client.setex(metadata_key, ttl, meta_serialized)

        return bool(success)

    def _redis_delete(self, cache_key: str) -> bool:
        """Delete value from Redis."""
        deleted = self.redis_client.delete(cache_key)
        # Also delete metadata
        self.redis_client.delete(f"{cache_key}:meta")

        if deleted:
            self.metrics.deletes += 1

        return bool(deleted)

    def _memory_get(self, cache_key: str, default: Any) -> Any:
        """Get value from memory cache."""
        with self._lock:
            if cache_key not in self.memory_cache:
                self.metrics.misses += 1
                return default

            entry = self.memory_cache[cache_key]

            # Check TTL
            if entry.ttl_seconds:
                age = (datetime.now() - entry.created_at).total_seconds()
                if age > entry.ttl_seconds:
                    del self.memory_cache[cache_key]
                    self.metrics.misses += 1
                    return default

            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            self.metrics.hits += 1
            return entry.value

    def _memory_set(
        self, cache_key: str, value: Any, ttl: int, metadata: Optional[Dict]
    ) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Check if we need to evict entries
            if len(self.memory_cache) >= self.max_memory_entries:
                self._evict_lru_entries()

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl,
                metadata=metadata or {},
            )

            self.memory_cache[cache_key] = entry
            self.metrics.sets += 1
            return True

    def _memory_delete(self, cache_key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                self.metrics.deletes += 1
                return True
            return False

    def _evict_lru_entries(self):
        """Evict least recently used entries from memory cache."""
        with self._lock:
            if len(self.memory_cache) < self.max_memory_entries:
                return

            # Sort by last accessed time
            sorted_entries = sorted(
                self.memory_cache.items(), key=lambda x: x[1].last_accessed
            )

            # Remove oldest 10%
            entries_to_remove = max(1, len(sorted_entries) // 10)

            for i in range(entries_to_remove):
                cache_key = sorted_entries[i][0]
                del self.memory_cache[cache_key]

    def exists(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_cache_key(key)

        try:
            if self.backend == "redis" and self.redis_client:
                return bool(self.redis_client.exists(cache_key))
            else:
                with self._lock:
                    if cache_key not in self.memory_cache:
                        return False

                    # Check TTL
                    entry = self.memory_cache[cache_key]
                    if entry.ttl_seconds:
                        age = (datetime.now() - entry.created_at).total_seconds()
                        if age > entry.ttl_seconds:
                            del self.memory_cache[cache_key]
                            return False

                    return True
        except Exception as e:
            print(f"Cache exists check error: {e}")
            return False

    def get_or_set(
        self,
        key: Union[str, Dict[str, Any]],
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get value from cache or compute and set it.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time to live

        Returns:
            Cached or computed value
        """
        # Try to get existing value
        value = self.get(key)
        if value is not None:
            return value

        # Compute new value
        try:
            value = factory()
            self.set(key, value, ttl)
            return value
        except Exception as e:
            print(f"Factory function error: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.metrics.hits + self.metrics.misses
        hit_rate = self.metrics.hits / total_requests if total_requests > 0 else 0.0

        self.metrics.hit_rate = hit_rate

        stats = {
            "backend": self.backend,
            "redis_available": REDIS_AVAILABLE,
            "metrics": asdict(self.metrics),
            "config": {
                "default_ttl": self.default_ttl,
                "max_memory_entries": self.max_memory_entries,
                "enable_compression": self.enable_compression,
                "cache_prefix": self.cache_prefix,
            },
        }

        if self.backend == "memory":
            stats["memory_entries"] = len(self.memory_cache)
        elif self.backend == "redis" and self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_info"] = {
                    "used_memory": info.get("used_memory"),
                    "connected_clients": info.get("connected_clients"),
                    "uptime_in_seconds": info.get("uptime_in_seconds"),
                }
            except:
                pass

        return stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries (for memory cache)."""
        if self.backend != "memory":
            return 0

        expired_keys = []
        current_time = datetime.now()

        with self._lock:
            for cache_key, entry in self.memory_cache.items():
                if entry.ttl_seconds:
                    age = (current_time - entry.created_at).total_seconds()
                    if age > entry.ttl_seconds:
                        expired_keys.append(cache_key)

            for cache_key in expired_keys:
                del self.memory_cache[cache_key]

        return len(expired_keys)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        return {
            "cache_layer_info": {
                "backend": self.backend,
                "redis_available": REDIS_AVAILABLE,
                "redis_connected": bool(self.redis_client),
                "default_ttl": self.default_ttl,
                "max_memory_entries": self.max_memory_entries,
                "enable_compression": self.enable_compression,
                "cache_prefix": self.cache_prefix,
            },
            "stats": self.get_stats(),
        }


# Factory function for easy instantiation
def create_cache_layer(
    backend: str = "memory",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    default_ttl: int = 3600,
    max_memory_entries: int = 10000,
    cache_prefix: str = "apisage",
) -> CacheLayer:
    """
    Factory function to create a configured CacheLayer.

    Args:
        backend: Cache backend ("memory" or "redis")
        redis_host: Redis server host
        redis_port: Redis server port
        default_ttl: Default TTL in seconds
        max_memory_entries: Max entries for memory cache
        cache_prefix: Prefix for cache keys

    Returns:
        Configured CacheLayer instance
    """
    return CacheLayer(
        backend=backend,
        redis_host=redis_host,
        redis_port=redis_port,
        default_ttl=default_ttl,
        max_memory_entries=max_memory_entries,
        cache_prefix=cache_prefix,
    )


# Global instance for the main API
_cache_layer = None


def get_cache() -> CacheLayer:
    """
    Get or create the global cache layer instance.

    Returns:
        Global CacheLayer instance
    """
    global _cache_layer
    if _cache_layer is None:
        _cache_layer = create_cache_layer()
    return _cache_layer
