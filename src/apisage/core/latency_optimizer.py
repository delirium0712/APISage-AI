"""
Latency Optimization Module for APISage
Implements advanced caching, request batching, and performance optimizations
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import lz4.frame
from functools import wraps, lru_cache
from collections import defaultdict, deque
import threading

import structlog
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 1
    size_bytes: int = 0
    ttl: Optional[float] = None
    compression_ratio: float = 1.0

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class BatchRequest:
    """Batched request for processing"""
    request_id: str
    data: Any
    callback: asyncio.Future
    timestamp: float
    priority: int = 0


class MultiTierCache:
    """
    Multi-tier caching system with L1 (memory), L2 (compressed), and semantic layers
    Implements LRU, LFU, and adaptive replacement policies
    """

    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        semantic_cache_size: int = 500,
        compression_threshold: int = 1024,
        ttl_seconds: int = 3600
    ):
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.semantic_cache_size = semantic_cache_size
        self.compression_threshold = compression_threshold
        self.default_ttl = ttl_seconds

        # L1 Cache (Hot data, uncompressed)
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l1_access_order = deque(maxlen=l1_size)

        # L2 Cache (Warm data, compressed)
        self.l2_cache: Dict[str, CacheEntry] = {}
        self.l2_access_order = deque(maxlen=l2_size)

        # Semantic cache for similarity-based retrieval
        self.semantic_cache: Dict[str, Tuple[np.ndarray, CacheEntry]] = {}
        self.semantic_embeddings: List[np.ndarray] = []
        self.semantic_keys: List[str] = []

        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "semantic_hits": 0,
            "compressions": 0,
            "evictions": 0
        }

        # Lock for thread safety
        self._lock = threading.RLock()

    def _compute_key(self, data: Any) -> str:
        """Compute cache key from data"""
        if isinstance(data, str):
            key_data = data
        elif isinstance(data, dict):
            key_data = json.dumps(data, sort_keys=True)
        else:
            key_data = str(data)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _compress(self, data: Any) -> bytes:
        """Compress data using LZ4"""
        serialized = pickle.dumps(data)
        compressed = lz4.frame.compress(serialized)
        return compressed

    def _decompress(self, data: bytes) -> Any:
        """Decompress LZ4 data"""
        decompressed = lz4.frame.decompress(data)
        return pickle.loads(decompressed)

    def get(self, key: str, semantic_similarity: Optional[float] = None) -> Optional[Any]:
        """
        Get value from cache with multi-tier lookup

        Args:
            key: Cache key
            semantic_similarity: Threshold for semantic similarity matching (0-1)
        """
        with self._lock:
            # Check L1 cache
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    self.stats["l1_hits"] += 1
                    self._promote_to_l1(key, entry)
                    return entry.value
                else:
                    del self.l1_cache[key]

            # Check L2 cache
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not entry.is_expired():
                    # Decompress and promote to L1
                    entry.value = self._decompress(entry.value)
                    entry.update_access()
                    self.stats["l2_hits"] += 1
                    self._promote_to_l1(key, entry)
                    return entry.value
                else:
                    del self.l2_cache[key]

            # Check semantic cache if similarity threshold provided
            if semantic_similarity and self.semantic_cache:
                similar_entry = self._find_semantic_match(key, semantic_similarity)
                if similar_entry:
                    self.stats["semantic_hits"] += 1
                    return similar_entry.value

            self.stats["l1_misses"] += 1
            self.stats["l2_misses"] += 1
            return None

    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        embedding: Optional[np.ndarray] = None
    ):
        """
        Put value into cache with automatic tiering

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            embedding: Vector embedding for semantic caching
        """
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl or self.default_ttl
            )

            # Calculate size
            try:
                serialized = pickle.dumps(value)
                entry.size_bytes = len(serialized)
            except:
                entry.size_bytes = 0

            # Add to semantic cache if embedding provided
            if embedding is not None:
                self._add_to_semantic_cache(key, entry, embedding)

            # Decide tier based on size
            if entry.size_bytes > self.compression_threshold:
                # Compress and add to L2
                compressed_value = self._compress(value)
                entry.compression_ratio = len(compressed_value) / entry.size_bytes
                entry.value = compressed_value
                self._add_to_l2(key, entry)
                self.stats["compressions"] += 1
            else:
                # Add to L1
                self._add_to_l1(key, entry)

    def _add_to_l1(self, key: str, entry: CacheEntry):
        """Add entry to L1 cache with eviction if needed"""
        # Evict if at capacity
        if len(self.l1_cache) >= self.l1_size:
            self._evict_from_l1()

        self.l1_cache[key] = entry
        self.l1_access_order.append(key)

    def _add_to_l2(self, key: str, entry: CacheEntry):
        """Add entry to L2 cache with eviction if needed"""
        # Evict if at capacity
        if len(self.l2_cache) >= self.l2_size:
            self._evict_from_l2()

        self.l2_cache[key] = entry
        self.l2_access_order.append(key)

    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1"""
        if key in self.l2_cache:
            del self.l2_cache[key]
            if key in self.l2_access_order:
                self.l2_access_order.remove(key)

        self._add_to_l1(key, entry)

    def _evict_from_l1(self):
        """Evict least recently used entry from L1"""
        if self.l1_access_order:
            lru_key = self.l1_access_order.popleft()
            if lru_key in self.l1_cache:
                evicted = self.l1_cache.pop(lru_key)
                # Demote to L2 if valuable
                if evicted.access_count > 1:
                    compressed_value = self._compress(evicted.value)
                    evicted.value = compressed_value
                    self._add_to_l2(lru_key, evicted)
                self.stats["evictions"] += 1

    def _evict_from_l2(self):
        """Evict least recently used entry from L2"""
        if self.l2_access_order:
            lru_key = self.l2_access_order.popleft()
            if lru_key in self.l2_cache:
                del self.l2_cache[lru_key]
                self.stats["evictions"] += 1

    def _add_to_semantic_cache(self, key: str, entry: CacheEntry, embedding: np.ndarray):
        """Add entry to semantic cache"""
        if len(self.semantic_cache) >= self.semantic_cache_size:
            # Remove oldest
            oldest_key = self.semantic_keys[0]
            del self.semantic_cache[oldest_key]
            self.semantic_embeddings.pop(0)
            self.semantic_keys.pop(0)

        self.semantic_cache[key] = (embedding, entry)
        self.semantic_embeddings.append(embedding)
        self.semantic_keys.append(key)

    def _find_semantic_match(self, key: str, threshold: float) -> Optional[CacheEntry]:
        """Find semantically similar cached entry"""
        # This would need actual embedding for the key
        # For now, returning None as placeholder
        return None

    def clear_expired(self):
        """Clear all expired entries"""
        with self._lock:
            # Clear L1
            expired_keys = [k for k, v in self.l1_cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.l1_cache[key]

            # Clear L2
            expired_keys = [k for k, v in self.l2_cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.l2_cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["semantic_hits"]
            total_misses = self.stats["l1_misses"] + self.stats["l2_misses"]

            return {
                **self.stats,
                "l1_size": len(self.l1_cache),
                "l2_size": len(self.l2_cache),
                "semantic_size": len(self.semantic_cache),
                "hit_rate": total_hits / max(1, total_hits + total_misses),
                "l1_hit_rate": self.stats["l1_hits"] / max(1, self.stats["l1_hits"] + self.stats["l1_misses"]),
                "l2_hit_rate": self.stats["l2_hits"] / max(1, self.stats["l2_hits"] + self.stats["l2_misses"])
            }


class RequestBatcher:
    """
    Intelligent request batching for reducing API calls
    Groups similar requests and processes them together
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout_ms: int = 100,
        max_queue_size: int = 1000
    ):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size

        # Request queues by type
        self.queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_queue_size))
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.processors: Dict[str, Callable] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "average_batch_size": 0,
            "queue_overflows": 0
        }

    def register_processor(self, request_type: str, processor: Callable):
        """Register a batch processor for a request type"""
        self.processors[request_type] = processor

    async def add_request(
        self,
        request_type: str,
        data: Any,
        priority: int = 0
    ) -> Any:
        """
        Add request to batch queue

        Args:
            request_type: Type of request for batching
            data: Request data
            priority: Request priority (higher = more important)

        Returns:
            Future result of the batched request
        """
        if request_type not in self.processors:
            raise ValueError(f"No processor registered for request type: {request_type}")

        # Create request entry
        future = asyncio.Future()
        request = BatchRequest(
            request_id=f"{request_type}_{time.time()}_{id(data)}",
            data=data,
            callback=future,
            timestamp=time.time(),
            priority=priority
        )

        # Add to queue
        queue = self.queues[request_type]
        if len(queue) >= self.max_queue_size:
            self.stats["queue_overflows"] += 1
            raise OverflowError(f"Queue for {request_type} is full")

        queue.append(request)
        self.stats["total_requests"] += 1

        # Start processing task if not running
        if request_type not in self.processing_tasks or self.processing_tasks[request_type].done():
            self.processing_tasks[request_type] = asyncio.create_task(
                self._process_batch(request_type)
            )

        return await future

    async def _process_batch(self, request_type: str):
        """Process batches for a specific request type"""
        queue = self.queues[request_type]
        processor = self.processors[request_type]

        while queue:
            batch = []
            batch_start = time.time()

            # Collect batch
            while len(batch) < self.batch_size and queue:
                # Check timeout
                if batch and (time.time() - batch_start) * 1000 > self.batch_timeout_ms:
                    break

                if queue:
                    batch.append(queue.popleft())
                else:
                    # Wait a bit for more requests
                    await asyncio.sleep(self.batch_timeout_ms / 1000)

            if not batch:
                break

            # Sort by priority
            batch.sort(key=lambda x: x.priority, reverse=True)

            # Process batch
            try:
                # Extract data for processing
                batch_data = [req.data for req in batch]

                # Call processor
                results = await processor(batch_data)

                # Distribute results
                for i, request in enumerate(batch):
                    if i < len(results):
                        request.callback.set_result(results[i])
                    else:
                        request.callback.set_exception(
                            Exception("Batch processing returned insufficient results")
                        )

                # Update statistics
                self.stats["total_batches"] += 1
                batch_sizes = self.stats.get("batch_sizes", [])
                batch_sizes.append(len(batch))
                self.stats["average_batch_size"] = sum(batch_sizes) / len(batch_sizes)

                logger.info(
                    f"Processed batch",
                    request_type=request_type,
                    batch_size=len(batch),
                    processing_time_ms=(time.time() - batch_start) * 1000
                )

            except Exception as e:
                # Set exception for all requests in batch
                for request in batch:
                    request.callback.set_exception(e)
                logger.error(f"Batch processing failed", request_type=request_type, error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            **self.stats,
            "queue_sizes": {k: len(v) for k, v in self.queues.items()},
            "active_processors": list(self.processing_tasks.keys())
        }


class ConnectionPool:
    """
    Connection pooling for external services
    Reduces connection overhead and manages resources
    """

    def __init__(
        self,
        min_connections: int = 5,
        max_connections: int = 20,
        connection_timeout: int = 30,
        idle_timeout: int = 300
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout

        # Connection pools by service
        self.pools: Dict[str, List[Any]] = defaultdict(list)
        self.active_connections: Dict[str, int] = defaultdict(int)
        self.connection_factory: Dict[str, Callable] = {}

        # Statistics
        self.stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connections_closed": 0,
            "timeouts": 0
        }

        self._lock = threading.RLock()

    def register_factory(self, service: str, factory: Callable):
        """Register connection factory for a service"""
        self.connection_factory[service] = factory

    async def get_connection(self, service: str) -> Any:
        """Get a connection from the pool"""
        with self._lock:
            pool = self.pools[service]

            # Reuse existing connection
            while pool:
                conn = pool.pop()
                # Check if connection is still valid
                if await self._is_connection_valid(conn):
                    self.stats["connections_reused"] += 1
                    self.active_connections[service] += 1
                    return conn
                else:
                    self.stats["connections_closed"] += 1

            # Create new connection if under limit
            if self.active_connections[service] < self.max_connections:
                if service not in self.connection_factory:
                    raise ValueError(f"No factory registered for service: {service}")

                try:
                    conn = await asyncio.wait_for(
                        self.connection_factory[service](),
                        timeout=self.connection_timeout
                    )
                    self.stats["connections_created"] += 1
                    self.active_connections[service] += 1
                    return conn
                except asyncio.TimeoutError:
                    self.stats["timeouts"] += 1
                    raise

            # Wait for available connection
            raise ConnectionError(f"No connections available for {service}")

    async def return_connection(self, service: str, conn: Any):
        """Return a connection to the pool"""
        with self._lock:
            self.active_connections[service] -= 1

            # Add back to pool if healthy and under limit
            if len(self.pools[service]) < self.max_connections:
                self.pools[service].append(conn)
            else:
                # Close excess connection
                await self._close_connection(conn)
                self.stats["connections_closed"] += 1

    async def _is_connection_valid(self, conn: Any) -> bool:
        """Check if connection is still valid"""
        # Implementation depends on connection type
        # For now, return True as placeholder
        return True

    async def _close_connection(self, conn: Any):
        """Close a connection"""
        # Implementation depends on connection type
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                **self.stats,
                "pool_sizes": {k: len(v) for k, v in self.pools.items()},
                "active_connections": dict(self.active_connections)
            }


# Optimization decorators
def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """Decorator for automatic caching with configurable TTL"""
    cache = MultiTierCache(ttl_seconds=ttl)

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"

            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.put(cache_key, result, ttl=ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"

            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.put(cache_key, result, ttl=ttl)

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def batched(batch_size: int = 10, timeout_ms: int = 100):
    """Decorator for automatic request batching"""
    batcher = RequestBatcher(batch_size=batch_size, batch_timeout_ms=timeout_ms)

    def decorator(func):
        # Register the function as a batch processor
        request_type = func.__name__
        batcher.register_processor(request_type, func)

        @wraps(func)
        async def wrapper(data: Any):
            return await batcher.add_request(request_type, data)

        return wrapper
    return decorator


# Global optimizer instance
global_cache = MultiTierCache()
global_batcher = RequestBatcher()
global_pool = ConnectionPool()