# Performance optimizations for agent system

import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps, lru_cache
import json

from ankigen_core.logging import logger
from ankigen_core.models import Card


T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for agent response caching"""

    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000
    cache_backend: str = "memory"  # "memory" or "file"
    cache_directory: Optional[str] = None

    def __post_init__(self):
        if self.cache_backend == "file" and not self.cache_directory:
            self.cache_directory = "cache/agents"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations"""

    enable_batch_processing: bool = True
    max_batch_size: int = 10
    batch_timeout: float = 2.0  # seconds
    enable_parallel_execution: bool = True
    max_concurrent_requests: int = 5
    enable_request_deduplication: bool = True
    enable_response_caching: bool = True
    cache_config: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""

    value: T
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    cache_key: str = ""

    def is_expired(self, ttl: int) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > ttl

    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


class MemoryCache(Generic[T]):
    """In-memory cache with LRU eviction"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            if entry.is_expired(self.config.cache_ttl):
                await self._remove(key)
                return None

            entry.touch()
            self._update_access_order(key)

            logger.debug(f"Cache hit for key: {key[:20]}...")
            return entry.value

    async def set(self, key: str, value: T) -> None:
        """Set value in cache"""
        async with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.config.max_cache_size:
                await self._evict_lru()

            entry = CacheEntry(value=value, created_at=time.time(), cache_key=key)

            self._cache[key] = entry
            self._update_access_order(key)

            logger.debug(f"Cache set for key: {key[:20]}...")

    async def remove(self, key: str) -> bool:
        """Remove entry from cache"""
        async with self._lock:
            return await self._remove(key)

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cache cleared")

    async def _remove(self, key: str) -> bool:
        """Internal remove method"""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    async def _evict_lru(self) -> None:
        """Evict least recently used entries"""
        if not self._access_order:
            return

        # Remove oldest entries
        to_remove = self._access_order[: len(self._access_order) // 4]  # Remove 25%
        for key in to_remove:
            await self._remove(key)

        logger.debug(f"Evicted {len(to_remove)} cache entries")

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        return {
            "entries": len(self._cache),
            "max_size": self.config.max_cache_size,
            "total_accesses": total_accesses,
            "hit_rate": total_accesses / max(1, len(self._cache)),
        }


class BatchProcessor:
    """Batch processor for agent requests"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._batches: Dict[str, List[Dict[str, Any]]] = {}
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def add_request(
        self, batch_key: str, request_data: Dict[str, Any], processor_func: Callable
    ) -> Any:
        """Add request to batch for processing"""

        if not self.config.enable_batch_processing:
            # Process immediately if batching is disabled
            return await processor_func([request_data])

        async with self._lock:
            # Initialize batch if needed
            if batch_key not in self._batches:
                self._batches[batch_key] = []
                self._start_batch_timer(batch_key, processor_func)

            # Add request to batch
            self._batches[batch_key].append(request_data)

            # Process immediately if batch is full
            if len(self._batches[batch_key]) >= self.config.max_batch_size:
                return await self._process_batch(batch_key, processor_func)

            # Wait for timer or batch completion
            return await self._wait_for_batch_result(
                batch_key, request_data, processor_func
            )

    def _start_batch_timer(self, batch_key: str, processor_func: Callable) -> None:
        """Start timer for batch processing"""

        async def timer():
            await asyncio.sleep(self.config.batch_timeout)
            async with self._lock:
                if batch_key in self._batches and self._batches[batch_key]:
                    await self._process_batch(batch_key, processor_func)

        self._batch_timers[batch_key] = asyncio.create_task(timer())

    async def _process_batch(
        self, batch_key: str, processor_func: Callable
    ) -> List[Any]:
        """Process accumulated batch"""
        if batch_key not in self._batches:
            return []

        batch = self._batches.pop(batch_key)

        # Cancel timer
        if batch_key in self._batch_timers:
            self._batch_timers[batch_key].cancel()
            del self._batch_timers[batch_key]

        if not batch:
            return []

        logger.debug(f"Processing batch {batch_key} with {len(batch)} requests")

        try:
            # Process the batch
            results = await processor_func(batch)
            return results if isinstance(results, list) else [results]

        except Exception as e:
            logger.error(f"Batch processing failed for {batch_key}: {e}")
            raise

    async def _wait_for_batch_result(
        self, batch_key: str, request_data: Dict[str, Any], processor_func: Callable
    ) -> Any:
        """Wait for batch processing to complete"""
        # This is a simplified implementation
        # In a real implementation, you'd use events/conditions to coordinate
        # between requests in the same batch

        while batch_key in self._batches:
            await asyncio.sleep(0.1)

        # For now, process individually as fallback
        return await processor_func([request_data])


class RequestDeduplicator:
    """Deduplicates identical agent requests"""

    def __init__(self):
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    @lru_cache(maxsize=1000)
    def _generate_request_hash(self, request_data: str) -> str:
        """Generate hash for request deduplication"""
        return hashlib.md5(request_data.encode()).hexdigest()

    async def deduplicate_request(
        self, request_data: Dict[str, Any], processor_func: Callable
    ) -> Any:
        """Deduplicate and process request"""

        # Generate hash for deduplication
        request_str = json.dumps(request_data, sort_keys=True)
        request_hash = self._generate_request_hash(request_str)

        async with self._lock:
            # Check if request is already pending
            if request_hash in self._pending_requests:
                logger.debug(f"Deduplicating request: {request_hash[:16]}...")
                return await self._pending_requests[request_hash]

            # Create future for this request
            future = asyncio.create_task(
                self._process_unique_request(request_hash, request_data, processor_func)
            )

            self._pending_requests[request_hash] = future

            try:
                result = await future
                return result
            finally:
                # Clean up completed request
                async with self._lock:
                    self._pending_requests.pop(request_hash, None)

    async def _process_unique_request(
        self, request_hash: str, request_data: Dict[str, Any], processor_func: Callable
    ) -> Any:
        """Process unique request"""
        logger.debug(f"Processing unique request: {request_hash[:16]}...")
        return await processor_func(request_data)


class PerformanceOptimizer:
    """Main performance optimization coordinator"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cache = (
            MemoryCache(config.cache_config) if config.enable_response_caching else None
        )
        self.batch_processor = (
            BatchProcessor(config) if config.enable_batch_processing else None
        )
        self.deduplicator = (
            RequestDeduplicator() if config.enable_request_deduplication else None
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def optimize_agent_call(
        self,
        agent_name: str,
        request_data: Dict[str, Any],
        processor_func: Callable,
        cache_key_generator: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> Any:
        """Optimize agent call with caching, batching, and deduplication"""

        # Generate cache key
        cache_key = None
        if self.cache and cache_key_generator:
            cache_key = cache_key_generator(request_data)

            # Check cache first
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Apply rate limiting
        async with self._semaphore:
            # Apply deduplication
            if self.deduplicator and self.config.enable_request_deduplication:
                result = await self.deduplicator.deduplicate_request(
                    request_data, processor_func
                )
            else:
                result = await processor_func(request_data)

            # Cache result
            if self.cache and cache_key and result is not None:
                await self.cache.set(cache_key, result)

            return result

    async def optimize_batch_processing(
        self, batch_key: str, request_data: Dict[str, Any], processor_func: Callable
    ) -> Any:
        """Optimize using batch processing"""
        if self.batch_processor:
            return await self.batch_processor.add_request(
                batch_key, request_data, processor_func
            )
        else:
            return await processor_func([request_data])

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics"""
        stats = {
            "config": {
                "batch_processing": self.config.enable_batch_processing,
                "parallel_execution": self.config.enable_parallel_execution,
                "request_deduplication": self.config.enable_request_deduplication,
                "response_caching": self.config.enable_response_caching,
            },
            "concurrency": {
                "max_concurrent": self.config.max_concurrent_requests,
                "current_available": self._semaphore._value,
            },
        }

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        return stats


# Global performance optimizer
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer(
    config: Optional[PerformanceConfig] = None,
) -> PerformanceOptimizer:
    """Get global performance optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config or PerformanceConfig())
    return _global_optimizer


# Decorators for performance optimization
def cache_response(cache_key_func: Callable[[Any], str], ttl: int = 3600):
    """Decorator to cache function responses"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            if not optimizer.cache:
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = cache_key_func(*args, **kwargs)

            # Check cache
            cached_result = await optimizer.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            if result is not None:
                await optimizer.cache.set(cache_key, result)

            return result

        return wrapper

    return decorator


def rate_limit(max_concurrent: int = 5):
    """Decorator to apply rate limiting"""
    semaphore = asyncio.Semaphore(max_concurrent)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions for cache key generation
def generate_card_cache_key(
    topic: str, subject: str, num_cards: int, difficulty: str, **kwargs
) -> str:
    """Generate cache key for card generation"""
    key_data = {
        "topic": topic,
        "subject": subject,
        "num_cards": num_cards,
        "difficulty": difficulty,
        "context": kwargs.get("context", {}),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return f"cards:{hashlib.md5(key_str.encode()).hexdigest()}"


def generate_judgment_cache_key(
    cards: List[Card], judgment_type: str = "general"
) -> str:
    """Generate cache key for card judgment"""
    # Use card content to generate stable hash
    card_data = []
    for card in cards:
        card_data.append(
            {
                "question": card.front.question,
                "answer": card.back.answer,
                "type": card.card_type,
            }
        )

    key_data = {"cards": card_data, "judgment_type": judgment_type}
    key_str = json.dumps(key_data, sort_keys=True)
    return f"judgment:{hashlib.md5(key_str.encode()).hexdigest()}"
