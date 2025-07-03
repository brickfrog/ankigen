# Tests for ankigen_core/agents/performance.py

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch

from ankigen_core.agents.performance import (
    CacheConfig,
    PerformanceConfig,
    CacheEntry,
    MemoryCache,
    BatchProcessor,
    RequestDeduplicator,
    PerformanceOptimizer,
    PerformanceMonitor,
    get_performance_optimizer,
    get_performance_monitor,
    cache_response,
    rate_limit,
    generate_card_cache_key,
    generate_judgment_cache_key
)
from ankigen_core.models import Card, CardFront, CardBack


# Test CacheConfig
def test_cache_config_defaults():
    """Test CacheConfig default values"""
    config = CacheConfig()
    
    assert config.enable_caching is True
    assert config.cache_ttl == 3600
    assert config.max_cache_size == 1000
    assert config.cache_backend == "memory"
    assert config.cache_directory is None


def test_cache_config_file_backend():
    """Test CacheConfig with file backend"""
    config = CacheConfig(cache_backend="file")
    
    assert config.cache_directory == "cache/agents"


# Test PerformanceConfig
def test_performance_config_defaults():
    """Test PerformanceConfig default values"""
    config = PerformanceConfig()
    
    assert config.enable_batch_processing is True
    assert config.max_batch_size == 10
    assert config.batch_timeout == 2.0
    assert config.enable_parallel_execution is True
    assert config.max_concurrent_requests == 5
    assert config.enable_request_deduplication is True
    assert config.enable_response_caching is True
    assert isinstance(config.cache_config, CacheConfig)


# Test CacheEntry
def test_cache_entry_creation():
    """Test CacheEntry creation"""
    with patch('time.time', return_value=1000.0):
        entry = CacheEntry(value="test", created_at=1000.0)
        
        assert entry.value == "test"
        assert entry.created_at == 1000.0
        assert entry.access_count == 0
        assert entry.last_accessed == 1000.0


def test_cache_entry_expiration():
    """Test CacheEntry expiration"""
    entry = CacheEntry(value="test", created_at=1000.0)
    
    with patch('time.time', return_value=1500.0):
        assert entry.is_expired(ttl=300) is False  # Not expired
        
    with patch('time.time', return_value=2000.0):
        assert entry.is_expired(ttl=300) is True  # Expired


def test_cache_entry_touch():
    """Test CacheEntry touch method"""
    entry = CacheEntry(value="test", created_at=1000.0)
    initial_count = entry.access_count
    
    with patch('time.time', return_value=1500.0):
        entry.touch()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed == 1500.0


# Test MemoryCache
@pytest.fixture
def memory_cache():
    """Memory cache for testing"""
    config = CacheConfig(max_cache_size=3, cache_ttl=300)
    return MemoryCache(config)


async def test_memory_cache_set_and_get(memory_cache):
    """Test basic cache set and get operations"""
    await memory_cache.set("key1", "value1")
    
    result = await memory_cache.get("key1")
    assert result == "value1"


async def test_memory_cache_miss(memory_cache):
    """Test cache miss"""
    result = await memory_cache.get("nonexistent")
    assert result is None


async def test_memory_cache_expiration(memory_cache):
    """Test cache entry expiration"""
    with patch('time.time', return_value=1000.0):
        await memory_cache.set("key1", "value1")
    
    # Move forward in time beyond TTL
    with patch('time.time', return_value=2000.0):
        result = await memory_cache.get("key1")
        assert result is None


async def test_memory_cache_lru_eviction(memory_cache):
    """Test LRU eviction when cache is full"""
    # Fill cache to capacity
    await memory_cache.set("key1", "value1")
    await memory_cache.set("key2", "value2")
    await memory_cache.set("key3", "value3")
    
    # Access key1 to make it recently used
    await memory_cache.get("key1")
    
    # Add another item, should evict oldest unused
    await memory_cache.set("key4", "value4")
    
    # key1 should still be there (recently accessed)
    assert await memory_cache.get("key1") == "value1"
    
    # key4 should be there (newest)
    assert await memory_cache.get("key4") == "value4"


async def test_memory_cache_remove(memory_cache):
    """Test cache entry removal"""
    await memory_cache.set("key1", "value1")
    
    removed = await memory_cache.remove("key1")
    assert removed is True
    
    result = await memory_cache.get("key1")
    assert result is None
    
    # Removing non-existent key
    removed = await memory_cache.remove("nonexistent")
    assert removed is False


async def test_memory_cache_clear(memory_cache):
    """Test cache clearing"""
    await memory_cache.set("key1", "value1")
    await memory_cache.set("key2", "value2")
    
    await memory_cache.clear()
    
    assert await memory_cache.get("key1") is None
    assert await memory_cache.get("key2") is None


def test_memory_cache_stats(memory_cache):
    """Test cache statistics"""
    stats = memory_cache.get_stats()
    
    assert "entries" in stats
    assert "max_size" in stats
    assert "total_accesses" in stats
    assert "hit_rate" in stats


# Test BatchProcessor
@pytest.fixture
def batch_processor():
    """Batch processor for testing"""
    config = PerformanceConfig(max_batch_size=3, batch_timeout=0.1)
    return BatchProcessor(config)


async def test_batch_processor_immediate_processing_when_disabled():
    """Test immediate processing when batching is disabled"""
    config = PerformanceConfig(enable_batch_processing=False)
    processor = BatchProcessor(config)
    
    mock_func = AsyncMock(return_value=["result"])
    
    result = await processor.add_request("batch1", {"data": "test"}, mock_func)
    
    assert result == ["result"]
    mock_func.assert_called_once_with([{"data": "test"}])


async def test_batch_processor_batch_size_trigger(batch_processor):
    """Test batch processing triggered by size limit"""
    mock_func = AsyncMock(return_value=["result1", "result2", "result3"])
    
    # Add requests up to batch size
    tasks = []
    for i in range(3):
        task = asyncio.create_task(batch_processor.add_request(
            "batch1", {"data": f"test{i}"}, mock_func
        ))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # All requests should get results
    assert len(results) == 3
    mock_func.assert_called_once()


# Test RequestDeduplicator
@pytest.fixture
def request_deduplicator():
    """Request deduplicator for testing"""
    return RequestDeduplicator()


async def test_request_deduplicator_unique_requests(request_deduplicator):
    """Test deduplicator with unique requests"""
    mock_func = AsyncMock(side_effect=lambda x: f"result_for_{x['id']}")
    
    result1 = await request_deduplicator.deduplicate_request(
        {"id": "1", "data": "test1"}, mock_func
    )
    result2 = await request_deduplicator.deduplicate_request(
        {"id": "2", "data": "test2"}, mock_func
    )
    
    assert result1 == "result_for_{'id': '1', 'data': 'test1'}"
    assert result2 == "result_for_{'id': '2', 'data': 'test2'}"
    assert mock_func.call_count == 2


async def test_request_deduplicator_duplicate_requests(request_deduplicator):
    """Test deduplicator with duplicate requests"""
    mock_func = AsyncMock(return_value="shared_result")
    
    # Send identical requests concurrently
    tasks = [
        request_deduplicator.deduplicate_request(
            {"data": "identical"}, mock_func
        )
        for _ in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All should get the same result
    assert all(result == "shared_result" for result in results)
    
    # Function should only be called once
    mock_func.assert_called_once()


# Test PerformanceOptimizer
@pytest.fixture
def performance_optimizer():
    """Performance optimizer for testing"""
    config = PerformanceConfig(
        max_concurrent_requests=2,
        enable_response_caching=True
    )
    return PerformanceOptimizer(config)


async def test_performance_optimizer_caching(performance_optimizer):
    """Test performance optimizer caching"""
    mock_func = AsyncMock(return_value="cached_result")
    
    def cache_key_gen(data):
        return f"key_{data['id']}"
    
    # First call should execute function
    result1 = await performance_optimizer.optimize_agent_call(
        "test_agent",
        {"id": "123"},
        mock_func,
        cache_key_gen
    )
    
    # Second call with same data should use cache
    result2 = await performance_optimizer.optimize_agent_call(
        "test_agent",
        {"id": "123"},
        mock_func,
        cache_key_gen
    )
    
    assert result1 == "cached_result"
    assert result2 == "cached_result"
    
    # Function should only be called once
    mock_func.assert_called_once()


async def test_performance_optimizer_concurrency_limit(performance_optimizer):
    """Test performance optimizer concurrency limiting"""
    # Slow function to test concurrency
    async def slow_func(data):
        await asyncio.sleep(0.1)
        return f"result_{data['id']}"
    
    # Start more tasks than the concurrency limit
    tasks = [
        performance_optimizer.optimize_agent_call(
            "test_agent",
            {"id": str(i)},
            slow_func
        )
        for i in range(5)
    ]
    
    # All should complete successfully despite concurrency limit
    results = await asyncio.gather(*tasks)
    assert len(results) == 5


def test_performance_optimizer_stats(performance_optimizer):
    """Test performance optimizer statistics"""
    stats = performance_optimizer.get_performance_stats()
    
    assert "config" in stats
    assert "concurrency" in stats
    assert "cache" in stats  # Should have cache stats
    
    assert stats["config"]["response_caching"] is True
    assert stats["concurrency"]["max_concurrent"] == 2


# Test PerformanceMonitor
async def test_performance_monitor():
    """Test performance monitoring"""
    monitor = PerformanceMonitor()
    
    # Record some metrics
    await monitor.record_execution_time("operation1", 1.5)
    await monitor.record_execution_time("operation1", 2.0)
    await monitor.record_execution_time("operation2", 0.5)
    
    report = monitor.get_performance_report()
    
    assert "operation1" in report
    assert "operation2" in report
    
    op1_stats = report["operation1"]
    assert op1_stats["count"] == 2
    assert op1_stats["avg_time"] == 1.75
    assert op1_stats["min_time"] == 1.5
    assert op1_stats["max_time"] == 2.0


# Test decorators
async def test_cache_response_decorator():
    """Test cache_response decorator"""
    call_count = 0
    
    @cache_response(lambda x: f"key_{x}", ttl=300)
    async def test_func(param):
        nonlocal call_count
        call_count += 1
        return f"result_{param}"
    
    # First call
    result1 = await test_func("test")
    assert result1 == "result_test"
    assert call_count == 1
    
    # Second call should use cache
    result2 = await test_func("test")
    assert result2 == "result_test"
    assert call_count == 1  # Should not increment


async def test_rate_limit_decorator():
    """Test rate_limit decorator"""
    execution_times = []
    
    @rate_limit(max_concurrent=1)
    async def test_func(delay):
        start_time = time.time()
        await asyncio.sleep(delay)
        end_time = time.time()
        execution_times.append((start_time, end_time))
        return "done"
    
    # Start multiple tasks
    tasks = [
        test_func(0.1),
        test_func(0.1),
        test_func(0.1)
    ]
    
    await asyncio.gather(*tasks)
    
    # With max_concurrent=1, executions should be sequential
    assert len(execution_times) == 3
    
    # Check that they don't overlap significantly
    for i in range(len(execution_times) - 1):
        current_end = execution_times[i][1]
        next_start = execution_times[i + 1][0]
        # Allow small overlap due to timing precision
        assert next_start >= current_end - 0.01


# Test utility functions
def test_generate_card_cache_key():
    """Test card cache key generation"""
    key1 = generate_card_cache_key(
        topic="Python",
        subject="programming",
        num_cards=5,
        difficulty="intermediate"
    )
    
    key2 = generate_card_cache_key(
        topic="Python",
        subject="programming", 
        num_cards=5,
        difficulty="intermediate"
    )
    
    # Same parameters should generate same key
    assert key1 == key2
    
    # Different parameters should generate different key
    key3 = generate_card_cache_key(
        topic="Java",
        subject="programming",
        num_cards=5,
        difficulty="intermediate"
    )
    
    assert key1 != key3


def test_generate_judgment_cache_key():
    """Test judgment cache key generation"""
    cards = [
        Card(
            front=CardFront(question="What is Python?"),
            back=CardBack(answer="A programming language", explanation="", example=""),
            card_type="basic"
        ),
        Card(
            front=CardFront(question="What is Java?"),
            back=CardBack(answer="A programming language", explanation="", example=""),
            card_type="basic"
        )
    ]
    
    key1 = generate_judgment_cache_key(cards, "accuracy")
    key2 = generate_judgment_cache_key(cards, "accuracy")
    
    # Same cards and judgment type should generate same key
    assert key1 == key2
    
    # Different judgment type should generate different key
    key3 = generate_judgment_cache_key(cards, "clarity")
    assert key1 != key3


# Test global instances
def test_get_performance_optimizer_singleton():
    """Test performance optimizer singleton"""
    optimizer1 = get_performance_optimizer()
    optimizer2 = get_performance_optimizer()
    
    assert optimizer1 is optimizer2


def test_get_performance_monitor_singleton():
    """Test performance monitor singleton"""
    monitor1 = get_performance_monitor()
    monitor2 = get_performance_monitor()
    
    assert monitor1 is monitor2


# Integration tests
async def test_full_optimization_pipeline():
    """Test complete optimization pipeline"""
    config = PerformanceConfig(
        enable_batch_processing=True,
        enable_request_deduplication=True,
        enable_response_caching=True,
        max_batch_size=2,
        batch_timeout=0.1
    )
    
    optimizer = PerformanceOptimizer(config)
    
    call_count = 0
    
    async def mock_processor(data):
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"
    
    def cache_key_gen(data):
        return f"key_{data['id']}"
    
    # Multiple calls with same data should be deduplicated and cached
    tasks = [
        optimizer.optimize_agent_call(
            "test_agent",
            {"id": "same"},
            mock_processor,
            cache_key_gen
        )
        for _ in range(3)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All should get same result
    assert all(result == results[0] for result in results)
    
    # Processor should only be called once due to deduplication
    assert call_count == 1


# Error handling tests
async def test_memory_cache_error_handling():
    """Test memory cache error handling"""
    cache = MemoryCache(CacheConfig())
    
    # Test with None values
    await cache.set("key", None)
    result = await cache.get("key")
    assert result is None


async def test_batch_processor_error_handling():
    """Test batch processor error handling"""
    processor = BatchProcessor(PerformanceConfig())
    
    async def failing_func(data):
        raise Exception("Processing failed")
    
    with pytest.raises(Exception, match="Processing failed"):
        await processor.add_request("batch", {"data": "test"}, failing_func)


async def test_performance_optimizer_error_recovery():
    """Test performance optimizer error recovery"""
    optimizer = PerformanceOptimizer(PerformanceConfig())
    
    async def sometimes_failing_func(data):
        if data.get("fail"):
            raise Exception("Intentional failure")
        return "success"
    
    # Successful call
    result = await optimizer.optimize_agent_call(
        "test_agent",
        {"id": "1"},
        sometimes_failing_func
    )
    assert result == "success"
    
    # Failing call should propagate error
    with pytest.raises(Exception, match="Intentional failure"):
        await optimizer.optimize_agent_call(
            "test_agent",
            {"id": "2", "fail": True},
            sometimes_failing_func
        )