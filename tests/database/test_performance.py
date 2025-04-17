import pytest
import time
import statistics
from pathlib import Path
from database.database_manager import DatabaseManager

@pytest.fixture
def perf_db():
    """Create a database instance for performance testing"""
    db_path = Path("perf_node_zero.db")
    if db_path.exists():
        db_path.unlink()
    
    db = DatabaseManager()
    db.db_path = db_path
    db._connect()
    db._init_db()
    
    yield db
    
    # Cleanup
    db.close()
    if db_path.exists():
        db_path.unlink()

def test_save_performance(perf_db):
    """Test performance of save operations"""
    iterations = 1000
    times = []
    
    for i in range(iterations):
        state = {
            'id': f'perf_test_{i}',
            'timestamp': time.time(),
            'state_type': 'performance',
            'metrics': {'iteration': i, 'data': 'x' * 1000},  # 1KB of data
            'version_data': {'test': 'performance'}
        }
        
        start = time.time()
        perf_db.save_neural_state(state)
        end = time.time()
        
        times.append(end - start)
    
    # Calculate statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    std_dev = statistics.stdev(times)
    
    print(f"\nSave Performance (n={iterations}):")
    print(f"Mean: {mean*1000:.2f}ms")
    print(f"Median: {median*1000:.2f}ms")
    print(f"Std Dev: {std_dev*1000:.2f}ms")
    
    # Assert that 95% of operations complete within 50ms
    threshold = 0.05  # 50ms
    slow_operations = sum(1 for t in times if t > threshold)
    assert slow_operations / iterations < 0.05  # Less than 5% of operations are slow

def test_read_performance(perf_db):
    """Test performance of read operations"""
    # First, populate the database
    for i in range(1000):
        state = {
            'id': f'read_test_{i}',
            'timestamp': time.time(),
            'state_type': 'read_test',
            'metrics': {'iteration': i},
            'version_data': {'test': 'read_performance'}
        }
        perf_db.save_neural_state(state)
    
    iterations = 1000
    times = []
    
    for i in range(iterations):
        start = time.time()
        state = perf_db.get_neural_state(f'read_test_{i % 1000}')
        end = time.time()
        
        times.append(end - start)
        assert state is not None
    
    # Calculate statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    std_dev = statistics.stdev(times)
    
    print(f"\nRead Performance (n={iterations}):")
    print(f"Mean: {mean*1000:.2f}ms")
    print(f"Median: {median*1000:.2f}ms")
    print(f"Std Dev: {std_dev*1000:.2f}ms")
    
    # Assert that 95% of operations complete within 20ms
    threshold = 0.02  # 20ms
    slow_operations = sum(1 for t in times if t > threshold)
    assert slow_operations / iterations < 0.05

def test_concurrent_performance(perf_db):
    """Test performance under concurrent access"""
    import threading
    
    def worker(db, count):
        for i in range(count):
            state = {
                'id': f'concurrent_perf_{threading.current_thread().name}_{i}',
                'timestamp': time.time(),
                'state_type': 'concurrent',
                'metrics': {'thread': threading.current_thread().name, 'iteration': i},
                'version_data': {'test': 'concurrent_performance'}
            }
            db.save_neural_state(state)
    
    thread_count = 10
    operations_per_thread = 100
    threads = []
    
    start = time.time()
    
    for i in range(thread_count):
        thread = threading.Thread(
            target=worker,
            args=(perf_db, operations_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end = time.time()
    total_time = end - start
    total_operations = thread_count * operations_per_thread
    operations_per_second = total_operations / total_time
    
    print(f"\nConcurrent Performance:")
    print(f"Total Operations: {total_operations}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Operations/Second: {operations_per_second:.2f}")
    
    # Assert minimum throughput
    assert operations_per_second > 100  # At least 100 operations per second

def test_database_size(perf_db):
    """Test database size growth"""
    initial_size = perf_db.db_path.stat().st_size
    
    # Add 1000 records
    for i in range(1000):
        state = {
            'id': f'size_test_{i}',
            'timestamp': time.time(),
            'state_type': 'size_test',
            'metrics': {'data': 'x' * 10000},  # 10KB of data
            'version_data': {'test': 'size_performance'}
        }
        perf_db.save_neural_state(state)
    
    final_size = perf_db.db_path.stat().st_size
    size_growth = final_size - initial_size
    
    print(f"\nDatabase Size Growth:")
    print(f"Initial Size: {initial_size/1024:.2f}KB")
    print(f"Final Size: {final_size/1024:.2f}KB")
    print(f"Growth: {size_growth/1024:.2f}KB")
    
    # Assert that size growth is reasonable (less than 15MB for 1000 10KB records)
    assert size_growth < 15 * 1024 * 1024 