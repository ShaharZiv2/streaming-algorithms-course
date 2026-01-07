from estimators.count_sketch import CountSketch


def test_count_sketch_r5_b2048():
    """Test Count Sketch with r=5 rows and b=2048 buckets"""
    cs = CountSketch(r=5, b=2048)

    # Add some items
    stream = ['a', 'b', 'c', 'a', 'b', 'a']
    for item in stream:
        cs.update(item)

    # Check estimates (Count Sketch can have some error)
    assert abs(cs.estimate('a') - 3) <= 2  # 'a' appears 3 times
    assert abs(cs.estimate('b') - 2) <= 2  # 'b' appears 2 times
    assert abs(cs.estimate('c') - 1) <= 2  # 'c' appears 1 time
    assert abs(cs.estimate('d')) <= 2      # 'd' never appears


def test_count_sketch_r7_b8192():
    """Test Count Sketch with r=7 rows and b=8192 buckets"""
    cs = CountSketch(r=7, b=8192)

    # Add some items with counts
    cs.update('x', count=5)
    cs.update('y', count=3)
    cs.update('x', count=2)

    # Check estimates
    assert abs(cs.estimate('x') - 7) <= 2  # 'x' has count 5+2=7
    assert abs(cs.estimate('y') - 3) <= 2  # 'y' has count 3
    assert abs(cs.estimate('z')) <= 2      # 'z' never appears


def test_count_sketch_r9_b8192():
    """Test Count Sketch with r=9 rows and b=8192 buckets"""
    cs = CountSketch(r=9, b=8192)

    # Add numeric items
    stream = [1, 2, 3, 1, 2, 1, 1]
    for item in stream:
        cs.update(item)

    # Check estimates
    assert abs(cs.estimate(1) - 4) <= 2  # 1 appears 4 times
    assert abs(cs.estimate(2) - 2) <= 2  # 2 appears 2 times
    assert abs(cs.estimate(3) - 1) <= 2  # 3 appears 1 time


def test_count_sketch_parameters():
    """Test that Count Sketch has correct parameters"""
    cs1 = CountSketch(r=5, b=2048)
    assert cs1.rows == 5
    assert cs1.buckets == 2048
    assert len(cs1.hash_functions) == 5
    assert len(cs1.sign_functions) == 5
    assert cs1.table.shape == (5, 2048)

    cs2 = CountSketch(r=7, b=8192)
    assert cs2.rows == 7
    assert cs2.buckets == 8192
    assert cs2.table.shape == (7, 8192)

    cs3 = CountSketch(r=9, b=2048)
    assert cs3.rows == 9
    assert cs3.buckets == 2048
    assert cs3.table.shape == (9, 2048)


def test_count_sketch_negative_updates():
    """Test that Count Sketch handles negative updates (deletions)"""
    cs = CountSketch(r=7, b=2048)

    # Add and remove items
    cs.update('a', count=5)
    cs.update('a', count=-2)

    # Should estimate around 3
    assert abs(cs.estimate('a') - 3) <= 2

