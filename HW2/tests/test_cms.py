from estimators.cms import CountMinSketch


def test_cms_w2048():
    """Test CMS with width 2048"""
    cms = CountMinSketch(w=2048)

    # Add some items
    stream = ['a', 'b', 'c', 'a', 'b', 'a']
    for item in stream:
        cms.update(item)

    # Check estimates
    assert cms.estimate('a') >= 3  # 'a' appears 3 times
    assert cms.estimate('b') >= 2  # 'b' appears 2 times
    assert cms.estimate('c') >= 1  # 'c' appears 1 time
    assert cms.estimate('d') == 0  # 'd' never appears

    # CMS never underestimates, so these should be exact or overestimates
    assert cms.estimate('a') == 3
    assert cms.estimate('b') == 2
    assert cms.estimate('c') == 1


def test_cms_w8192():
    """Test CMS with width 8192"""
    cms = CountMinSketch(w=8192)

    # Add some items with counts
    cms.update('x', count=5)
    cms.update('y', count=3)
    cms.update('x', count=2)

    # Check estimates
    assert cms.estimate('x') >= 7  # 'x' has count 5+2=7
    assert cms.estimate('y') >= 3  # 'y' has count 3
    assert cms.estimate('z') == 0  # 'z' never appears


def test_cms_parameters():
    """Test that CMS has correct parameters"""
    cms = CountMinSketch(w=2048)
    assert cms.width == 2048
    assert cms.depth == 4
    assert len(cms.hash_functions) == 4
    assert cms.table.shape == (4, 2048)

    cms2 = CountMinSketch(w=8192)
    assert cms2.width == 8192
    assert cms2.depth == 4
    assert cms2.table.shape == (4, 8192)

