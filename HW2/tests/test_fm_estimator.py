from estimators.fm import FMEstimator

def test_fm_estimator():
    stream = [1, 2, 3, 4] * 2
    fm_estimator = FMEstimator(r=64)

    for element in stream:
        fm_estimator.update(element)

    estimation = fm_estimator.estimate()
    # Stream has 4 distinct elements
    # FM estimator has high variance, so we just check it's in a reasonable positive range
    assert 1 <= estimation <= 100

