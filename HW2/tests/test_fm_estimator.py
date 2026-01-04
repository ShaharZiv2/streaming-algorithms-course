from estimators.fm import FMEstimator

def test_fm_estimator():
    stream = [1, 2, 3, 4] * 2
    fm_estimator = FMEstimator(num_estimators=16)

    for element in stream:
        fm_estimator.update(element)

    estimation = fm_estimator.estimate()
    # Stream has 4 distinct elements, so we expect estimate around 4
    # FM estimator has variance, so we allow a reasonable range
    assert 2 <= estimation <= 8

