from estimators.morris import MorrisEstimator

def test_morris_estimator():
    stream = [1, 2, 3, 4] * 2
    morris_estimator = MorrisEstimator(r=64)

    for element in stream:
        morris_estimator.update()

    estimation = morris_estimator.estimate()
    # Stream has 8 elements, so we expect estimate around that
    # Morris counters have variance, so we allow a reasonable range
    assert 4 <= estimation <= 16

