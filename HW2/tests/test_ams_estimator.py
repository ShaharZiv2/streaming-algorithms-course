from estimators.ams import AMSEstimator

def test_ams_estimator():
    stream = [1,2,3,4] * 2
    ams_estimator = AMSEstimator(256)
    for element in stream:
        ams_estimator.update(element)

    estimation = ams_estimator.estimate()
    assert 14 <= estimation <= 18