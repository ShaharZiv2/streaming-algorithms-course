import numpy as np
import pandas as pd

from estimators import FMEstimator, AMSEstimator, MorrisEstimator
from schemas.args import Args


class SketchManager:
    _instance = None

    def __init__(self, args: Args):
        self.src_ip_f0_est = FMEstimator()
        self.src_ip_f2_est = AMSEstimator(args.ams_r)

        self.dst_ip_f0_est = FMEstimator()
        self.dst_ip_f2_est = AMSEstimator(args.ams_r)

        self.dst_port_f0_est = FMEstimator()

        self.num_events_f1_est = MorrisEstimator()

    def __new__(cls, args: Args):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def sketch(self, window: pd.DataFrame) -> None:
        """Updates all estimators with the relevant fields"""
        self.src_ip_f0_est.vectorized_update(window['srcip'])
        self.src_ip_f2_est.vectorized_update(window['srcip'])

        self.dst_ip_f0_est.vectorized_update(window['dstip'])
        self.dst_ip_f2_est.vectorized_update(window['dstip'])

        self.dst_port_f0_est.vectorized_update(window['dsport'])

        self.num_events_f1_est.vectorized_update(window.index)

    def reset(self):
        for estimator in self.estimators:
            estimator.reset()

    def estimate(self):
        return np.array([estimator.estimate() for estimator in self.estimators])

    @property
    def estimators(self):
        return [
            self.src_ip_f0_est,
            self.src_ip_f2_est,
            self.dst_ip_f0_est,
            self.dst_ip_f2_est,
            self.dst_port_f0_est,
            self.num_events_f1_est,
        ]
