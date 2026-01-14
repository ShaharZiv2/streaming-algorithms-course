import numpy as np
import pandas as pd

from estimators import FMEstimator, AMSEstimator, MorrisEstimator
from schemas.args import Args
from utils.time_utils import func_timer


class SketchManager:
    _instance = None

    def __init__(self, args: Args):
        self.src_ip_f0_est = FMEstimator()
        self.src_ip_f2_est = AMSEstimator(args.ams_r)

        self.dst_ip_f0_est = FMEstimator()
        self.dst_ip_f2_est = AMSEstimator(args.ams_r)

        self.dst_port_f0_est = FMEstimator()

        self.num_events_f1_est = MorrisEstimator()

        self.attack_f1_estimator = MorrisEstimator()

    def __new__(cls, args: Args):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def sketch(self, event: pd.DataFrame) -> None:
        """Updates all estimators with the relevant fields"""
        self.src_ip_f0_est.update(event['srcip'])
        self.src_ip_f2_est.update(event['srcip'])

        self.dst_ip_f0_est.update(event['dstip'])
        self.dst_ip_f2_est.update(event['dstip'])

        self.dst_port_f0_est.update(event['dsport'])

        self.num_events_f1_est.update()

        if event['Label'] == '1':
            self.attack_f1_estimator.update()

    def reset(self):
        for estimator in self.estimators:
            estimator.reset()

    def estimate(self):
        estimations = [estimator.estimate() for estimator in self.estimators]
        estimations.append(int(self.attack_f1_estimator.estimate() / self.num_events_f1_est.estimate() > 0.06))
        return estimations

    @property
    def estimators(self):
        return [
            self.src_ip_f0_est,
            self.src_ip_f2_est,
            self.dst_ip_f0_est,
            self.dst_ip_f2_est,
            self.dst_port_f0_est,
            self.num_events_f1_est,
            self.attack_f1_estimator
        ]

    @property
    def estimator_names(self):
        return ['SrcIP_F0', 'SrcIP_F2', 'DstIP_F0', 'DstIP_F2', 'DstPortF0', 'NumEventsF1', 'NumAttacksF1', 'Attack']