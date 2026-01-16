import numpy as np
import pandas as pd

from estimators import FMEstimator, AMSEstimator, MorrisEstimator, CountSketch, CountMinSketch, HeavyHittersSketch
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
        self.count_source_ip_heavy_hitter = HeavyHittersSketch()
        self.count_dest_ip_heavy_hitter = HeavyHittersSketch()
        self.count_dst_port_heavy_hitter = HeavyHittersSketch()
        self.source_ip_cms = CountMinSketch(args.cms_width)
        self.dest_ip_cms = CountMinSketch(args.cms_width)
        self.dst_port_cms = CountMinSketch(args.cms_width)
        self.k = 5




    def __new__(cls, args: Args):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    @func_timer
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

        self.count_source_ip_heavy_hitter.update(event['srcip'])
        self.count_dest_ip_heavy_hitter.update(event['dstip'])
        self.count_dst_port_heavy_hitter.update(event['dsport'])
        self.source_ip_cms.update(event['srcip'])
        self.dest_ip_cms.update(event['dstip'])
        self.dst_port_cms.update(event['dsport'])


    def reset(self):
        for estimator in self.estimators.values():
            estimator.reset()

    def estimate(self):
        estimations = [estimator.estimate() for estimator in self.estimators.values()]
        SrcIP_FractionOfMass = sum([self.source_ip_cms.estimate(ip) for ip in self.count_source_ip_heavy_hitter.estimate()]) / self.num_events_f1_est.estimate()
        DstIP_FractionOfMass = sum([self.dest_ip_cms.estimate(ip) for ip in self.count_dest_ip_heavy_hitter.estimate()]) / self.num_events_f1_est.estimate()
        DstPort_FractionOfMass = sum([self.dst_port_cms.estimate(port) for port in self.count_dst_port_heavy_hitter.estimate()]) / self.num_events_f1_est.estimate()
        estimations.extend([SrcIP_FractionOfMass, DstIP_FractionOfMass, DstPort_FractionOfMass])
        SrcIP_Concentration =  self.src_ip_f2_est.estimate() / (self.src_ip_f0_est.estimate() ** 2)
        DstIP_Concentration =  self.dst_ip_f2_est.estimate() / (self.dst_ip_f0_est.estimate() ** 2)
        DstPort_Concentration =  self.dst_ip_f2_est.estimate() / (self.dst_ip_f0_est.estimate() ** 2)
        estimations.extend([SrcIP_Concentration, DstIP_Concentration, DstPort_Concentration])

        # Attack detection
        estimations.append(int(self.attack_f1_estimator.estimate() / self.num_events_f1_est.estimate() > 0.06))

        return estimations

    @property
    def estimators(self):
        return {
            'SrcIP_F0': self.src_ip_f0_est,
            'SrcIP_F2': self.src_ip_f2_est,
            'DstIP_F0': self.dst_ip_f0_est,
            'DstIP_F2': self.dst_ip_f2_est,
            'DstPortF0': self.dst_port_f0_est,
            'NumEventsF1': self.num_events_f1_est,
            'NumAttacksF1': self.attack_f1_estimator,
        }

    @property
    def estimator_names(self):
        return list(self.estimators.keys()) + ['Attack']
