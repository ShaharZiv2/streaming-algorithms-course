import pandas as pd

from estimators import FMEstimator, AMSEstimator, MorrisEstimator, CountMinSketch, HeavyHittersSketch
from schemas.args import Args


class SketchManager:
    _instance = None

    def __init__(self, args: Args):
        self.src_ip_f0 = FMEstimator()
        self.src_ip_f2 = AMSEstimator(args.ams_r)
        self.src_ip_cms = CountMinSketch(args.cms_width)
        self.src_ip_heavy_hitter = HeavyHittersSketch()

        self.dst_ip_f0 = FMEstimator()
        self.dst_ip_f2 = AMSEstimator(args.ams_r)
        self.dst_ip_cms = CountMinSketch(args.cms_width)
        self.dst_ip_heavy_hitter = HeavyHittersSketch()

        self.dst_port_f2 = AMSEstimator(args.ams_r)
        self.dst_port_f0 = FMEstimator()
        self.dst_port_cms = CountMinSketch(args.cms_width)
        self.dst_port_heavy_hitter = HeavyHittersSketch()

        self.num_events_f1 = MorrisEstimator()

        self.attack_f1_estimator = MorrisEstimator()

    def __new__(cls, args: Args):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    # @func_timer
    def sketch(self, event: pd.DataFrame) -> None:
        """Updates all estimators with the relevant fields"""
        self.src_ip_f0.update(event['srcip'])
        self.src_ip_f2.update(event['srcip'])
        self.src_ip_cms.update(event['srcip'])
        self.src_ip_heavy_hitter.update(event['srcip'])

        self.dst_ip_f0.update(event['dstip'])
        self.dst_ip_f2.update(event['dstip'])
        self.dst_ip_cms.update(event['dstip'])
        self.dst_ip_heavy_hitter.update(event['dstip'])

        self.dst_port_f2.update(event['dsport'])
        self.dst_port_f0.update(event['dsport'])
        self.dst_port_heavy_hitter.update(event['dsport'])
        self.dst_port_cms.update(event['dsport'])

        self.num_events_f1.update()

        if event['Label'] == '1':
            self.attack_f1_estimator.update()


    def reset(self):
        for estimator in self.estimators:
            estimator.reset()

    def estimate(self):
        SrcIP_FractionOfMass = sum([self.source_ip_cms.estimate(ip) for ip in
                                    self.count_source_ip_heavy_hitter.estimate()]) / self.num_events_f1_est.estimate()
        DstIP_FractionOfMass = sum([self.dest_ip_cms.estimate(ip) for ip in
                                    self.count_dest_ip_heavy_hitter.estimate()]) / self.num_events_f1_est.estimate()
        DstPort_FractionOfMass = sum([self.dst_port_cms.estimate(port) for port in
                                      self.count_dst_port_heavy_hitter.estimate()]) / self.num_events_f1_est.estimate()
        SrcIP_Concentration = self.src_ip_f2_est.estimate() / (self.src_ip_f0_est.estimate() ** 2)
        DstIP_Concentration = self.dst_ip_f2_est.estimate() / (self.dst_ip_f0_est.estimate() ** 2)
        DstPort_Concentration = self.dst_port_f2_est.estimate() / (self.dst_port_f0_est.estimate() ** 2)

        estimations = {
            'SrcIP_F0': self.src_ip_f0_est.estimate(),
            'SrcIP_F2': self.src_ip_f2_est.estimate(),
            'DstIP_F0': self.dst_ip_f0_est.estimate(),
            'DstIP_F2': self.dst_ip_f2_est.estimate(),
            'DstPortF0': self.dst_port_f0_est.estimate(),
            'NumEventsF1': self.num_events_f1_est.estimate(),
            'NumAttacksF1': self.attack_f1_estimator.estimate(),
            'SrcIP_FractionOfMass': SrcIP_FractionOfMass,
            'DstIP_FractionOfMass': DstIP_FractionOfMass,
            'DstPort_FractionOfMass': DstPort_FractionOfMass,
            'SrcIP_Concentration': SrcIP_Concentration,
            'DstIP_Concentration': DstIP_Concentration,
            'DstPort_Concentration': DstPort_Concentration,
        }
        # Attack detection
        estimations['Attack'] = int(self.attack_f1_estimator.estimate() / self.num_events_f1_est.estimate() > 0.06)

        return estimations

    @property
    def estimators(self):
        return [
            self.src_ip_f0,
            self.src_ip_f2,
            self.src_ip_cms,
            self.src_ip_heavy_hitter,
            self.dst_ip_f0,
            self.dst_ip_f2,
            self.dst_ip_cms,
            self.dst_ip_heavy_hitter,
            self.dst_port_f0,
            self.dst_port_f2,
            self.dst_port_cms,
            self.dst_port_heavy_hitter,
            self.num_events_f1,
            self.attack_f1_estimator,
        ]
