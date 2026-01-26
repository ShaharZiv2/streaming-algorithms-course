import pandas as pd

from estimators import FMEstimator, AMSEstimator, MorrisEstimator, CountMinSketch, HeavyHittersSketch
from schemas.args import Args


class SketchManager:

    def __init__(self, args: Args):
        self.src_ip_f0 = FMEstimator()
        self.src_ip_f2 = AMSEstimator(args.ams_r)
        self.src_ip_cms = CountMinSketch(args.cms_width)
        self.src_ip_heavy_hitter = HeavyHittersSketch()

        self.dst_ip_f0 = FMEstimator()
        self.dst_ip_f2 = AMSEstimator(args.ams_r)
        self.dst_ip_cms = CountMinSketch(args.cms_width)
        self.dst_ip_heavy_hitter = HeavyHittersSketch()

        self.dst_port_f0 = FMEstimator()
        self.dst_port_f2 = AMSEstimator(args.ams_r)
        self.dst_port_cms = CountMinSketch(args.cms_width)
        self.dst_port_heavy_hitter = HeavyHittersSketch()

        self.num_events_f1 = MorrisEstimator()
        self.attack_f1 = MorrisEstimator()

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

        self.dst_port_f0.update(event['dsport'])
        self.dst_port_f2.update(event['dsport'])
        self.dst_port_cms.update(event['dsport'])
        self.dst_port_heavy_hitter.update(event['dsport'])

        self.num_events_f1.update()

        if event['Label'] == '1':
            self.attack_f1.update()


    def reset(self):
        for estimator in self.estimators:
            estimator.reset()

    def estimate(self):
        # 1. Total Volume (F1) for normalization
        total_volume = self.num_events_f1.estimate()

        # F1 squared for burst index normalization
        f1_squared = total_volume ** 2

        # --- FEATURE D: SOURCE IP ---
        # Get top-k candidate keys from the candidate list
        src_candidates = self.src_ip_heavy_hitter.estimate()
        src_counts = [self.src_ip_cms.estimate(ip) for ip in src_candidates]

        # Feature D1: estimated maximum count among candidate keys
        SrcIP_MaxCount = max(src_counts) if src_counts else 0

        # Feature D2: fraction of mass in top-k candidates
        SrcIP_FractionOfMass = sum(src_counts) / total_volume

        # --- FEATURE D: DESTINATION IP ---
        # Get top-k candidate keys for destination IPs
        dst_ip_candidates = self.dst_ip_heavy_hitter.estimate()
        dst_ip_counts = [self.dst_ip_cms.estimate(ip) for ip in dst_ip_candidates]

        # Feature D1: estimated maximum count among candidate keys
        DstIP_MaxCount = max(dst_ip_counts) if dst_ip_counts else 0

        # Feature D2: fraction of mass in top-k candidates
        DstIP_FractionOfMass = sum(dst_ip_counts) / total_volume

        # --- FEATURE D: DESTINATION PORT ---
        # Get top-k candidate keys for destination ports
        dst_port_candidates = self.dst_port_heavy_hitter.estimate()
        dst_port_counts = [self.dst_port_cms.estimate(port) for port in dst_port_candidates]

        # Feature D1: estimated maximum count among candidate keys
        DstPort_MaxCount = max(dst_port_counts) if dst_port_counts else 0

        # Feature D2: fraction of mass in top-k candidates
        DstPort_FractionOfMass = sum(dst_port_counts) / total_volume

        # --- FEATURE C: BURST / CONCENTRATION ---
        # F2 estimates for source and destination IPs
        src_f2 = self.src_ip_f2.estimate()
        dst_f2 = self.dst_ip_f2.estimate()
        dst_port_f2 = self.dst_port_f2.estimate()

        # Burst Index: normalized by F1^2
        # Values closer to 1 indicate a single IP is dominating the traffic (burst)
        SrcIP_BurstIndex = src_f2 / f1_squared
        DstIP_BurstIndex = dst_f2 / f1_squared
        DstPort_BurstIndex = dst_port_f2 / f1_squared

        # Concentration: normalized by F0^2 (distinct elements)
        # Shows if burstiness is caused by a few specific actors
        src_f0 = self.src_ip_f0.estimate()
        dst_f0 = self.dst_ip_f0.estimate()
        dst_port_f0 = self.dst_port_f0.estimate()

        SrcIP_Concentration = src_f2 / (src_f0 ** 2)
        DstIP_Concentration = dst_f2 / (dst_f0 ** 2)
        DstPort_Concentration = dst_port_f2 / (dst_port_f0 ** 2)

        estimations = {
            # F0 Features (Distinct Elements)
            'SrcIPF0': src_f0,
            'DstIPF0': dst_f0,
            'DstPortF0': dst_port_f0,

            # F2 Features (Second Frequency Moment)
            'SrcIPF2': src_f2,
            'DstIPF2': dst_f2,
            'DstPortF2': dst_port_f2,

            # F1 Features (Total Volume)
            'Events_F1': total_volume,

            # Burst Index Features (F2/F1^2)
            'SrcIP_BurstIndex': SrcIP_BurstIndex,
            'DstIP_BurstIndex': DstIP_BurstIndex,
            'DstPort_BurstIndex': DstPort_BurstIndex,

            # Concentration Features (F2/F0^2)
            'SrcIP_Concentration': SrcIP_Concentration,
            'DstIP_Concentration': DstIP_Concentration,
            'DstPort_Concentration': DstPort_Concentration,

            # Heavy Hitters - MaxCount Features
            'SrcIP_MaxCount': SrcIP_MaxCount,
            'DstIP_MaxCount': DstIP_MaxCount,
            'DstPort_MaxCount': DstPort_MaxCount,

            # Heavy Hitters - Fraction of Mass Features
            'SrcIP_FractionOfMass': SrcIP_FractionOfMass,
            'DstIP_FractionOfMass': DstIP_FractionOfMass,
            'DstPort_FractionOfMass': DstPort_FractionOfMass,
        }

        # Attack detection
        attack_ratio = self.attack_f1.estimate() / total_volume
        estimations['Attack'] = int(attack_ratio > 0.06)

        return estimations

    def calculate_total_memory(self) -> int:
        """
        Calculate the total memory used by all estimators in bytes.

        Returns:
            int: Total memory in bytes
        """
        total_memory = 0
        for estimator in self.estimators:
            total_memory += estimator.report()
        return total_memory

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
            self.attack_f1,
        ]
