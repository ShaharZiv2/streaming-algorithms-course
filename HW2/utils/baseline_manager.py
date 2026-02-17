from abc import abstractmethod
from collections import Counter
from typing import Dict

from utils.sketch_manager_base import SketchManagerBase


class BaselineManager(SketchManagerBase):

    def __init__(self):
        self.src_ip_counter = Counter()
        self.dst_ip_counter = Counter()
        self.dst_port_counter = Counter()
        self.total_events = 0
        self.attacks = 0

    def sketch(self, event: Dict[str, str]) -> None:
        self.src_ip_counter[event['srcip']] += 1
        self.dst_ip_counter[event['dstip']] += 1
        self.dst_port_counter[event['dsport']] += 1


        self.total_events += 1

        if event['Label'] == '1':
            self.attacks += 1

    def estimate(self):

        # F1 squared for burst index normalization
        f1_squared = self.total_events ** 2

        # --- FEATURE D: SOURCE IP ---
        # Get top-k candidate keys from the candidate list
        src_ip_counts = [count for item, count in self.src_ip_counter.most_common(5)]

        # Feature D1: estimated maximum count among candidate keys
        SrcIP_MaxCount = max(src_ip_counts)

        # Feature D2: fraction of mass in top-k candidates
        SrcIP_FractionOfMass = sum(src_ip_counts) / self.total_events

        # --- FEATURE D: DESTINATION IP ---
        # Get top-k candidate keys for destination IPs
        dst_ip_counts = [count for item, count in self.dst_ip_counter.most_common(5)]

        # Feature D1: estimated maximum count among candidate keys
        DstIP_MaxCount = max(dst_ip_counts)

        # Feature D2: fraction of mass in top-k candidates
        DstIP_FractionOfMass = sum(dst_ip_counts) / self.total_events

        # --- FEATURE D: DESTINATION PORT ---
        # Get top-k candidate keys for destination ports
        dst_port_counts = [count for item, count in self.dst_port_counter.most_common(5)]

        # Feature D1: estimated maximum count among candidate keys
        DstPort_MaxCount = max(dst_port_counts)

        # Feature D2: fraction of mass in top-k candidates
        DstPort_FractionOfMass = sum(dst_port_counts) / self.total_events

        # --- FEATURE C: BURST / CONCENTRATION ---
        # F2 estimates for source and destination IPs
        src_f2 = sum([value**2 for value in self.src_ip_counter.values()])
        dst_f2 = sum([value**2 for value in self.dst_ip_counter.values()])
        dst_port_f2 = sum([value**2 for value in self.dst_port_counter.values()])

        # Burst Index: normalized by F1^2
        # Values closer to 1 indicate a single IP is dominating the traffic (burst)
        SrcIP_BurstIndex = src_f2 / f1_squared
        DstIP_BurstIndex = dst_f2 / f1_squared
        DstPort_BurstIndex = dst_port_f2 / f1_squared

        # Concentration: normalized by F0^2 (distinct elements)
        # Shows if burstiness is caused by a few specific actors
        src_f0 = len(self.src_ip_counter)
        dst_f0 = len(self.dst_ip_counter)
        dst_port_f0 = len(self.dst_port_counter)

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
            'Events_F1': self.total_events,

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
        attack_ratio = self.attacks / self.total_events
        estimations['Attack'] = int(attack_ratio > 0.06)

        return estimations

    def reset(self):
        self.__init__()

    def calculate_total_memory(self) -> int:
        return 0