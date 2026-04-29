# baselines/throughput_based.py
#
# Rate-Based (RB) algorithm from the paper:
#   Estimate throughput as the harmonic mean of the last 5 chunk measurements,
#   then pick the highest available bitrate whose demand ≤ predicted throughput.

import numpy as np
from config import BITRATE_LADDER_KBPS, NUM_BITRATES, HISTORY_LEN


class ThroughputBased:
    def __init__(self, window: int = 5):
        self.window = min(window, HISTORY_LEN)

    def select_action(self, state: dict) -> int:
        past_tput = state["throughputs"][-self.window:]   # Mbps, float32 array
        # Filter out zero-padded history slots
        valid = past_tput[past_tput > 1e-6]
        if len(valid) == 0:
            return 0    # no history yet → conservative

        harmonic_mean_mbps = len(valid) / float(np.sum(1.0 / valid))

        # Pick highest bitrate whose kbps/1000 ≤ predicted Mbps
        for i in range(NUM_BITRATES - 1, -1, -1):
            if BITRATE_LADDER_KBPS[i] / 1000.0 <= harmonic_mean_mbps:
                return i
        return 0    # predicted throughput below all bitrates

    def reset(self):
        pass
