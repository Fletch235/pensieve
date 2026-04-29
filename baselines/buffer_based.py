# baselines/buffer_based.py
#
# Buffer-Based (BB) algorithm (Huang et al. 2014, §5.1 of the Pensieve paper):
#   Reservoir = 5 seconds  →  below this, always use lowest bitrate.
#   Cushion   = 10 seconds →  above reservoir+cushion, always use highest bitrate.
#   Between the two: linearly interpolate across the bitrate ladder.

from config import BITRATE_LADDER_KBPS, NUM_BITRATES


class BufferBased:
    RESERVOIR_SEC = 5.0
    CUSHION_SEC   = 10.0

    def select_action(self, state: dict) -> int:
        buf = float(state["buffer"])
        n   = NUM_BITRATES

        if buf < self.RESERVOIR_SEC:
            return 0
        elif buf >= self.RESERVOIR_SEC + self.CUSHION_SEC:
            return n - 1
        else:
            frac = (buf - self.RESERVOIR_SEC) / self.CUSHION_SEC
            return min(int(frac * n), n - 1)

    def reset(self):
        pass
