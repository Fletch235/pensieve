# baselines/fixed_bitrate.py

from config import NUM_BITRATES


class FixedBitrate:
    """Always selects the same bitrate level regardless of network conditions."""

    def __init__(self, bitrate_idx: int):
        assert 0 <= bitrate_idx < NUM_BITRATES
        self.bitrate_idx = bitrate_idx

    def select_action(self, state: dict) -> int:
        return self.bitrate_idx

    def reset(self):
        pass
