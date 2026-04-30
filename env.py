# env.py — VideoStreamingEnv
#
# Chunk-level adaptive bitrate streaming simulator.
# Faithfully models Pensieve §4.1:
#   - Buffer drains during chunk download (video plays while we wait)
#   - Rebuffering = download_time exceeds current buffer
#   - Throughput is taken directly from the input trace (one value per chunk)

import random
import numpy as np
from typing import Optional

from config import (
    BITRATE_LADDER_KBPS, NUM_BITRATES, NUM_CHUNKS,
    CHUNK_DURATION_SEC, BUFFER_CAP_SEC, HISTORY_LEN,
    REBUFFER_PENALTY, SMOOTHNESS_PENALTY,
)
from video.chunk_sizes import CHUNK_SIZES


class VideoStreamingEnv:
    """Chunk-level DASH streaming simulator.

    Interface is intentionally gym-like (reset / step) but has no gym dependency.

    State dict keys
    ---------------
    throughputs     : float32 array (HISTORY_LEN,)  — past k chunk throughputs (Mbps)
    download_times  : float32 array (HISTORY_LEN,)  — past k download times (s)
    chunk_sizes     : float32 array (NUM_BITRATES,) — next chunk sizes at each bitrate (bytes)
    buffer          : float scalar                   — current buffer occupancy (s)
    chunks_remaining: float scalar                   — chunks left including current
    last_bitrate    : float scalar                   — last chosen bitrate index (0–5)
    rebuffer_history: float32 array (HISTORY_LEN,)  — seconds of rebuffer per past chunk
    bitrate_history : float32 array (HISTORY_LEN,)  — bitrate index chosen per past chunk
    tput_cv         : float scalar                   — throughput coeff. of variation
    buffer_fill_rate: float scalar                   — (buffer_now - buffer_prev) / CHUNK_DURATION_SEC
    """

    def __init__(self, trace: Optional[list[float]] = None):
        self._trace: list[float] = trace or []
        self._reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, trace: Optional[list[float]] = None) -> dict:
        """Start a new episode.  If trace is None the current trace is reused."""
        if trace is not None:
            self._trace = trace
        if not self._trace:
            raise ValueError("Provide a trace via reset(trace=...) or the constructor.")
        self._reset_state()
        return self._get_state()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        """Download one chunk at the chosen bitrate level.

        Parameters
        ----------
        action : int   index into BITRATE_LADDER_KBPS (0 = lowest, 5 = highest)

        Returns
        -------
        next_state, reward, done, info
        """
        assert 0 <= action < NUM_BITRATES, f"Invalid action {action}"
        assert self._chunk_idx < NUM_CHUNKS, "Episode already finished; call reset()."

        # --- Throughput for this chunk ----------------------------------------
        trace_idx   = min(self._trace_idx, len(self._trace) - 1)
        tput_mbps   = max(self._trace[trace_idx], 1e-6)   # guard zero-division
        tput_bytes_per_sec = tput_mbps * 1e6 / 8.0

        # --- Download time (seconds) ------------------------------------------
        chunk_bytes    = CHUNK_SIZES[self._chunk_idx][action]
        download_time  = chunk_bytes / tput_bytes_per_sec

        # --- Buffer dynamics --------------------------------------------------
        # Video plays during download; buffer drains at 1 second per second.
        buffer_before = self._buffer
        drain         = download_time

        if drain > buffer_before:
            rebuffer_sec  = drain - buffer_before
            buffer_after  = CHUNK_DURATION_SEC        # start fresh after rebuffer
        else:
            rebuffer_sec  = 0.0
            buffer_after  = buffer_before - drain + CHUNK_DURATION_SEC

        buffer_after = min(buffer_after, BUFFER_CAP_SEC)

        # --- Reward (QoE_lin) -------------------------------------------------
        bitrate_mbps      = BITRATE_LADDER_KBPS[action] / 1000.0
        prev_bitrate_mbps = BITRATE_LADDER_KBPS[self._last_action] / 1000.0

        reward = (
            bitrate_mbps
            - REBUFFER_PENALTY   * rebuffer_sec
            - SMOOTHNESS_PENALTY * abs(bitrate_mbps - prev_bitrate_mbps)
        )

        # --- Update throughput measurement used by this step ------------------
        measured_tput = (BITRATE_LADDER_KBPS[action] / 1000.0) / max(download_time / CHUNK_DURATION_SEC, 1e-6)
        # (We record the effective throughput experienced, not the trace value,
        #  to match what a real player would observe.)
        measured_tput = chunk_bytes * 8 / 1e6 / download_time    # actual Mbps

        # --- Advance state ----------------------------------------------------
        self._tput_history     = np.roll(self._tput_history,     -1)
        self._dl_time_history  = np.roll(self._dl_time_history,  -1)
        self._rebuffer_history = np.roll(self._rebuffer_history, -1)
        self._bitrate_history  = np.roll(self._bitrate_history,  -1)
        self._tput_history[-1]     = measured_tput
        self._dl_time_history[-1]  = download_time
        self._rebuffer_history[-1] = rebuffer_sec
        self._bitrate_history[-1]  = float(action)

        self._prev_buffer = buffer_before
        self._buffer      = buffer_after
        self._last_action = action
        self._chunk_idx  += 1
        self._trace_idx  += 1

        done = self._chunk_idx >= NUM_CHUNKS
        info = {
            "rebuffer":     rebuffer_sec,
            "bitrate_mbps": bitrate_mbps,
            "buffer":       buffer_after,
            "download_time": download_time,
            "throughput":   measured_tput,
        }

        return self._get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._chunk_idx        = 0
        self._trace_idx        = 0
        self._buffer           = 0.0
        self._last_action      = 0
        self._tput_history     = np.zeros(HISTORY_LEN, dtype=np.float32)
        self._dl_time_history  = np.zeros(HISTORY_LEN, dtype=np.float32)
        self._rebuffer_history = np.zeros(HISTORY_LEN, dtype=np.float32)
        self._bitrate_history  = np.zeros(HISTORY_LEN, dtype=np.float32)
        self._prev_buffer      = 0.0

    def _get_state(self) -> dict:
        next_chunk_idx = min(self._chunk_idx, NUM_CHUNKS - 1)
        next_sizes = np.array(
            [CHUNK_SIZES[next_chunk_idx][b] for b in range(NUM_BITRATES)],
            dtype=np.float32,
        )
        # Throughput coefficient of variation (0 when no history yet)
        tput_mean = float(np.mean(self._tput_history))
        tput_std  = float(np.std(self._tput_history))
        tput_cv   = tput_std / (tput_mean + 1e-6)

        buffer_fill_rate = (self._buffer - self._prev_buffer) / CHUNK_DURATION_SEC

        return {
            "throughputs":      self._tput_history.copy(),
            "download_times":   self._dl_time_history.copy(),
            "chunk_sizes":      next_sizes,
            "buffer":           np.float32(self._buffer),
            "chunks_remaining": np.float32(NUM_CHUNKS - self._chunk_idx),
            "last_bitrate":     np.float32(self._last_action),
            "rebuffer_history": self._rebuffer_history.copy(),
            "bitrate_history":  self._bitrate_history.copy(),
            "tput_cv":          np.float32(tput_cv),
            "buffer_fill_rate": np.float32(buffer_fill_rate),
        }


# ---------------------------------------------------------------------------
# Unit tests — run with:  python env.py
# ---------------------------------------------------------------------------

def _test_no_rebuffer_on_infinite_throughput():
    """Very high throughput → near-zero rebuffer (only first-chunk startup), buffer grows toward cap."""
    trace = [1000.0] * 100    # 1 Gbps
    env = VideoStreamingEnv()
    state = env.reset(trace=trace)
    total_rebuffer = 0.0
    done = False
    while not done:
        _, reward, done, info = env.step(NUM_BITRATES - 1)   # always max bitrate
        total_rebuffer += info["rebuffer"]
    # Buffer starts empty so first chunk always has a tiny startup stall; tolerate up to 0.1s
    assert total_rebuffer < 0.1, f"Expected near-zero rebuffer, got {total_rebuffer:.3f}s"
    print("PASS  no_rebuffer_on_infinite_throughput")


def _test_always_rebuffers_on_zero_throughput():
    """Near-zero throughput → rebuffer on every chunk."""
    trace = [0.001] * 100   # 1 kbps
    env = VideoStreamingEnv()
    env.reset(trace=trace)
    rebuffer_events = 0
    done = False
    while not done:
        _, _, done, info = env.step(0)     # lowest bitrate
        if info["rebuffer"] > 0:
            rebuffer_events += 1
    assert rebuffer_events == NUM_CHUNKS, (
        f"Expected {NUM_CHUNKS} rebuffer events, got {rebuffer_events}"
    )
    print("PASS  always_rebuffers_on_zero_throughput")


def _test_buffer_cap():
    """Buffer never exceeds BUFFER_CAP_SEC."""
    trace = [1000.0] * 200
    env = VideoStreamingEnv()
    env.reset(trace=trace)
    done = False
    while not done:
        _, _, done, info = env.step(0)    # lowest bitrate, fastest download
        assert info["buffer"] <= BUFFER_CAP_SEC + 1e-9, (
            f"Buffer exceeded cap: {info['buffer']:.2f}s"
        )
    print("PASS  buffer_cap")


def _test_episode_length():
    """Episode terminates after exactly NUM_CHUNKS steps."""
    trace = [2.0] * 200
    env = VideoStreamingEnv()
    env.reset(trace=trace)
    steps = 0
    done = False
    while not done:
        _, _, done, _ = env.step(2)
        steps += 1
    assert steps == NUM_CHUNKS, f"Expected {NUM_CHUNKS} steps, got {steps}"
    print("PASS  episode_length")


def _test_determinism():
    """Same trace + same actions → identical rewards."""
    trace = [2.5, 1.2, 3.0, 0.8] * 20
    actions = [3, 2, 4, 1, 5, 0, 2, 3] * 6
    rewards_a, rewards_b = [], []

    for rewards in (rewards_a, rewards_b):
        env = VideoStreamingEnv()
        env.reset(trace=trace)
        for i, a in enumerate(actions[:NUM_CHUNKS]):
            _, r, done, _ = env.step(a)
            rewards.append(r)
            if done:
                break

    assert rewards_a == rewards_b, "Determinism check failed"
    print("PASS  determinism")


def _test_fixed_bitrate_hand_calculation():
    """Verify reward for a single chunk against manual computation."""
    tput_mbps = 2.0                      # Mbps
    action    = 2                        # 1200 kbps
    chunk_bytes = CHUNK_SIZES[0][action]
    tput_bytes_per_sec = tput_mbps * 1e6 / 8.0
    download_time = chunk_bytes / tput_bytes_per_sec

    # buffer starts at 0 → rebuffer = download_time - 0 = download_time
    rebuffer_expected = download_time
    bitrate_mbps      = BITRATE_LADDER_KBPS[action] / 1000.0
    # last_action starts at 0 → prev_bitrate = 0.3 Mbps
    smoothness = abs(bitrate_mbps - BITRATE_LADDER_KBPS[0] / 1000.0)
    reward_expected = (
        bitrate_mbps
        - REBUFFER_PENALTY * rebuffer_expected
        - SMOOTHNESS_PENALTY * smoothness
    )

    env = VideoStreamingEnv()
    env.reset(trace=[tput_mbps] * 100)
    _, reward, _, info = env.step(action)

    assert abs(reward - reward_expected) < 1e-6, (
        f"Expected reward {reward_expected:.4f}, got {reward:.4f}"
    )
    assert abs(info["rebuffer"] - rebuffer_expected) < 1e-6
    print("PASS  fixed_bitrate_hand_calculation")


if __name__ == "__main__":
    _test_no_rebuffer_on_infinite_throughput()
    _test_always_rebuffers_on_zero_throughput()
    _test_buffer_cap()
    _test_episode_length()
    _test_determinism()
    _test_fixed_bitrate_hand_calculation()
    print("\nAll env unit tests passed.")
