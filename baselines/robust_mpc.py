# baselines/robust_mpc.py
#
# robustMPC — as described in §6.2 of the Pensieve paper (SIGCOMM 2017).
#
# Algorithm:
#   1. Predict next-chunk throughput = harmonic_mean(last 5 chunks)
#              × (1 / max_past_error_ratio)
#      where max_past_error_ratio = max over recent history of
#              (predicted_tput / actual_tput)
#      Conservative downscaling guards against over-optimism.
#   2. Enumerate all bitrate sequences of length HORIZON (default 5).
#   3. For each sequence, simulate the buffer + QoE using the predicted
#      throughput for each future chunk, using next-chunk sizes from the
#      current state (same size assumed for all future chunks beyond the next).
#   4. Pick the first action of the sequence with the highest QoE.
#
# State keys used:
#   throughputs      (HISTORY_LEN,)  — recent measured Mbps
#   chunk_sizes      (NUM_BITRATES,) — sizes of the NEXT chunk at each bitrate
#   buffer           scalar          — current buffer (seconds)
#   last_bitrate     scalar          — previous bitrate index
#
# Note: we do not use download_times, rebuffer_history or the new feature
# fields — robustMPC is a purely rule-based agent.

import itertools
import numpy as np

from config import (
    BITRATE_LADDER_KBPS, NUM_BITRATES,
    CHUNK_DURATION_SEC, BUFFER_CAP_SEC,
    REBUFFER_PENALTY, SMOOTHNESS_PENALTY,
    HISTORY_LEN,
)
from video.chunk_sizes import CHUNK_SIZES


class RobustMPC:
    """robustMPC baseline (5-chunk lookahead, conservative throughput estimate)."""

    HORIZON       = 5
    ERROR_WINDOW  = 5   # past prediction errors to track

    def __init__(self, horizon: int = HORIZON):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self._past_errors: list[float] = []   # ratio: predicted / actual
        self._prev_predicted: float    = 0.0

    # ------------------------------------------------------------------

    def select_action(self, state: dict, chunk_idx: int | None = None) -> int:
        """Return the best bitrate index for the current chunk.

        Parameters
        ----------
        state     : raw env state dict (numpy arrays / scalars)
        chunk_idx : current chunk index (0-based). If None, inferred from
                    chunks_remaining (less accurate near video end).
        """
        tput_history  = np.asarray(state["throughputs"], dtype=np.float64)
        chunk_sizes   = np.asarray(state["chunk_sizes"],  dtype=np.float64)
        buf           = float(state["buffer"])
        last_br       = int(round(float(state["last_bitrate"])))
        chunks_left   = int(round(float(state["chunks_remaining"])))

        # --- Throughput prediction -------------------------------------------
        valid = tput_history[tput_history > 1e-6]
        if len(valid) == 0:
            harmonic = BITRATE_LADDER_KBPS[0] / 1000.0
        else:
            window = valid[-min(5, len(valid)):]
            harmonic = len(window) / float(np.sum(1.0 / window))

        # Conservative scale: divide by max past prediction-error ratio
        if self._past_errors:
            max_ratio = max(self._past_errors)
        else:
            max_ratio = 1.0
        predicted_tput = harmonic / max(max_ratio, 1.0)

        # Record this prediction so we can compute error on next call
        self._prev_predicted = predicted_tput

        # --- Determine current chunk index -----------------------------------
        if chunk_idx is not None:
            cur_idx = chunk_idx
        else:
            total_chunks = len(CHUNK_SIZES)
            cur_idx = max(0, total_chunks - chunks_left)

        # --- Horizon search --------------------------------------------------
        best_qoe    = -float("inf")
        best_action = 0

        for actions in itertools.product(range(NUM_BITRATES), repeat=self.horizon):
            qoe, _ = self._simulate(
                actions, cur_idx, buf, last_br, predicted_tput
            )
            if qoe > best_qoe:
                best_qoe    = qoe
                best_action = actions[0]

        return best_action

    def update_error(self, actual_tput: float):
        """Call after each chunk download with the observed throughput.

        Keeps a rolling window of prediction-error ratios.
        """
        if self._prev_predicted > 0 and actual_tput > 1e-6:
            ratio = self._prev_predicted / actual_tput
            self._past_errors.append(ratio)
            if len(self._past_errors) > self.ERROR_WINDOW:
                self._past_errors.pop(0)

    # ------------------------------------------------------------------
    # Internal simulation helper
    # ------------------------------------------------------------------

    def _simulate(
        self,
        actions: tuple[int, ...],
        start_chunk: int,
        buf: float,
        prev_action: int,
        predicted_tput: float,
    ) -> tuple[float, float]:
        """Simulate *horizon* steps and return (total_qoe, final_buffer)."""
        total_qoe = 0.0
        tput_bytes = predicted_tput * 1e6 / 8.0
        n_chunks   = len(CHUNK_SIZES)

        for step, action in enumerate(actions):
            chunk_idx = min(start_chunk + step, n_chunks - 1)
            chunk_bytes = CHUNK_SIZES[chunk_idx][action]

            dl_time = chunk_bytes / max(tput_bytes, 1e-9)

            if dl_time > buf:
                rebuffer = dl_time - buf
                buf      = CHUNK_DURATION_SEC
            else:
                rebuffer = 0.0
                buf      = min(buf - dl_time + CHUNK_DURATION_SEC, BUFFER_CAP_SEC)

            br_mbps   = BITRATE_LADDER_KBPS[action]    / 1000.0
            prev_mbps = BITRATE_LADDER_KBPS[prev_action] / 1000.0
            total_qoe += (
                br_mbps
                - REBUFFER_PENALTY   * rebuffer
                - SMOOTHNESS_PENALTY * abs(br_mbps - prev_mbps)
            )
            prev_action = action

        return total_qoe, buf
