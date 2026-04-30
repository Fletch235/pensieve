# model/network.py
#
# Actor-Critic neural network for ABR bitrate selection.
# Architecture exactly matches Figure 5 of the Pensieve paper (SIGCOMM 2017):
#   - Separate 1D-CNN branches for throughput history, download-time history,
#     and next-chunk sizes
#   - Scalar inputs (buffer, chunks_remaining, last_bitrate) concatenated directly
#   - Shared 128-neuron hidden layer → actor softmax head + critic linear head

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HISTORY_LEN, NUM_BITRATES, BUFFER_CAP_SEC, NUM_CHUNKS

# Normalisation constants (match the paper's intuition)
_MAX_TPUT_MBPS   = 10.0      # clip measured throughput above this
_MAX_CHUNK_BYTES = 3_000_000 # ≈ max chunk size at 4300 kbps × 4s
_CNN_FILTERS     = 128
_CNN_KERNEL      = 4         # kernel size used for all 1D-CNNs
_HIDDEN_SIZE     = 128


def _conv_out_len(in_len: int, kernel: int = _CNN_KERNEL) -> int:
    return in_len - kernel + 1


# Input widths after flattening each CNN branch
_TPUT_FLAT     = _CNN_FILTERS * _conv_out_len(HISTORY_LEN)     # 128 × 5 = 640
_TIME_FLAT     = _CNN_FILTERS * _conv_out_len(HISTORY_LEN)     # 128 × 5 = 640
_SIZES_FLAT    = _CNN_FILTERS * _conv_out_len(NUM_BITRATES)    # 128 × 3 = 384
_REBUF_FLAT    = _CNN_FILTERS * _conv_out_len(HISTORY_LEN)     # 128 × 5 = 640
_BRHIST_FLAT   = _CNN_FILTERS * _conv_out_len(HISTORY_LEN)     # 128 × 5 = 640
_SCALAR_DIM    = 5   # buffer, remain, last_br, tput_cv, buffer_fill_rate
_CONCAT_SIZE   = _TPUT_FLAT + _TIME_FLAT + _SIZES_FLAT + _REBUF_FLAT + _BRHIST_FLAT + _SCALAR_DIM  # 2949


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network.

    forward() returns (logits, value).  The caller samples from logits
    via torch.distributions.Categorical.
    """

    def __init__(self, num_bitrates: int = NUM_BITRATES):
        super().__init__()
        self.num_bitrates = num_bitrates

        # 1D-CNN branches — each takes (batch, 1, seq_len)
        self.tput_conv   = nn.Conv1d(1, _CNN_FILTERS, kernel_size=_CNN_KERNEL)
        self.time_conv   = nn.Conv1d(1, _CNN_FILTERS, kernel_size=_CNN_KERNEL)
        self.size_conv   = nn.Conv1d(1, _CNN_FILTERS, kernel_size=_CNN_KERNEL)
        self.rebuf_conv  = nn.Conv1d(1, _CNN_FILTERS, kernel_size=_CNN_KERNEL)
        self.brhist_conv = nn.Conv1d(1, _CNN_FILTERS, kernel_size=_CNN_KERNEL)

        # Shared hidden layer
        self.hidden = nn.Linear(_CONCAT_SIZE, _HIDDEN_SIZE)

        # Output heads
        self.actor_head  = nn.Linear(_HIDDEN_SIZE, num_bitrates)
        self.critic_head = nn.Linear(_HIDDEN_SIZE, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state values are plain tensors, shape (batch, dim).
        Returns: logits (batch, num_bitrates), value (batch, 1)
        """
        tput   = state["throughputs"]        # (batch, HISTORY_LEN)
        times  = state["download_times"]      # (batch, HISTORY_LEN)
        sizes  = state["chunk_sizes"]          # (batch, NUM_BITRATES)
        buf    = state["buffer"]               # (batch, 1)
        remain = state["chunks_remaining"]    # (batch, 1)
        last_r = state["last_bitrate"]        # (batch, 1)
        rebuf  = state["rebuffer_history"]    # (batch, HISTORY_LEN)
        brhist = state["bitrate_history"]     # (batch, HISTORY_LEN)
        cv     = state["tput_cv"]              # (batch, 1)
        bfr    = state["buffer_fill_rate"]    # (batch, 1)

        # Unsqueeze channel dim for Conv1d: (batch, 1, seq_len)
        t1 = F.relu(self.tput_conv(tput.unsqueeze(1))).flatten(1)      # (batch, 640)
        t2 = F.relu(self.time_conv(times.unsqueeze(1))).flatten(1)     # (batch, 640)
        t3 = F.relu(self.size_conv(sizes.unsqueeze(1))).flatten(1)     # (batch, 384)
        t4 = F.relu(self.rebuf_conv(rebuf.unsqueeze(1))).flatten(1)    # (batch, 640)
        t5 = F.relu(self.brhist_conv(brhist.unsqueeze(1))).flatten(1)  # (batch, 640)

        x = torch.cat([t1, t2, t3, t4, t5, buf, remain, last_r, cv, bfr], dim=1)  # (batch, 2949)
        h = F.relu(self.hidden(x))                                      # (batch, 128)

        logits = self.actor_head(h)   # (batch, num_bitrates) — raw, no softmax
        value  = self.critic_head(h)  # (batch, 1)
        return logits, value


def normalise_state(state: dict, device: torch.device) -> dict:
    """Convert a numpy state dict (from env.step) into normalised tensors.

    All outputs are 2-D (1, dim) so they can be batched trivially.
    """
    def t(arr, norm=1.0):
        a = np.asarray(arr, dtype=np.float32)
        return torch.tensor(a / norm, dtype=torch.float32, device=device).unsqueeze(0)

    return {
        "throughputs":      t(state["throughputs"],      _MAX_TPUT_MBPS),
        "download_times":   t(state["download_times"],   BUFFER_CAP_SEC),
        "chunk_sizes":      t(state["chunk_sizes"],      _MAX_CHUNK_BYTES),
        "buffer":           t([state["buffer"]],         BUFFER_CAP_SEC),
        "chunks_remaining": t([state["chunks_remaining"]], NUM_CHUNKS),
        "last_bitrate":     t([state["last_bitrate"]],   NUM_BITRATES - 1),
        "rebuffer_history": t(state["rebuffer_history"], BUFFER_CAP_SEC),
        "bitrate_history":  t(state["bitrate_history"],  NUM_BITRATES - 1),
        "tput_cv":          t([state["tput_cv"]],        3.0),   # CV rarely exceeds 3
        "buffer_fill_rate": t([state["buffer_fill_rate"]], 1.0), # already in [−1, 1] range
    }


if __name__ == "__main__":
    import numpy as np

    model = ActorCritic()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Smoke-test with random state
    dummy = {
        "throughputs":      np.random.rand(HISTORY_LEN).astype(np.float32),
        "download_times":   np.random.rand(HISTORY_LEN).astype(np.float32),
        "chunk_sizes":      np.random.rand(NUM_BITRATES).astype(np.float32),
        "buffer":           np.float32(10.0),
        "chunks_remaining": np.float32(24.0),
        "last_bitrate":     np.float32(2.0),
        "rebuffer_history": np.zeros(HISTORY_LEN, dtype=np.float32),
        "bitrate_history":  np.random.randint(0, NUM_BITRATES, HISTORY_LEN).astype(np.float32),
        "tput_cv":          np.float32(0.3),
        "buffer_fill_rate": np.float32(0.1),
    }
    normed = normalise_state(dummy, torch.device("cpu"))
    logits, value = model(normed)
    print(f"logits shape: {logits.shape}, value shape: {value.shape}")
    probs = torch.softmax(logits, dim=-1)
    print(f"action probs: {probs.detach().numpy().round(3)}")
    print("PASS  network smoke-test")
