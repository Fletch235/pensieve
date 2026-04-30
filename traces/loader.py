# traces/loader.py
#
# Load network traces from disk or generate synthetic ones.
# Supports the plain-text format used by the public Pensieve repo:
#   One throughput value (Mbps) per line.  Lines starting with '#' are skipped.
#
# Real traces can be downloaded from:
#   https://github.com/hongzimao/pensieve  →  sim/cooked_traces/
#   https://github.com/hongzimao/pensieve  →  sim/cooked_test_traces/

import os
import random
from pathlib import Path
from typing import Optional

from config import TRAIN_FRAC, VAL_FRAC
from traces.synthetic import generate_trace_set


def load_trace(filepath: str | Path) -> list[float]:
    """Load a single trace file.  Returns a list of Mbps floats.

    Handles three formats:
      - Single column:           value_mbps
      - Two columns (FCC raw):   timestamp_s   value_kbps   (median > 10 → kbps)
      - Two columns (HSDPA log): timestamp_s   value_mbps   (median ≤ 10 → Mbps)
    """
    single_col: list[float] = []
    two_col: list[float] = []
    is_two_col = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                is_two_col = True
                two_col.append(float(parts[1]))
            else:
                single_col.append(float(parts[0]))

    if not is_two_col:
        return single_col

    # Auto-detect kbps vs Mbps: FCC kbps files have values in the hundreds;
    # HSDPA Mbps files have values below ~10.
    median = sorted(two_col)[len(two_col) // 2]
    scale = 1000.0 if median > 10.0 else 1.0
    return [v / scale for v in two_col]


def load_trace_dir(
    trace_dir: str | Path,
    min_tput_mbps: float = 0.2,
    max_tput_mbps: float = 6.0,
) -> list[list[float]]:
    """Load all trace files from a directory.

    Applies the same quality filter as the original Pensieve paper:
      - minimum throughput  > min_tput_mbps  (default 0.2 Mbps)
      - average throughput  < max_tput_mbps  (default 6.0 Mbps)
    Traces with fewer than 10 entries are also discarded.
    """
    trace_dir = Path(trace_dir)
    traces = []
    for p in sorted(trace_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            try:
                t = load_trace(p)
                if len(t) < 10:
                    continue
                if min(t) < min_tput_mbps:
                    continue
                if sum(t) / len(t) > max_tput_mbps:
                    continue
                traces.append(t)
            except (ValueError, UnicodeDecodeError):
                pass
    return traces


def get_splits(
    traces: list[list[float]],
    seed: int = 42,
) -> dict[str, list[list[float]]]:
    """Shuffle traces and split into train / val / test.

    The split is deterministic given the seed, so train and test sets never
    overlap even across multiple calls.
    """
    shuffled = traces.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


def get_synthetic_splits(
    n_traces: int = 1000,
    n_steps: int = 80,
    seed: int = 42,
) -> dict[str, list[list[float]]]:
    """Convenience: generate synthetic traces and split them."""
    traces = generate_trace_set(n_traces, n_steps=n_steps, base_seed=seed)
    return get_splits(traces, seed=seed)


def load_or_generate(
    trace_dir: Optional[str | Path] = None,
    n_synthetic: int = 1000,
    seed: int | float = 42,
) -> dict[str, list[list[float]]]:
    """Load real traces if trace_dir exists, otherwise use synthetic."""
    seed = int(seed)
    if trace_dir is not None and Path(trace_dir).is_dir():
        traces = load_trace_dir(trace_dir)
        if traces:
            print(f"Loaded {len(traces)} real traces from {trace_dir}")
            return get_splits(traces, seed=seed)
        print(f"No traces found in {trace_dir}, falling back to synthetic.")
    print(f"Using {n_synthetic} synthetic traces.")
    return get_synthetic_splits(n_synthetic, seed=seed)


if __name__ == "__main__":
    splits = get_synthetic_splits(200)
    for name, t in splits.items():
        print(f"{name}: {len(t)} traces")
