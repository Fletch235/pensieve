# traces/synthetic.py
#
# Markovian synthetic network trace generator.
# Matches the distribution described in Pensieve §5.3:
#   - State = average throughput (Mbps), drawn from a discrete grid
#   - Transitions follow a geometric distribution (nearby states more likely)
#   - Actual throughput per slot = Gaussian(state_mean, variance)
#   - Variance is uniform in [0.05, 0.5] per trace

import numpy as np
from typing import Optional


def generate_markov_trace(
    n_steps: int = 80,
    avg_range: tuple[float, float] = (0.2, 4.3),
    n_states: int = 20,
    noise_variance_range: tuple[float, float] = (0.05, 0.5),
    seed: Optional[int] = None,
) -> list[float]:
    """Return a list of throughput values (Mbps), one per trace slot.

    Args:
        n_steps: number of slots (80 slots × 4s/chunk ≈ 320 seconds).
        avg_range: (min, max) Mbps for the state means.
        n_states: number of discrete Markov states.
        noise_variance_range: (min, max) for per-trace Gaussian noise std.
        seed: RNG seed for reproducibility.

    Returns:
        List of floats, each ≥ 0.01 Mbps.
    """
    rng = np.random.default_rng(seed)

    state_means = np.linspace(avg_range[0], avg_range[1], n_states)
    noise_std = rng.uniform(*noise_variance_range)

    # Start in a random state
    state = rng.integers(0, n_states)
    trace: list[float] = []

    for _ in range(n_steps):
        # Sample actual throughput from Gaussian around state mean
        tput = rng.normal(state_means[state], noise_std)
        trace.append(max(0.01, float(tput)))

        # Geometric transition: stay put with high probability; otherwise
        # move ±1 with equal probability (clamped to [0, n_states-1])
        if rng.random() < 0.1:           # ~10% chance of transition per step
            delta = rng.choice([-1, 1])
            state = int(np.clip(state + delta, 0, n_states - 1))

    return trace


def generate_trace_set(
    n_traces: int,
    n_steps: int = 80,
    base_seed: int = 0,
) -> list[list[float]]:
    """Generate a reproducible set of synthetic traces."""
    return [
        generate_markov_trace(n_steps=n_steps, seed=base_seed + i)
        for i in range(n_traces)
    ]


if __name__ == "__main__":
    traces = generate_trace_set(5, seed=0) if False else generate_trace_set(5)
    for i, t in enumerate(traces):
        avg = sum(t) / len(t)
        print(f"Trace {i}: {len(t)} steps, avg={avg:.2f} Mbps, "
              f"min={min(t):.2f}, max={max(t):.2f}")
