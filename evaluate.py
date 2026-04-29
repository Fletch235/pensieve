# evaluate.py — evaluation, metrics, and plots
#
# Usage:
#   python evaluate.py                          # use best_model.pt, synthetic traces
#   python evaluate.py --model checkpoints/ckpt_050000.pt --traces path/to/dir
#
# Outputs (written to results/):
#   metrics_table.csv         — per-agent aggregate metrics
#   qoe_cdf.png               — CDF of per-trace average QoE
#   qoe_components.png        — bitrate utility / rebuffer / smoothness breakdown
#   rollout_trace_{i}.png     — 3 representative timeline rollouts

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

from config import (
    BITRATE_LADDER_KBPS, NUM_BITRATES, NUM_CHUNKS,
    REBUFFER_PENALTY, SMOOTHNESS_PENALTY, RESULTS_DIR,
    CHUNK_DURATION_SEC,
)
from env import VideoStreamingEnv
from model.network import ActorCritic, normalise_state
from train import load_checkpoint, set_seeds
from traces.loader import load_or_generate

from baselines.fixed_bitrate   import FixedBitrate
from baselines.throughput_based import ThroughputBased
from baselines.buffer_based    import BufferBased


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def rollout_baseline(env, agent, trace) -> dict:
    """Run one episode with a heuristic agent.  Returns summary dict."""
    state = env.reset(trace=trace)
    done  = False
    total_qoe = 0.0
    total_rebuffer = 0.0
    smoothness_pen = 0.0
    bitrate_util   = 0.0
    prev_br_mbps   = BITRATE_LADDER_KBPS[0] / 1000.0
    bitrates_chosen: list[float] = []
    buffers:         list[float] = []
    throughputs:     list[float] = []
    rebuffers:       list[float] = []

    while not done:
        action = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_qoe      += reward
        total_rebuffer += info["rebuffer"]
        br_mbps = BITRATE_LADDER_KBPS[action] / 1000.0
        smoothness_pen += abs(br_mbps - prev_br_mbps)
        bitrate_util   += br_mbps
        prev_br_mbps    = br_mbps
        bitrates_chosen.append(br_mbps)
        buffers.append(info["buffer"])
        throughputs.append(info["throughput"])
        rebuffers.append(info["rebuffer"])

    return {
        "total_qoe":       total_qoe,
        "avg_qoe":         total_qoe / NUM_CHUNKS,
        "total_rebuffer":  total_rebuffer,
        "avg_bitrate":     bitrate_util / NUM_CHUNKS,
        "smoothness":      smoothness_pen / NUM_CHUNKS,
        # timeline data
        "bitrates":   bitrates_chosen,
        "buffers":    buffers,
        "throughputs": throughputs,
        "rebuffers":  rebuffers,
    }


def rollout_rl(env, model, trace, device, greedy=True) -> dict:
    """Run one episode with the trained RL policy."""
    state = env.reset(trace=trace)
    done  = False
    total_qoe = 0.0
    total_rebuffer = 0.0
    smoothness_pen = 0.0
    bitrate_util   = 0.0
    prev_br_mbps   = BITRATE_LADDER_KBPS[0] / 1000.0
    bitrates_chosen: list[float] = []
    buffers:         list[float] = []
    throughputs:     list[float] = []
    rebuffers:       list[float] = []

    while not done:
        normed = normalise_state(state, device)
        with torch.no_grad():
            logits, _ = model(normed)
        action = logits.argmax(dim=-1).item() if greedy else Categorical(logits=logits).sample().item()
        state, reward, done, info = env.step(action)
        total_qoe      += reward
        total_rebuffer += info["rebuffer"]
        br_mbps = BITRATE_LADDER_KBPS[action] / 1000.0
        smoothness_pen += abs(br_mbps - prev_br_mbps)
        bitrate_util   += br_mbps
        prev_br_mbps    = br_mbps
        bitrates_chosen.append(br_mbps)
        buffers.append(info["buffer"])
        throughputs.append(info["throughput"])
        rebuffers.append(info["rebuffer"])

    return {
        "total_qoe":       total_qoe,
        "avg_qoe":         total_qoe / NUM_CHUNKS,
        "total_rebuffer":  total_rebuffer,
        "avg_bitrate":     bitrate_util / NUM_CHUNKS,
        "smoothness":      smoothness_pen / NUM_CHUNKS,
        "bitrates":   bitrates_chosen,
        "buffers":    buffers,
        "throughputs": throughputs,
        "rebuffers":  rebuffers,
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_all(agents: dict, env, test_traces, device) -> dict[str, list[dict]]:
    """Run every agent on every test trace.  Returns {name: [result_per_trace]}."""
    all_results = {name: [] for name in agents}
    for i, trace in enumerate(test_traces):
        if (i + 1) % 20 == 0:
            print(f"  evaluated {i+1}/{len(test_traces)} traces")
        for name, agent in agents.items():
            if name == "rl_policy":
                model, model_device = agent
                res = rollout_rl(env, model, trace, model_device)
            else:
                res = rollout_baseline(env, agent, trace)
            all_results[name].append(res)
    return all_results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1",
]


def plot_qoe_cdf(all_results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, results), color in zip(all_results.items(), COLORS):
        avg_qoes = sorted(r["avg_qoe"] for r in results)
        cdf = np.linspace(0, 1, len(avg_qoes))
        ax.plot(avg_qoes, cdf, label=name, color=color, linewidth=2)
    ax.set_xlabel("Average QoE per trace", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("QoE CDF across test traces", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_qoe_components(all_results: dict, out_path: str):
    names = list(all_results.keys())
    avg_br  = [np.mean([r["avg_bitrate"]   for r in all_results[n]]) for n in names]
    avg_reb = [np.mean([r["total_rebuffer"] / NUM_CHUNKS for r in all_results[n]]) * REBUFFER_PENALTY for n in names]
    avg_smo = [np.mean([r["smoothness"]    for r in all_results[n]]) * SMOOTHNESS_PENALTY for n in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, avg_br,  width, label="Bitrate utility (Mbps)", color="#4e79a7")
    ax.bar(x,         avg_reb, width, label=f"Rebuffer penalty (×{REBUFFER_PENALTY})", color="#e15759")
    ax.bar(x + width, avg_smo, width, label="Smoothness penalty", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("QoE component breakdown", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_timeline(results_for_trace: dict, trace_throughputs: list[float],
                  out_path: str, title: str = ""):
    """3-panel timeline plot: bitrate / buffer / throughput."""
    chunks = list(range(1, NUM_CHUNKS + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for (name, res), color in zip(results_for_trace.items(), COLORS):
        axes[0].plot(chunks, res["bitrates"],   label=name, color=color, linewidth=1.5)
        axes[1].plot(chunks, res["buffers"],    label=name, color=color, linewidth=1.5)
        axes[2].plot(chunks, res["throughputs"], label=name, color=color, linewidth=1.5, alpha=0.8)

    axes[0].set_ylabel("Bitrate (Mbps)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Buffer (s)", fontsize=10)
    axes[2].set_ylabel("Throughput (Mbps)", fontsize=10)
    axes[2].set_xlabel("Chunk index", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def write_metrics_csv(all_results: dict, out_path: str):
    rows = []
    for name, results in all_results.items():
        avg_qoe     = np.mean([r["avg_qoe"]       for r in results])
        med_qoe     = np.median([r["avg_qoe"]      for r in results])
        avg_br      = np.mean([r["avg_bitrate"]    for r in results])
        avg_reb     = np.mean([r["total_rebuffer"] for r in results])
        avg_smooth  = np.mean([r["smoothness"]     for r in results])
        rows.append({
            "agent":           name,
            "mean_avg_qoe":    f"{avg_qoe:.4f}",
            "median_avg_qoe":  f"{med_qoe:.4f}",
            "mean_bitrate_mbps": f"{avg_br:.4f}",
            "mean_rebuffer_s": f"{avg_reb:.4f}",
            "mean_smoothness": f"{avg_smooth:.4f}",
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")

    # Print table to console
    print("\n" + "=" * 72)
    print(f"{'Agent':<22} {'Mean QoE':>10} {'Median QoE':>12} "
          f"{'Avg BR (Mbps)':>14} {'Rebuf (s)':>10} {'Smooth':>8}")
    print("-" * 72)
    for r in rows:
        print(f"{r['agent']:<22} {r['mean_avg_qoe']:>10} {r['median_avg_qoe']:>12} "
              f"{r['mean_bitrate_mbps']:>14} {r['mean_rebuffer_s']:>10} {r['mean_smoothness']:>8}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_agents(model_path: str, device: torch.device) -> dict:
    agents = {}
    for i, br in enumerate(BITRATE_LADDER_KBPS):
        agents[f"fixed_{br}kbps"] = FixedBitrate(i)
    agents["throughput_based"] = ThroughputBased()
    agents["buffer_based"]     = BufferBased()

    if os.path.exists(model_path):
        model = ActorCritic().to(device)
        load_checkpoint(model_path, model)
        model.eval()
        agents["rl_policy"] = (model, device)
        print(f"Loaded RL policy from {model_path}")
    else:
        print(f"WARNING: model not found at {model_path} — skipping RL policy.")

    return agents


def evaluate(args):
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    splits     = load_or_generate(trace_dir=args.traces, n_synthetic=args.n_synthetic)
    test_traces = splits["test"]
    print(f"Evaluating on {len(test_traces)} test traces…")

    env    = VideoStreamingEnv()
    agents = build_agents(args.model, device)

    print("Running rollouts…")
    all_results = evaluate_all(agents, env, test_traces, device)

    # --- Metrics table -------------------------------------------------------
    write_metrics_csv(
        all_results,
        os.path.join(RESULTS_DIR, "metrics_table.csv"),
    )

    # --- QoE CDF -------------------------------------------------------------
    plot_qoe_cdf(all_results, os.path.join(RESULTS_DIR, "qoe_cdf.png"))

    # --- Component breakdown --------------------------------------------------
    plot_qoe_components(all_results, os.path.join(RESULTS_DIR, "qoe_components.png"))

    # --- Timeline rollouts (3 representative traces) -------------------------
    n_timelines = min(3, len(test_traces))
    # Pick one low-bandwidth, one mid, one high trace by average throughput
    avg_tputs = [np.mean(t) for t in test_traces]
    ranked    = np.argsort(avg_tputs)
    picks     = [ranked[len(ranked) // 6], ranked[len(ranked) // 2], ranked[-len(ranked) // 6]]

    for plot_i, trace_idx in enumerate(picks):
        trace = test_traces[trace_idx]
        trace_results = {}
        for name, agent in agents.items():
            if name == "rl_policy":
                model, model_device = agent
                trace_results[name] = rollout_rl(env, model, trace, model_device)
            else:
                trace_results[name] = rollout_baseline(env, agent, trace)

        avg_t = float(np.mean(trace))
        plot_timeline(
            {k: v for k, v in trace_results.items()
             if k in ("throughput_based", "buffer_based", "rl_policy")},
            trace,
            out_path=os.path.join(RESULTS_DIR, f"rollout_trace_{plot_i+1}.png"),
            title=f"Trace {plot_i+1} — avg throughput {avg_t:.2f} Mbps",
        )

    print("Evaluation complete.  Results in", RESULTS_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       type=str, default="checkpoints/best_model.pt")
    p.add_argument("--traces",      type=str, default=None)
    p.add_argument("--n-synthetic", type=int, default=1000)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
