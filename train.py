# train.py — training entry point
#
# Usage:
#   cd pensieve/
#   python train.py                         # synthetic traces, 50k iters
#   python train.py --traces path/to/dir    # real FCC/HSDPA traces
#   python train.py --iters 10000           # quick test run
#
# Checkpoints are saved to checkpoints/ every CHECKPOINT_EVERY iterations.
# The checkpoint with the best validation QoE is saved as best_model.pt.

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical

from config import (
    NUM_TRAIN_ITERS, CHECKPOINT_EVERY, CHECKPOINT_DIR, A3C_NUM_WORKERS,
)
from env import VideoStreamingEnv
from model.network import ActorCritic, normalise_state
from model.a2c import A2CTrainer, RolloutBuffer, Transition
from model.ppo import PPOTrainer
from model.a3c import A3CTrainer
from baselines.robust_mpc import RobustMPC
from traces.loader import load_or_generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rollout_episode(
    env: VideoStreamingEnv,
    model: ActorCritic,
    trace: list[float],
    device: torch.device,
    train: bool = True,
) -> tuple[RolloutBuffer, float, list[dict]]:
    """Run one full episode.

    Returns (buffer, total_qoe, per-chunk info dicts).
    """
    state  = env.reset(trace=trace)
    buf    = RolloutBuffer()
    done   = False
    total_qoe = 0.0
    infos: list[dict] = []

    while not done:
        normed = normalise_state(state, device)
        with torch.set_grad_enabled(False):
            logits, value = model(normed)

        dist   = Categorical(logits=logits)
        action = dist.sample().item() if train else logits.argmax(dim=-1).item()

        next_state, reward, done, info = env.step(action)
        total_qoe += reward

        buf.append(Transition(
            state=state,
            action=action,
            reward=reward,
            logits=logits.detach(),
            value=value.detach(),
        ))
        infos.append(info)
        state = next_state

    return buf, total_qoe, infos


def evaluate_split(
    env: VideoStreamingEnv,
    model: ActorCritic,
    traces: list[list[float]],
    device: torch.device,
) -> float:
    """Return mean QoE across all traces (greedy policy, no sampling)."""
    model.eval()
    qoes: list[float] = []
    for trace in traces:
        _, qoe, _ = rollout_episode(env, model, trace, device, train=False)
        qoes.append(qoe)
    model.train()
    return float(np.mean(qoes))


def evaluate_baseline(
    env: VideoStreamingEnv,
    baseline: RobustMPC,
    traces: list[list[float]],
) -> float:
    """Return mean QoE for a rule-based baseline across all traces."""
    qoes: list[float] = []
    for trace in traces:
        baseline.reset()
        state = env.reset(trace=trace)
        total = 0.0
        done  = False
        chunk_idx = 0
        while not done:
            action = baseline.select_action(state, chunk_idx=chunk_idx)
            next_state, reward, done, info = env.step(action)
            baseline.update_error(info["throughput"])
            total += reward
            state  = next_state
            chunk_idx += 1
        qoes.append(total)
    return float(np.mean(qoes))


def save_checkpoint(model: ActorCritic, iteration: int, tag: str = ""):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = f"ckpt_{iteration:06d}{('_' + tag) if tag else ''}.pt"
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({"iteration": iteration, "model_state": model.state_dict()}, path)
    return path


def load_checkpoint(path: str, model: ActorCritic):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("iteration", 0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    if args.algo == "a3c":
        train_a3c(args)
        return

    seed = int(args.random_seed)
    set_seeds(seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    trace_dir = None
    if args.traces_dir is not None:
        trace_dir = os.path.join(args.traces_dir, "cooked_traces")
    print(f"Trace source: {trace_dir if trace_dir is not None else f'synthetic (n={args.n_synthetic})'}")
    splits = load_or_generate(trace_dir=trace_dir, n_synthetic=args.n_synthetic, seed=seed)
    train_traces = splits["train"]
    val_traces   = splits["val"]
    print(f"Traces — train: {len(train_traces)}  val: {len(val_traces)}  "
          f"test: {len(splits['test'])}")

    # Model + trainer
    model   = ActorCritic().to(device)
    trainer = PPOTrainer(model) if args.algo == "ppo" else A2CTrainer(model)
    print(f"Algorithm: {args.algo.upper()}")
    env     = VideoStreamingEnv()
    robust_mpc     = RobustMPC()
    robust_mpc_qoe = evaluate_baseline(env, robust_mpc, val_traces)
    print(f"robustMPC baseline val QoE: {robust_mpc_qoe:.4f}")

    best_val_qoe = -float("inf")
    best_path    = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    log_every    = max(args.iters // 100, 50)
    recent_qoes: list[float] = []
    recent_losses: list[float] = []
    t0 = time.time()

    for iteration in range(1, args.iters + 1):
        trace  = random.choice(train_traces)
        rollout, episode_qoe, _ = rollout_episode(env, model, trace, device, train=True)

        loss_info = trainer.update(
            rollout,
            last_value=0.0,     # terminal episode
            normalise_state_fn=normalise_state,
            device=device,
        )

        recent_qoes.append(episode_qoe)
        recent_losses.append(loss_info["total_loss"])

        if iteration % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[{iteration:6d}/{args.iters}] "
                f"QoE={np.mean(recent_qoes):.3f}  "
                f"loss={np.mean(recent_losses):.4f}  "
                f"H={loss_info['entropy']:.3f}  "
                f"beta={loss_info['entropy_beta']:.3f}  "
                f"step={trainer.global_step}  "
                f"t={elapsed:.0f}s"
            )
            recent_qoes.clear()
            recent_losses.clear()

        if iteration % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, iteration)
            val_qoe = evaluate_split(env, model, val_traces, device)
            pct = (val_qoe - robust_mpc_qoe) / abs(robust_mpc_qoe) * 100 if robust_mpc_qoe != 0 else float("nan")
            print(f"  → val QoE = {val_qoe:.4f}  ({pct:+.1f}% vs robustMPC)", end="")
            if val_qoe > best_val_qoe:
                best_val_qoe = val_qoe
                torch.save(
                    {"iteration": iteration, "model_state": model.state_dict()},
                    best_path,
                )
                print("  ★ new best", end="")
            print()

    print(f"\nTraining done.  Best val QoE: {best_val_qoe:.4f}  ({best_path})")


# ---------------------------------------------------------------------------
# A3C training loop (separate because the outer loop structure differs)
# ---------------------------------------------------------------------------

def train_a3c(args):
    import queue as queue_mod

    seed        = int(args.random_seed)
    num_workers = args.a3c_workers
    set_seeds(seed)
    device = torch.device("cpu")
    print(f"Device: {device}  |  Algorithm: A3C  |  Workers: {num_workers}")

    trace_dir = None
    if args.traces_dir is not None:
        trace_dir = os.path.join(args.traces_dir, "cooked_traces")
    print(f"Trace source: {trace_dir if trace_dir is not None else f'synthetic (n={args.n_synthetic})'}")
    splits = load_or_generate(trace_dir=trace_dir, n_synthetic=args.n_synthetic, seed=seed)
    train_traces = splits["train"]
    val_traces   = splits["val"]
    print(f"Traces — train: {len(train_traces)}  val: {len(val_traces)}  "
          f"test: {len(splits['test'])}")

    trainer = A3CTrainer(train_traces, normalise_state, seed, num_workers=num_workers)
    trainer.start()
    print(f"A3C: {num_workers} worker processes launched")

    env          = VideoStreamingEnv()
    robust_mpc     = RobustMPC()
    robust_mpc_qoe = evaluate_baseline(env, robust_mpc, val_traces)
    print(f"robustMPC baseline val QoE: {robust_mpc_qoe:.4f}")

    best_val_qoe = -float("inf")
    best_path    = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    log_every    = max(args.iters // 100, 50)
    recent_qoes:   list[float] = []
    recent_losses: list[float] = []
    t0      = time.time()
    updates = 0
    last_info: dict = {}

    try:
        while updates < args.iters:
            info = trainer.step()
            if info is None:
                continue   # timeout — workers may be slow to start

            updates += 1
            last_info = info
            recent_qoes.append(info["qoe"])
            recent_losses.append(info["total_loss"])

            if updates % log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"[{updates:6d}/{args.iters}] "
                    f"QoE={np.mean(recent_qoes):.3f}  "
                    f"loss={np.mean(recent_losses):.4f}  "
                    f"H={last_info['entropy']:.3f}  "
                    f"beta={last_info['entropy_beta']:.3f}  "
                    f"step={trainer.global_step}  "
                    f"t={elapsed:.0f}s"
                )
                recent_qoes.clear()
                recent_losses.clear()

            if updates % CHECKPOINT_EVERY == 0:
                save_checkpoint(trainer.model, updates)
                val_qoe = evaluate_split(env, trainer.model, val_traces, device)
                pct = (val_qoe - robust_mpc_qoe) / abs(robust_mpc_qoe) * 100 if robust_mpc_qoe != 0 else float("nan")
                print(f"  → val QoE = {val_qoe:.4f}  ({pct:+.1f}% vs robustMPC)", end="")
                if val_qoe > best_val_qoe:
                    best_val_qoe = val_qoe
                    torch.save(
                        {"iteration": updates, "model_state": trainer.model.state_dict()},
                        best_path,
                    )
                    print("  \u2605 new best", end="")
                print()
    finally:
        trainer.stop()

    print(f"\nA3C training done.  Best val QoE: {best_val_qoe:.4f}  ({best_path})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir",   type=str,   default=None,
                   help="Path to folder containing a cooked_traces/ subdirectory.")
    p.add_argument("--iters",        type=int,   default=NUM_TRAIN_ITERS,
                   help="Number of training iterations.")
    p.add_argument("--n-synthetic",  type=int,   default=1000,
                   help="Number of synthetic traces to generate if no --traces-dir given.")
    p.add_argument("--algo",         type=str,   default="ppo", choices=["ppo", "a2c", "a3c"],
                   help="Training algorithm (default: ppo).")
    p.add_argument("--a3c-workers",  type=int,   default=A3C_NUM_WORKERS,
                   help=f"Worker processes for A3C (default: {A3C_NUM_WORKERS}).")
    p.add_argument("--random-seed",  type=float, default=None,
                   help="Random seed (default: time.time()).")
    args = p.parse_args()
    if args.random_seed is None:
        args.random_seed = time.time()
    return args


if __name__ == "__main__":
    # Guard required for torch.multiprocessing 'spawn' used by A3C.
    # Always present in the original script — do not remove.
    train(parse_args())
