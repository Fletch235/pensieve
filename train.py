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
    NUM_TRAIN_ITERS, CHECKPOINT_EVERY, CHECKPOINT_DIR,
)
from env import VideoStreamingEnv
from model.network import ActorCritic, normalise_state
from model.a2c import A2CTrainer, RolloutBuffer, Transition
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
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    splits = load_or_generate(trace_dir=args.traces, n_synthetic=args.n_synthetic)
    train_traces = splits["train"]
    val_traces   = splits["val"]
    print(f"Traces — train: {len(train_traces)}  val: {len(val_traces)}  "
          f"test: {len(splits['test'])}")

    # Model + trainer
    model   = ActorCritic().to(device)
    trainer = A2CTrainer(model)
    env     = VideoStreamingEnv()

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
            print(f"  → val QoE = {val_qoe:.4f}", end="")
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces",      type=str,  default=None,
                   help="Path to directory of real trace files.")
    p.add_argument("--iters",       type=int,  default=NUM_TRAIN_ITERS,
                   help="Number of training iterations.")
    p.add_argument("--n-synthetic", type=int,  default=1000,
                   help="Number of synthetic traces to generate if no --traces given.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
