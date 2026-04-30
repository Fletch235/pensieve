# model/a3c.py
#
# A3C: Asynchronous Advantage Actor-Critic
#
# Design (gradient-queue, avoids shared optimizer state):
#   - N worker processes each keep a local model copy.
#   - Workers: sync local ← shared_model → collect episode → compute grads
#              → push {grads, metrics} to grad_queue.
#   - Main process: reads grad_queue → applies grads to private_model →
#              optimizer.step() → syncs shared_model ← private_model.
#   - Workers never touch the optimizer; the main process is the sole
#     optimizer consumer, so Adam state never needs to be in shared memory.

import random
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical

from config import (
    ACTOR_LR, CRITIC_LR,
    ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY,
    GRAD_CLIP_NORM, A3C_NUM_WORKERS,
)
from model.a2c import RolloutBuffer, Transition, compute_returns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy_beta(step: int) -> float:
    progress = min(step / ENTROPY_BETA_DECAY, 1.0)
    return ENTROPY_BETA_START + progress * (ENTROPY_BETA_END - ENTROPY_BETA_START)


# ---------------------------------------------------------------------------
# Worker process (top-level for pickle compatibility with 'spawn')
# ---------------------------------------------------------------------------

def _worker_fn(
    worker_id: int,
    shared_model: nn.Module,        # read-only: workers sync from this
    grad_queue: mp.Queue,            # write: push grads + metrics here
    train_traces: list,
    stop_event: mp.Event,
    global_step_val: mp.Value,       # read: current global step for β schedule
    normalise_state_fn,
    seed: int,
):
    """Collect episodes, compute policy gradients, push to grad_queue."""
    # Deferred imports: with 'spawn' the worker is a fresh interpreter
    from model.network import ActorCritic
    from env import VideoStreamingEnv

    random.seed(seed + worker_id * 137)
    torch.manual_seed(seed + worker_id * 137)

    local_model = ActorCritic()
    env = VideoStreamingEnv()
    device = torch.device("cpu")

    while not stop_event.is_set():
        # --- Sync local weights from shared model ----------------------------
        local_model.load_state_dict(shared_model.state_dict())
        local_model.train()

        with global_step_val.get_lock():
            current_step = global_step_val.value
        beta = _entropy_beta(current_step)

        # --- Collect one full episode ----------------------------------------
        trace = random.choice(train_traces)
        state = env.reset(trace=trace)
        buf = RolloutBuffer()
        done = False
        total_qoe = 0.0

        while not done:
            normed = normalise_state_fn(state, device)
            with torch.no_grad():
                logits, value = local_model(normed)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)
            total_qoe += reward
            buf.append(Transition(
                state=state, action=action, reward=reward,
                logits=logits.detach(), value=value.detach(),
            ))
            state = next_state

        # --- Compute policy-gradient loss on local model ---------------------
        rewards = [t.reward for t in buf.transitions]
        returns = compute_returns(rewards, last_value=0.0)

        actor_losses, critic_losses, entropies = [], [], []
        for transition, R in zip(buf.transitions, returns):
            normed = normalise_state_fn(transition.state, device)
            logits, value = local_model(normed)
            dist = Categorical(logits=logits)
            action_t = torch.tensor([transition.action])
            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()
            advantage = R - value.squeeze().detach()
            actor_losses.append(-log_prob * advantage)
            critic_losses.append((value.squeeze() - R) ** 2)
            entropies.append(entropy)

        actor_loss   = torch.stack(actor_losses).mean()
        critic_loss  = torch.stack(critic_losses).mean()
        entropy_mean = torch.stack(entropies).mean()
        total_loss   = actor_loss + 0.5 * critic_loss - beta * entropy_mean

        local_model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), GRAD_CLIP_NORM)

        # --- Package gradients as plain CPU tensors (detached copies) --------
        grads = {
            name: p.grad.cpu().clone()
            for name, p in local_model.named_parameters()
            if p.grad is not None
        }

        grad_queue.put({
            "grads":        grads,
            "n_steps":      len(buf),
            "qoe":          total_qoe,
            "total_loss":   total_loss.item(),
            "entropy":      entropy_mean.item(),
            "entropy_beta": beta,
        })


# ---------------------------------------------------------------------------
# A3CTrainer — used by train.py
# ---------------------------------------------------------------------------

class A3CTrainer:
    """A3C coordinator.

    Workers push gradient dicts to a queue; the main process applies them.
    The optimizer lives exclusively in the main process — no shared state needed.

    Usage::

        trainer = A3CTrainer(train_traces, normalise_state, seed)
        trainer.start()
        for _ in range(total_iters):
            info = trainer.step()   # blocks until one worker pushes grads
            if info is None:
                continue            # timeout — keep waiting
            # log info...
        trainer.stop()
        # trainer.model holds the fully trained weights
    """

    def __init__(
        self,
        train_traces: list,
        normalise_state_fn,
        seed: int,
        num_workers: int = A3C_NUM_WORKERS,
    ):
        from model.network import ActorCritic

        ctx = mp.get_context("spawn")

        # Private model + optimizer — only the main process touches these
        self.model = ActorCritic()
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if "critic_head" not in n
        ]
        self.optimizer = torch.optim.Adam([
            {"params": backbone_params,                    "lr": ACTOR_LR},
            {"params": self.model.critic_head.parameters(), "lr": CRITIC_LR},
        ])
        self.global_step = 0

        # Shared model — workers read from this to stay up to date
        self._shared_model = ActorCritic()
        self._shared_model.load_state_dict(self.model.state_dict())
        self._shared_model.share_memory()

        # IPC
        # maxsize throttles workers if main process falls behind
        self._grad_queue      = ctx.Queue(maxsize=num_workers * 4)
        self._stop_event      = ctx.Event()
        self._global_step_val = ctx.Value("i", 0)

        self._ctx          = ctx
        self._num_workers  = num_workers
        self._train_traces = train_traces
        self._normalise    = normalise_state_fn
        self._seed         = seed
        self._workers: list = []

    # ------------------------------------------------------------------

    def start(self):
        for wid in range(self._num_workers):
            p = self._ctx.Process(
                target=_worker_fn,
                args=(
                    wid,
                    self._shared_model,
                    self._grad_queue,
                    self._train_traces,
                    self._stop_event,
                    self._global_step_val,
                    self._normalise,
                    self._seed,
                ),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    def step(self, timeout: float = 5.0) -> dict | None:
        """Apply one worker gradient update.

        Returns a metrics dict, or None if no worker reported within *timeout* seconds.
        """
        import queue as _queue
        try:
            payload = self._grad_queue.get(timeout=timeout)
        except _queue.Empty:
            return None

        # Apply gradients to private model and step optimizer
        self.optimizer.zero_grad()
        for name, p in self.model.named_parameters():
            if name in payload["grads"]:
                p.grad = payload["grads"][name]
        self.optimizer.step()

        # Sync shared model so workers pick up the new weights
        self._shared_model.load_state_dict(self.model.state_dict())

        self.global_step += payload["n_steps"]
        with self._global_step_val.get_lock():
            self._global_step_val.value = self.global_step

        return {
            "total_loss":   payload["total_loss"],
            "entropy":      payload["entropy"],
            "entropy_beta": payload["entropy_beta"],
            "qoe":          payload["qoe"],
        }

    def stop(self):
        self._stop_event.set()
        for p in self._workers:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
