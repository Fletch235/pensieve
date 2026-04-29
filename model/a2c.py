# model/a2c.py
#
# Synchronous Advantage Actor-Critic (A2C) training logic.
# Simplified from the paper's A3C: single-worker, full-episode rollouts.
#
# Key design choices matching the paper:
#   - γ = 0.99 (100-step effective horizon)
#   - Entropy bonus β decays 1.0 → 0.1 over ENTROPY_BETA_DECAY steps
#   - Separate learning rates for actor and critic parameter groups
#   - Gradient clipping (max_norm = 0.5) for stability

import torch
import torch.nn as nn
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import NamedTuple

from config import (
    GAMMA, ACTOR_LR, CRITIC_LR,
    ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY,
    GRAD_CLIP_NORM,
)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    state:   dict
    action:  int
    reward:  float
    logits:  torch.Tensor    # (1, num_bitrates) — detached
    value:   torch.Tensor    # (1, 1)            — detached


@dataclass
class RolloutBuffer:
    transitions: list[Transition] = field(default_factory=list)

    def append(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


# ---------------------------------------------------------------------------
# A2C update
# ---------------------------------------------------------------------------

def compute_returns(rewards: list[float], last_value: float, gamma: float = GAMMA) -> list[float]:
    """Discounted returns R_t = r_t + γ r_{t+1} + γ² r_{t+2} + ..."""
    returns: list[float] = []
    R = last_value
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


class A2CTrainer:
    """Wraps the model and optimizer; exposes a single update() call."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.global_step = 0

        # Separate learning rates via parameter groups
        actor_params  = list(model.parameters())   # all share backbone; we just
        critic_params = list(model.critic_head.parameters())  # up-weight critic LR

        # Build a single optimizer with two groups
        backbone_params = [
            p for n, p in model.named_parameters()
            if "critic_head" not in n
        ]
        self.optimizer = torch.optim.Adam([
            {"params": backbone_params,          "lr": ACTOR_LR},
            {"params": model.critic_head.parameters(), "lr": CRITIC_LR},
        ])

    # ------------------------------------------------------------------
    # Entropy coefficient schedule
    # ------------------------------------------------------------------

    @property
    def entropy_beta(self) -> float:
        progress = min(self.global_step / ENTROPY_BETA_DECAY, 1.0)
        return ENTROPY_BETA_START + progress * (ENTROPY_BETA_END - ENTROPY_BETA_START)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        rollout: RolloutBuffer,
        last_value: float,
        normalise_state_fn,
        device: torch.device,
    ) -> dict:
        """Perform one A2C gradient step on a completed rollout.

        Parameters
        ----------
        rollout          : RolloutBuffer with T transitions
        last_value       : V(s_{T+1}) — 0.0 for terminal episodes
        normalise_state_fn : function mapping raw state dict → tensor dict
        device           : torch device

        Returns
        -------
        dict with scalar loss components for logging
        """
        rewards = [t.reward for t in rollout.transitions]
        returns = compute_returns(rewards, last_value)

        actor_losses, critic_losses, entropies = [], [], []

        for transition, R in zip(rollout.transitions, returns):
            # Re-compute logits and value with gradient (stored values are detached)
            normed = normalise_state_fn(transition.state, device)
            logits, value = self.model(normed)

            dist    = Categorical(logits=logits)
            action  = torch.tensor([transition.action], device=device)

            log_prob   = dist.log_prob(action)
            entropy    = dist.entropy()
            advantage  = R - value.squeeze().detach()

            actor_losses.append(-log_prob * advantage)
            critic_losses.append((value.squeeze() - R) ** 2)
            entropies.append(entropy)

        actor_loss  = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()
        entropy     = torch.stack(entropies).mean()

        beta = self.entropy_beta
        total_loss = actor_loss + 0.5 * critic_loss - beta * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        self.global_step += len(rollout)

        return {
            "total_loss":   total_loss.item(),
            "actor_loss":   actor_loss.item(),
            "critic_loss":  critic_loss.item(),
            "entropy":      entropy.item(),
            "entropy_beta": beta,
        }
