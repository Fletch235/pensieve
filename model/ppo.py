# model/ppo.py
#
# PPO-Clip trainer for ABR bitrate selection.
# Drop-in replacement for A2CTrainer: identical __init__ and update() signatures.
#
# Differences vs A2C (model/a2c.py):
#   - Clipped surrogate objective  (Schulman et al. 2017, eq. 7)
#   - PPO_EPOCHS gradient passes per collected rollout
#   - Old log-probs computed once from the stored logits in the rollout buffer

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import (
    GAMMA, ACTOR_LR, CRITIC_LR,
    ENTROPY_BETA_START, ENTROPY_BETA_END, ENTROPY_BETA_DECAY,
    GRAD_CLIP_NORM, PPO_CLIP_EPS, PPO_EPOCHS,
)
from model.a2c import RolloutBuffer, compute_returns


class PPOTrainer:
    """PPO-Clip trainer.

    API is identical to A2CTrainer so train.py needs no structural changes.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.global_step = 0

        backbone_params = [
            p for n, p in model.named_parameters()
            if "critic_head" not in n
        ]
        self.optimizer = torch.optim.Adam([
            {"params": backbone_params,                    "lr": ACTOR_LR},
            {"params": model.critic_head.parameters(),     "lr": CRITIC_LR},
        ])

    # ------------------------------------------------------------------
    # Entropy coefficient schedule  (same decay as A2C)
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
        """PPO-Clip gradient step over PPO_EPOCHS passes of the rollout.

        Parameters
        ----------
        rollout            : RolloutBuffer collected under the old policy
        last_value         : V(s_{T+1}); 0.0 for terminal episodes
        normalise_state_fn : state dict → tensor dict   (from model.network)
        device             : torch device

        Returns
        -------
        dict with averaged loss components for logging (same keys as A2CTrainer)
        """
        rewards = [t.reward for t in rollout.transitions]
        returns = compute_returns(rewards, last_value)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)  # (T,)

        # Old log-probs from stored logits — computed once, no grad
        with torch.no_grad():
            old_log_probs = torch.stack([
                Categorical(logits=t.logits.to(device)).log_prob(
                    torch.tensor([t.action], device=device)
                )
                for t in rollout.transitions
            ]).squeeze(-1)  # (T,)

        total_losses, actor_losses, critic_losses, entropy_vals = [], [], [], []

        for _ in range(PPO_EPOCHS):
            log_probs_new, values_new, entropies = [], [], []

            for transition in rollout.transitions:
                normed = normalise_state_fn(transition.state, device)
                logits, value = self.model(normed)
                dist = Categorical(logits=logits)
                action_t = torch.tensor([transition.action], device=device)
                log_probs_new.append(dist.log_prob(action_t))
                values_new.append(value.squeeze())
                entropies.append(dist.entropy())

            log_probs_new_t = torch.stack(log_probs_new).squeeze(-1)  # (T,)
            values_new_t    = torch.stack(values_new)                  # (T,)
            entropy         = torch.stack(entropies).mean()

            advantages = returns_t - values_new_t.detach()             # (T,)

            ratio  = torch.exp(log_probs_new_t - old_log_probs)
            surr1  = ratio * advantages
            surr2  = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * advantages

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = ((values_new_t - returns_t) ** 2).mean()
            beta        = self.entropy_beta
            total_loss  = actor_loss + 0.5 * critic_loss - beta * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self.optimizer.step()

            total_losses.append(total_loss.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_vals.append(entropy.item())

        self.global_step += len(rollout)

        return {
            "total_loss":   sum(total_losses) / PPO_EPOCHS,
            "actor_loss":   sum(actor_losses) / PPO_EPOCHS,
            "critic_loss":  sum(critic_losses) / PPO_EPOCHS,
            "entropy":      sum(entropy_vals)  / PPO_EPOCHS,
            "entropy_beta": self.entropy_beta,
        }
