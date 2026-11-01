"""Shared actor/critic with position-preserving card embeddings and legal action masking."""

import torch
from torch import nn
from torch.distributions import Categorical
import taverngym as tg


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        # N_CARDS includes the empty index 0 (the engine exports db.len() + 1).
        self.embedding = nn.Embedding(tg.N_CARDS, config.embedding_dim, padding_idx=0)
        width = tg.OBS_FLOATS + tg.OBS_CARDS * config.embedding_dim
        layers = []
        for hidden in config.hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.Tanh()])
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.actor = nn.Linear(width, tg.N_ACTIONS)
        self.critic = nn.Linear(width, 1)
        # Some CPU PyTorch builds lack LAPACK; orthogonal_ then fails. Checkpoints overwrite these.
        try:
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)
        except RuntimeError:
            nn.init.xavier_uniform_(self.actor.weight)
            nn.init.xavier_uniform_(self.critic.weight)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward(self, features, cards):
        # Engine features are scaled but late-game stats can grow large. A fixed symlog
        # transform is identical during rollout, PPO updates, evaluation and restore.
        features = features.sign() * features.abs().log1p()
        x = torch.cat((features, self.embedding(cards.long()).flatten(start_dim=1)), dim=1)
        x = self.encoder(x)
        return self.actor(x), self.critic(x).squeeze(-1)

    def policy(self, features, cards, mask):
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("A policy observation has no legal action")
        logits, values = self(features, cards)
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf)), values


def observation_tensors(obs, device):
    return (torch.as_tensor(obs["features"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["cards"], dtype=torch.long, device=device),
            torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device))
