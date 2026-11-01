"""Masked actor-critic for TavernGym observations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_cards_slots: int,
        n_card_vocab: int,
        n_actions: int,
        hidden: int = 256,
        card_dim: int = 32,
    ):
        super().__init__()
        self.card_emb = nn.Embedding(n_card_vocab, card_dim, padding_idx=0)
        self.feat = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.merge = nn.Sequential(
            nn.Linear(hidden + card_dim, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def encode(self, features: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
        # features: [B, F], cards: [B, C]
        h = self.feat(features)
        emb = self.card_emb(cards.clamp_min(0))  # [B, C, D]
        # mean pool non-padding card slots
        mask = (cards > 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (emb * mask).sum(dim=1) / denom
        return self.merge(torch.cat([h, pooled], dim=-1))

    def forward(self, features, cards, action_mask):
        x = self.encode(features, cards)
        logits = self.policy(x)
        # mask illegal actions
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(action_mask.bool(), logits, torch.full_like(logits, neg_inf))
        value = self.value(x).squeeze(-1)
        return logits, value

    def act(self, features, cards, action_mask, deterministic: bool = False):
        logits, value = self.forward(features, cards, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_prob, entropy, value
