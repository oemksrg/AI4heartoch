"""纯 NumPy 的 masked softmax 策略（不依赖 PyTorch）。"""

from __future__ import annotations

import numpy as np


def softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = logits.copy()
    x[~mask] = -1e9
    x = x - np.max(x)
    e = np.exp(x)
    e[~mask] = 0.0
    s = e.sum()
    if s <= 0:
        legal = np.flatnonzero(mask)
        p = np.zeros_like(logits, dtype=np.float64)
        p[legal] = 1.0 / len(legal)
        return p
    return e / s


class LinearPolicy:
    """features -> actions 的线性策略 + 标量 value 头。"""

    def __init__(self, n_features: int, n_actions: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(n_features)
        self.W = rng.normal(0, scale, size=(n_features, n_actions)).astype(np.float64)
        self.b = np.zeros(n_actions, dtype=np.float64)
        self.Vw = rng.normal(0, scale, size=n_features).astype(np.float64)
        self.Vb = 0.0

    def logits_value(self, features: np.ndarray):
        logits = features @ self.W + self.b
        value = float(features @ self.Vw + self.Vb)
        return logits, value

    def act(self, features: np.ndarray, mask: np.ndarray, rng: np.random.Generator, deterministic=False):
        logits, value = self.logits_value(features)
        probs = softmax_masked(logits, mask.astype(bool))
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(len(probs), p=probs))
        logp = float(np.log(probs[action] + 1e-12))
        return action, logp, value, probs

    def update(self, batch, lr: float = 3e-4, vf_coef: float = 0.5, ent_coef: float = 0.01):
        """batch: list of dict(features, action, adv, ret, probs_at_step optional via mask)"""
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        gVw = np.zeros_like(self.Vw)
        gVb = 0.0

        for item in batch:
            f = item["features"]
            a = item["action"]
            adv = item["adv"]
            ret = item["ret"]
            mask = item["mask"].astype(bool)
            logits, value = self.logits_value(f)
            probs = softmax_masked(logits, mask)

            # policy grad: (one_hot - probs) * adv  for softmax
            dlogits = -probs
            dlogits[a] += 1.0
            dlogits *= adv
            gW += np.outer(f, dlogits)
            gb += dlogits

            # entropy bonus grad approx on logits: - (p * (log p + 1 - H?)) keep simple: encourage probs
            # dH/dlogit_i = -p_i * (log p_i - sum p log p) ; add ent_coef * that
            logp = np.log(np.clip(probs, 1e-12, 1.0))
            H = -np.sum(probs * logp)
            dent = -probs * (logp + H)
            dent[~mask] = 0.0
            gW += ent_coef * np.outer(f, dent)
            gb += ent_coef * dent

            # value
            dv = vf_coef * 2.0 * (value - ret)
            gVw += dv * f
            gVb += dv

        n = max(len(batch), 1)
        self.W += lr * gW / n
        self.b += lr * gb / n
        self.Vw += lr * gVw / n
        self.Vb += lr * gVb / n

    def save(self, path: str):
        np.savez_compressed(path, W=self.W, b=self.b, Vw=self.Vw, Vb=np.array([self.Vb]))

    @classmethod
    def load(cls, path: str) -> "LinearPolicy":
        data = np.load(path)
        obj = cls(data["W"].shape[0], data["W"].shape[1])
        obj.W = data["W"]
        obj.b = data["b"]
        obj.Vw = data["Vw"]
        obj.Vb = float(data["Vb"][0])
        return obj
