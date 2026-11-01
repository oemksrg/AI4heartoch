"""对比随机 / 规则基线 / NumPy A2C 的名次。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_policy(env, choose_fn, episodes: int, seed: int, max_steps: int = 1500):
    placements, rewards = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        while steps < max_steps:
            if steps > 0 and steps % 35 == 0 and obs["action_mask"][0]:
                action = 0
            else:
                action = choose_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        placements.append(info.get("placement"))
        rewards.append(total)
    pl = [p for p in placements if p is not None]
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_placement": float(np.mean(pl)) if pl else None,
        "placements": pl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numpy-checkpoint", type=str, default="runs/numpy_a2c_latest.npz")
    args = parser.parse_args()

    import taverngym
    from train.numpy_policy import LinearPolicy

    baseline_path = ROOT / "scripts" / "takeover_baseline.py"
    spec = importlib.util.spec_from_file_location("takeover_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(baseline)

    env = taverngym.BattlegroundsEnv(seed=args.seed)
    rng = random.Random(args.seed)

    def random_action(obs):
        legal = np.flatnonzero(obs["action_mask"])
        return int(rng.choice(legal))

    results = {
        "random": run_policy(env, random_action, args.episodes, args.seed),
        "heuristic": run_policy(env, baseline.heuristic_action, args.episodes, args.seed + 1000),
    }

    np_ckpt = Path(args.numpy_checkpoint)
    if np_ckpt.is_file():
        policy = LinearPolicy.load(str(np_ckpt))
        np_rng = np.random.default_rng(args.seed)

        def numpy_action(obs):
            a, _, _, _ = policy.act(obs["features"], obs["action_mask"], np_rng, deterministic=True)
            return a

        results["a2c_numpy"] = run_policy(env, numpy_action, args.episodes, args.seed + 2000)
    else:
        results["a2c_numpy"] = {"error": f"missing checkpoint: {np_ckpt}"}

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
