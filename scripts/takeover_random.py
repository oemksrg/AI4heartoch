"""用 BattlegroundsEnv 接管 1 席：在合法动作里随机选（接手训练的最小可运行基线）。

用法（先激活 .venv）:
  python scripts/takeover_random.py --episodes 3 --seed 0
"""

from __future__ import annotations

import argparse
import random


def choose_action(mask, rng: random.Random) -> int:
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        raise RuntimeError("no legal actions")
    return rng.choice(legal)


def run_episode(env, rng: random.Random) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    illegal = 0
    while True:
        action = choose_action(obs["action_mask"], rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if info.get("illegal"):
            illegal += 1
        if terminated or truncated:
            break
    return {
        "steps": steps,
        "reward": total_reward,
        "placement": info.get("placement"),
        "illegal": illegal,
        "turn": info.get("turn"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Random legal-action takeover baseline")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import taverngym

    env = taverngym.BattlegroundsEnv(seed=args.seed)
    rng = random.Random(args.seed)
    print(
        f"takeover seat vs 7 bots | episodes={args.episodes} "
        f"n_actions={env.n_actions} obs={env.observation_sizes}"
    )

    results = []
    for ep in range(args.episodes):
        result = run_episode(env, rng)
        results.append(result)
        print(
            f"ep={ep} placement={result['placement']} "
            f"reward={result['reward']:.3f} steps={result['steps']} "
            f"illegal={result['illegal']} turn={result['turn']}"
        )

    placements = [r["placement"] for r in results if r["placement"] is not None]
    if placements:
        avg = sum(placements) / len(placements)
        print(f"avg_placement={avg:.2f} (1=first, 8=last)")
    print("OK: takeover baseline finished")


if __name__ == "__main__":
    main()
