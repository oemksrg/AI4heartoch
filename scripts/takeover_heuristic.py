"""用简单启发式接管 1 席：优先升级酒馆 / 购买 / 刷新，否则随机合法动作。

这是规则策略基线，便于确认观测与动作接口后再接 RL。

用法:
  python scripts/takeover_heuristic.py --episodes 5 --seed 0
"""

from __future__ import annotations

import argparse
import random

# 与 docs/TAVERNGYM.md 动作区间一致的常用动作
END_TURN = 0
REFRESH = 1
FREEZE = 2
UPGRADE = 3


def prefer(mask, *actions: int) -> int | None:
    for a in actions:
        if 0 <= a < len(mask) and mask[a]:
            return a
    return None


def choose_action(mask, rng: random.Random) -> int:
    # 能升级先升级；再尝试常见买牌区间（8-31 为部分买/卖相关，具体以 mask 为准）
    chosen = prefer(mask, UPGRADE)
    if chosen is not None:
        return chosen

    buy_like = [i for i in range(8, 40) if i < len(mask) and mask[i]]
    if buy_like and rng.random() < 0.7:
        return rng.choice(buy_like)

    chosen = prefer(mask, REFRESH)
    if chosen is not None and rng.random() < 0.4:
        return chosen

    chosen = prefer(mask, END_TURN)
    if chosen is not None and rng.random() < 0.5:
        return chosen

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
    parser = argparse.ArgumentParser(description="Heuristic takeover baseline")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import taverngym

    env = taverngym.BattlegroundsEnv(seed=args.seed)
    rng = random.Random(args.seed)
    print(f"heuristic takeover | episodes={args.episodes} seed={args.seed}")

    results = []
    for ep in range(args.episodes):
        result = run_episode(env, rng)
        results.append(result)
        print(
            f"ep={ep} placement={result['placement']} "
            f"reward={result['reward']:.3f} steps={result['steps']} "
            f"illegal={result['illegal']}"
        )

    placements = [r["placement"] for r in results if r["placement"] is not None]
    if placements:
        print(f"avg_placement={sum(placements) / len(placements):.2f}")
    print("OK: heuristic baseline finished")


if __name__ == "__main__":
    main()
