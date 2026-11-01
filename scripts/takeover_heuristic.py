"""用简单启发式接管 1 席：处理选择 → 出牌/购买/升级 → 适时结束回合。

这是规则策略基线，便于确认观测与动作接口后再接 RL。

用法:
  .\\.venv\\Scripts\\python.exe scripts\\takeover_heuristic.py --episodes 5 --seed 0
"""

from __future__ import annotations

import argparse
import random

import numpy as np

# 与 docs/TAVERNGYM.md 动作区间一致
END_TURN = 0
REFRESH = 1
FREEZE = 2
UPGRADE = 3
HERO_POWER = range(4, 6)
BUY = range(8, 16)
SELL = range(16, 23)
ACTIVATE = range(23, 30)
CHOOSE = range(30, 46)
PLAY = range(46, 126)
MAGNETIZE = range(126, 196)
MOVE = range(196, 245)


def first_legal(mask, lo: int, hi: int) -> int | None:
    for a in range(lo, hi):
        if mask[a]:
            return a
    return None


def choose_action(mask, rng: random.Random, since_end: int) -> int:
    """强制选择优先；购物/升级后适时 end turn，避免无限不出回合。"""
    # 有 Discover / 目标等待选时必须先处理
    a = first_legal(mask, CHOOSE.start, CHOOSE.stop)
    if a is not None:
        return a

    # 同一招募阶段做够操作后，优先结束回合
    if since_end >= 6 and mask[END_TURN] and rng.random() < 0.85:
        return END_TURN

    a = first_legal(mask, PLAY.start, PLAY.stop)
    if a is not None and rng.random() < 0.8:
        return a
    a = first_legal(mask, MAGNETIZE.start, MAGNETIZE.stop)
    if a is not None and rng.random() < 0.7:
        return a

    a = first_legal(mask, BUY.start, BUY.stop)
    if a is not None:
        return a
    if mask[UPGRADE]:
        return UPGRADE
    a = first_legal(mask, HERO_POWER.start, HERO_POWER.stop)
    if a is not None:
        return a
    a = first_legal(mask, ACTIVATE.start, ACTIVATE.stop)
    if a is not None and rng.random() < 0.5:
        return a

    if mask[REFRESH] and rng.random() < 0.35:
        return REFRESH
    if mask[FREEZE] and rng.random() < 0.25:
        return FREEZE

    if since_end >= 3 and mask[END_TURN]:
        return END_TURN

    a = first_legal(mask, MOVE.start, MOVE.stop)
    if a is not None and rng.random() < 0.2:
        return a
    a = first_legal(mask, SELL.start, SELL.stop)
    if a is not None and rng.random() < 0.3:
        return a

    if mask[END_TURN]:
        return END_TURN
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        raise RuntimeError("no legal actions")
    return rng.choice(legal)


def run_episode(env, rng: random.Random) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    illegal = 0
    since_end = 0
    while True:
        action = choose_action(obs["action_mask"], rng, since_end)
        since_end = 0 if action == END_TURN else since_end + 1
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
    print(f"taverngym {taverngym.version()} | heuristic takeover")
    print(f"episodes={args.episodes} seed={args.seed} obs={env.observation_sizes}")

    results = []
    for ep in range(args.episodes):
        result = run_episode(env, rng)
        results.append(result)
        print(
            f"ep={ep} placement={result['placement']} "
            f"reward={result['reward']:+.3f} steps={result['steps']} "
            f"illegal={result['illegal']} turn={result['turn']}"
        )

    placements = [r["placement"] for r in results if r["placement"] is not None]
    rewards = [r["reward"] for r in results]
    print("--- summary ---")
    print(f"mean reward: {float(np.mean(rewards)):+.3f}")
    if placements:
        print(f"avg_placement={sum(placements) / len(placements):.2f} (1=best, 8=worst)")
    print("OK: heuristic baseline finished")


if __name__ == "__main__":
    main()
