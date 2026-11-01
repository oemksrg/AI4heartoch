# AutoTavern 中文上手指南与落地计划

本文面向在本机 `E:\AI4heartoch` 基于 [MCG-NJU/AutoTavern](https://github.com/MCG-NJU/AutoTavern) 做**酒馆战棋接管训练 AI** 的开发者。官方英文说明见根目录 [README.md](README.md) 与 [docs/](docs/)。

若需要给完全不了解背景的人看「我们在做什么、每一步干什么、做到哪了」，请先读 [从零了解本项目.md](从零了解本项目.md)。

## 1. 一句话介绍

AutoTavern 把《炉石传说：酒馆战棋》的决策做成可复现的研究环境：用 **TavernGym** 跑完整对局并接学习算法，用 **TavernBench** 在固定题库上评测模型判断力。对应论文 *Build, Don't Play*（南京大学 / 上海 AI Lab）。

## 2. 能做 / 不能做

| 已开源（可用） | 未开源（需自研） |
| --- | --- |
| TavernGym：Rust 仿真 + Python 绑定 | 训练好的 TavernAgent 与权重 |
| 单局 `Game`、单席 `BattlegroundsEnv`、批量 `BatchEnv` | 官方训练代码与流程 |
| 978 个游戏对象（客户端补丁 36.6.3） | TavernBench 题目生成器 |
| TavernBench：960 题、标签、部分存档、已发布答案与打分 | — |

结论：本仓库是「**环境 + 评测**」底座；真正的「接管训练」要在本仓库之上自己写策略 / RL / LLM Agent。

## 3. 目录结构（你会常碰的部分）

| 路径 | 说明 |
| --- | --- |
| `README.md` | 官方总览与 Quick Start |
| `docs/` | TavernGym / TavernBench / 架构 / 内容覆盖等文档 |
| `crates/` | Rust 引擎与相关 crate |
| `python/` | Python 包源码（`taverngym`） |
| `tavernbench/` | 评测数据、打分与结果 |
| `tests/` | 测试 |
| `pyproject.toml` | Python 包装配置（maturin 构建） |
| `rust-toolchain.toml` | 所需 Rust 工具链版本 |

建议阅读顺序：`README.md` → `docs/TAVERNGYM.md` → `docs/ARCHITECTURE.md` → `docs/TAVERNBENCH.md`。

## 4. 本机依赖

跑仿真（TavernGym）需要：

1. **Python 3.10+**
2. **Rust 工具链**：[rustup](https://rustup.rs)（进入仓库后会按 `rust-toolchain.toml` 自动安装对应版本）
3. 能编译原生扩展的环境（Windows 上通常还需 MSVC Build Tools）

只给已发布答案打分（TavernBench report）时：只需 Python 标准库，不必编译仿真器。

## 5. 安装与最小验证

### 5.1 安装仿真器

在仓库根目录：

```bash
pip install .
```

开发/测试可选：

```bash
pip install -e ".[test]"
pip install ".[bench]"   # 若要用官方模型 runner
pytest
```

### 5.2 跑一局（脚本 bot）

```python
import taverngym

game = taverngym.Game(seed=1)
while not game.is_over():
    for seat in range(game.num_players):
        if game.needs_action(seat):
            game.step(seat, game.bot_action(seat))

print(list(game.placements()))
```

把某一席的 `bot_action` 换成你自己的合法动作，即开始「接管」该席。

### 5.3 单席学习环境

```python
import taverngym

env = taverngym.BattlegroundsEnv(seed=0)
obs, info = env.reset()
action = int(obs["action_mask"].nonzero()[0][0])  # 示例：选第一个合法动作
obs, reward, terminated, truncated, info = env.step(action)
```

观察大致包含：约 **245** 离散动作、**3227** 数值特征、**124** 卡牌索引；合法动作由 `action_mask` 给出。批量训练用 `BatchEnv`（见 `docs/TAVERNGYM.md`）。

### 5.4 只打分（无需仿真器）

```bash
cd tavernbench
python -m tavernbench.report --set data/eval --runs results/eval \
    --systems tavernagent,gpt-6-astra,claude-opus-5.5,jev
```

## 6. 分阶段计划（本仓库落地）

### 阶段 A：拿到代码与文档（本次）

- [x] 克隆 [MCG-NJU/AutoTavern](https://github.com/MCG-NJU/AutoTavern) 到本地工作区
- [x] 保留官方 README / docs
- [x] 编写本中文上手文档

### 阶段 B：环境跑通

- [x] 安装 Python 3.10+（本机：`E:\AI4heartoch\.venv`，基于 Anaconda 3.10）与 Rust 1.98.1（rustup）
- [x] 安装 Visual Studio Build Tools（MSVC，编译 Python 扩展需要）
- [x] `pip install -e .` 编译成功，`import taverngym` 可用
- [x] 跑通验证脚本：`python scripts/verify_bot_game.py`
- [x] 接管基线：`scripts/takeover_random.py` / `scripts/takeover_heuristic.py`
- [ ] （可选）跑通 TavernBench 打分命令

### 阶段 C：上传到你自己的 GitHub

- [x] 仓库：https://github.com/oemksrg/AI4heartoch
- [x] `origin` → 你的仓库；`upstream` → `MCG-NJU/AutoTavern`
- [x] 已提交本中文文档并 push

### 阶段 D：接管训练（后续主线）

1. **脚本策略**（已提供基线）：
   - 激活环境：`E:\AI4heartoch\.venv\Scripts\Activate.ps1`
   - 验证整局 bot：`python scripts/verify_bot_game.py`
   - 随机合法动作：`python scripts/takeover_random.py --episodes 3`
   - 简单启发式：`python scripts/takeover_heuristic.py --episodes 5`
2. **强化学习**（已提供最小训练器；官方训练代码未开源）：
   - 推荐（不依赖 PyTorch）：`python scripts/train_numpy_a2c.py --steps 20000 --n-games 16`
   - PyTorch 版（若本机 `import torch` 正常）：`python scripts/train_a2c.py --steps 30000`
   - 对比评估：`python scripts/eval_compare.py --episodes 12`
   - **外部权重 `last.pt`**（Transformer 策略，约 65MB）：  
     `D:\app\anaconda\python.exe scripts/eval_last_pt.py --checkpoint last.pt --episodes 5`  
     （当前项目 `.venv` 的 torch DLL 可能无法加载，请用 Anaconda 的 Python；架构为逆向加载，强度可能低于原训练代码）
   - 权重默认在 `runs/` / `last.pt`（`*.pt` 已 gitignore）
3. **LLM / Agent**：对状态做文本化或工具调用，在 TavernBench 上对比已发布答案。
4. **评测闭环**：仿真局内名次 + Bench 分数一起看，避免只刷一种指标。

说明：短训后名次仍可能接近第 8——观测维度高、奖励稀疏，需要加长训练或换更强网络；先保证训练循环能稳定跑完。

## 7. Git 远程建议

典型 remote 布局：

```text
upstream  https://github.com/MCG-NJU/AutoTavern.git   # 官方上游（只拉）
origin    https://github.com/<你的用户名>/<你的仓库>.git  # 你的 fork / 镜像（推送）
```

同步上游：

```bash
git fetch upstream
git merge upstream/main
```

## 8. 许可与声明

- 代码与 TavernBench 数据等以 **MIT** 发布（见 `LICENSE`）；第三方游戏内容权利见 `NOTICE`。
- 独立非商业研究项目，与暴雪无关；不包含官方客户端文件，也不连接暴雪服务。

## 9. 引用

若在研究中使用 TavernGym / TavernBench，请按官方 `README.md` / `CITATION.cff` 引用。

---

有问题优先查官方文档；本文件只做中文落地索引与计划，不替代 `docs/` 中的技术细节。
