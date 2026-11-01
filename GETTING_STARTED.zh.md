# AutoTavern 中文上手指南与落地计划

本文面向在本机 `E:\AI4heartoch` 基于 [MCG-NJU/AutoTavern](https://github.com/MCG-NJU/AutoTavern) 做**酒馆战棋接管训练 AI** 的开发者。官方英文说明见根目录 [README.md](README.md) 与 [docs/](docs/)。

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

- [ ] 安装 Python 3.10+ 与 Rust（rustup）
- [ ] `pip install .` 编译成功，`import taverngym` 可用
- [ ] 跑通 5.2 一局 bot 对局
- [ ] （可选）跑通 TavernBench 打分命令

### 阶段 C：上传到你自己的 GitHub

- [ ] 在你的账号下 Fork 或新建仓库（不要直接 push 到 `MCG-NJU/AutoTavern`，除非你是该组织协作者）
- [ ] 本地 `origin` 指向你的仓库；可选保留 `upstream` → `MCG-NJU/AutoTavern`
- [ ] 提交本中文文档并 `git push`

### 阶段 D：接管训练（后续主线）

1. **脚本策略**：在 `BattlegroundsEnv` 上用规则/启发式选合法动作，确认观测与奖励含义。
2. **强化学习**：用 `BatchEnv` 接 PPO 等（需自研训练循环；官方训练代码未开源）。
3. **LLM / Agent**：对状态做文本化或工具调用，在 TavernBench 上对比已发布答案。
4. **评测闭环**：仿真局内名次 + Bench 分数一起看，避免只刷一种指标。

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
