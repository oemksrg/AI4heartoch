# AutoTavern 模型交接文档

## 交付内容

- `mine/runs/ppo/best.pt`：评估最佳 PPO 模型，默认使用。
- `mine/runs/ppo/latest.pt`：导出时的最新 PPO 模型。
- `MODEL_INFO.json`：两个模型的训练步数、参数量与 SHA256。
- `mine/model.py`、`mine/config.py`：模型结构与配置定义。
- `mine/arena/`：本地推理连接器、固定服务器契约及公开兼容补丁。
- `crates/`、`python/`、构建配置：公开 TavernGym 1.0.0 源码及原项目许可证。
- `mine/arena/upstream-base.tar`：公开基础源码快照，用于无 Git 仓库的兼容性核验。

两个模型各有 1,528,054 个参数，卡牌嵌入维度 16，共享网络为两层 256 单元，
策略头与价值头共享编码器。它们是纯推理检查点，不包含优化器、游戏存档、随机状态、
账号、参赛密钥、比赛记录、训练日志或本机绝对路径。不能用这些文件精确断点续训。
交付包不包含原项目 Git 历史或编译二进制；第三方开源作者信息与许可证保留。

## 安装

需要 Linux/macOS、Python 3.12+、Git（应用公开补丁）、Rust 1.98.1 工具链及网络。
不需要原机器的目录或账号。Windows 建议通过 WSL 使用脚本。

```bash
conda create -n autotavern python=3.12 pip -y
conda activate autotavern
cd autotavern_model_handoff
# 如尚未安装 Rust，先按 https://rustup.rs 官方说明安装 rustup。
bash setup.sh
python -m mine.arena.run --check
```

`setup.sh` 会安装 NumPy、PyTorch 并从附带源码编译公开引擎，首次构建需要一些时间。
CPU 即可推理。不要直接安装其他版本的 TavernGym 替换附带版本，否则指纹检查将失败。
源码基础提交为 `fb61201a557ddd36a411cf44aac51ce4d9b606ce`，
本地引擎指纹 `c807e9aa1a6746ee`；固定服务器 1.1.0 指纹为 `6d8712c93022ba11`。

## 模型推理与输入

入口是 `mine.arena.policy.act(observation, context)`，连接前由
`mine.arena.policy.initialize(checkpoint)` 预加载权重。
`LocalPolicy(checkpoint)` 也可直接调用。使用合法动作掩码后的贪心动作，每次返回整数。

服务器必须提供真实 observe2：float32[4017] 特征、int32[447] 卡牌索引、bool[245]
合法掩码，以及 `context.player_view` 完整正常可见 JSON。
旧模型只读取经过公开补丁证明不变的 3227/124 前缀；不会补零伪造缺失输入。
模型没有针对 v2 新增扩展重新训练。每次连接检查布局、卡牌词表、版本与源码指纹，
每步校验真实可见血甲、经济等信息与数值观测一致。契约变更时停止，需要重新审阅适配。

## 先练习，再匹配

接收者必须注册/使用自己的账号与 AI 身份，提供自己的 Agent Key。
不要沿用交付方账号。将密钥单独写到 `mine/arena.key`，仅一行，不加引号，权限 600：

```bash
# 用编辑器填写密钥，不要放进命令行或 URL。
touch mine/arena.key
chmod 600 mine/arena.key
# 使用本地编辑器填写并保存，然后运行一局不计分练习：
python -m mine.arena.run --mode practice --key-file mine/arena.key
```

确认 practice 正常结束后，由使用者主动启动常规匹配：

```bash
# 至少另一个参赛账号后才开局，保持心跳，空位可由规则机器人补齐。
bash mine/play_arena.sh
# 可选：不强制等待另一个账号，仍遵守服务器匹配倒计时。
bash mine/play_arena.sh --no-wait-for-human
# 指定最新模型：
bash mine/play_arena.sh --checkpoint mine/runs/ppo/latest.pt
```

启动脚本自动寻找名为 autotavern 的环境，也可设置 AUTOTAVERN_PYTHON 为 Python 路径。
每次只打一局，自动心跳、终端进度与最终名次展示，结算后退出，不自动再次排队。
默认不保存本地日志/对局结果；只有显式传 `--out-dir` 才写结果文件。
保持终端运行，或自行在 tmux 中启动。已有队列/比赛时脚本拒绝重复启动。
模型、代码和存档仅在本机使用，不上传到服务器；服务器收到协议操作与合法动作。

## 校验与限制

解压后可对比 MODEL_INFO.json 内哈希。包内 SHA256SUMS 可校验交付文件：在解压目录运行 sha256sum -c SHA256SUMS。
两份权重导出前后逐张量完全一致。导出时仅保留允许字段，并扫描密钥、本机路径和账号，
不会修改原始训练文件。模型能力受已有训练限制，单局成绩不代表稳定竞技水平。
精确匹配服务器协议的公开文档：https://rizling.top/agent.md 和 https://rizling.top/protocol 。
