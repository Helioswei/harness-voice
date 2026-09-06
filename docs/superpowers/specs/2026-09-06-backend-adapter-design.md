# Hermes Voice — Backend 输出端适配层设计

> 状态：✅ 已与用户讨论定稿（2026-09-06）
> 目标：把"语音转文字后的下游"从写死 Hermes API 泛化为可插拔 Backend 适配层。

---

## 1. 背景与问题

现在 daemon 的文字下游是**唯一且写死**的：`voice/hermes_client.py` 的 `HermesClient`，
一个 OpenAI 兼容 chat 客户端，内部 `model` 硬编码为 `"hermes-agent"`。状态机强依赖
"发出文字 → 必有文字回复 → TTS 朗读"这一假设。

用户需要：

1. **写作听写**：语音转文字后**写入本地文件**，无 LLM、无回复朗读。
2. **对接任意 Agent**：如直接调 DeepSeek，只改地址/模型/key 即可；未来可能接本地
   OpenClaw / Ollama / llama.cpp 等模型。

本地模型/服务各不相同（有的要自动拉起进程、有的只是连上已运行的服务、有的是纯落盘），
无法用 shell 启动脚本承载——需要 **adapter 层同时负责"协议"和"启动/准备"**。

## 2. 目标 / 非目标

**目标**

- 抽象出一个 Backend 适配层，一个后端 = 一个子类，注册一行即可扩展。
- 实现三种类型：`hermes`（本机网关，自动配置/启动）、`openai`（任意 OpenAI 兼容端点）、`file`（落盘听写）。
- config.yaml 单一 `backend:` 块，改类型/地址/模型即换后端；无 `backend:` 块时按旧配置向后兼容。
- key 可选、按后端适配，不向 config.yaml 写入云端真 key。
- 写作模式沿用现有唤醒词框架：唤醒武装 → 连续听写逐句落盘 → 提示音反馈 → 超时回待唤醒。
- 启动入口不变：`bash scripts/start.sh`（缩为薄壳，不需 `source` 激活）。
- main.py 路径解析改为项目根绝对路径，从任意目录运行可用。

**非目标（明确不做）**

- 运行期语音热切换后端、多命名 profile + CLI 选型。
- 仓库/目录/logger 的"Hermes"软改名（README 措辞、CLAUDE.md 文件表订正）——**功能验收后再单独做**。
- 文件模式自动加标点 / 段落合并 / 时间戳（SenseVoice 无标点，逐行落盘即可）。
- 把云端 key 写进 config.yaml 并入库。
- 自动安装 Hermes Gateway 本体（仍要求 `hermes` CLI 已存在）。

## 3. Backend 适配层

### 3.1 接口 `voice/backend.py`（新增单模块）

仿照现有 `voice/stt_engine.py` 的工厂先例：

```python
class Backend:
    name = "backend"

    def prepare(self) -> None:
        """启动时调用一次：确保后端服务就绪（幂等）。云/文件型通常 no-op。
        失败时 raise BackendSetupError(message)；可选的网络类失败允许 warning 后继续。"""

    def handle(self, text: str) -> str | None:
        """处理一段转写文字。返回要朗读的回复文字；None = 静默（不朗读）。"""

    def clear_context(self) -> None:   # 会话超时/复位时调用（无状态后端 no-op）
    def close(self) -> None:           # 释放资源

def create_backend(config: dict) -> Backend: ...   # 按 config["backend"]["type"] 分发
```

实现方式：`openai`/`hermes` 共享同一套 OpenAI 兼容 HTTP 逻辑（多轮 `messages[]`、
Bearer 认证、`/v1/chat/completions`），以一个共享内部 client 为基类/组合；子类只在
`prepare()`、`name`、key 解析上不同。**`voice/hermes_client.py` 逻辑平移并泛化后移除**。

### 3.2 三种类型

| type | 协议 | `prepare()` | 用途 |
|---|---|---|---|
| `hermes` | OpenAI 兼容 | 自动配置 `~/.hermes/.env`、确保/重启 gateway、健康检查（现 start.sh 全逻辑搬入，幂等） | Hermes 本机网关（默认） |
| `openai` | OpenAI 兼容 | no-op（启动时若 key 未设给出 warning） | DeepSeek / Ollama / llama.cpp / 任何兼容服务 |
| `file` | 落盘 | 校验路径可写，失败则**中止**（此模式无它可用） | 写作听写 |

`type: hermes` 与 `type: openai` 均接 `/v1/chat/completions`；`type: hermes` 额外带
`prepare()` 管理本机 Hermes 进程。

### 3.3 配置 schema

config.yaml 中生效的唯一后端（上面默认 Hermes 块 + 下面三个替换示例）：

```yaml
# ── 后端（输出端）──────────────────────────────
# type 三选一：hermes（默认，本机网关，自动管理）| openai（任意兼容端点）| file（落盘听写）
backend:
  type: hermes
  base_url: "http://localhost:8642"     # hermes/openai 用
  model: "hermes-agent"                 # hermes/openai 用；DeepSeek 是 deepseek-chat
  api_key_env: "HERMES_API_KEY"         # 从该环境变量读 key；留空/解析不到 → 不发鉴权头
  # api_key: ""                          # 或直接写（云端 key 不建议入库）
  timeout: 120                          # hermes/openai 用
  # ── file 时 ──
  # path: "~/Documents/voice-notes.md"
```

其它示例（README/文档里给全配置块）：

```yaml
# DeepSeek（云端真 key，走环境变量）
backend:
  type: openai
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  api_key_env: "DEEPSEEK_API_KEY"
```

```yaml
# 本地已运行的 OpenAI 兼容服务（如 Ollama），无需 key → 不带 Authorization 头
backend:
  type: openai
  base_url: "http://localhost:11434/v1"
  model: "llama3.2"
```

```yaml
# 写作听写
backend:
  type: file
  path: "~/Documents/voice-notes.md"
```

**向后兼容**：config.yaml 无 `backend:` 块时，等价于
`{type: hermes, base_url: hermes_url, api_key: hermes_api_key, timeout: hermes_timeout,
model: "hermes-agent"}`，旧配置零改动直接跑。

### 3.4 key 解析规则（openai 家族）

解析顺序，第一个非空胜出：

1. `backend.api_key`（config 内直接写——简单但会被 git 提交，云端 key 不推荐）；
2. `backend.api_key_env` 指定的环境变量（如 `DEEPSEEK_API_KEY`，云端推荐）；
3. 兼容现状的 `HERMES_API_KEY` 环境变量（`type: hermes` 场景由 prepare() 注入）；
4. 都没有 → **请求不带 `Authorization` 头**（本机/无鉴权服务接受；也避免"空 key 发空 Bearer"）。

Hermes 本地场景 key 不写进 config：prepare() 负责把本机口令（沿用 `hermes-voice-key`
默认或用户 `HERMES_API_KEY`）写进 `~/.hermes/.env` 的 `API_SERVER_KEY`，并把同一口令
作为 daemon 的 key 来源。**config.yaml 永不存 Hermes 口令之外的真实密钥**。

### 3.5 错误处理

- 网络/鉴权失败：保持 `ConnectionError` 兜底，但**朗读文案泛化为中性**（如
  "后端服务不可达，请检查配置或网络"），不再写死"请先启动 Hermes 服务"。
- `file` 写盘失败（权限/磁盘/路径）：打日志并在当下 utterance 后 TTS 提示一句
  "写入文件失败"，避免用户写半天才发现没落盘。
- `prepare()` 失败：文件路径不可写 → 中止启动并报错；网络类（gateway 起不来 / 云端
  key 缺失）→ 打 warning 后继续，由运行时 ConnectionError 兜底。

## 4. main.py 状态机改动

现状 `hermes.send()` 出现在 4 处（LISTENING 主线、AWAKE 主线、`_process_interruption`、
`_renew_if_pending`）。改为统一 `backend.handle(text)`，并按返回值分叉：

```
text → reply = backend.handle(text)
        ├─ reply 非空 → 现有链路原样：_renew_if_pending → TTS 朗读(_speak_and_recover)
        │               → 打断递归(_process_interruption)     [对话模式]
        └─ reply 为空 → 播"已落盘"确认提示音，跳过全部 TTS/打断逻辑  [file 模式]
```

- **AWAKE 跟随时窗**天然适配两种模式：每次循环以 `read_utterance(session_timeout)`
  重新阻塞，故文件模式"逐句落盘、安静 30s 自动回待唤醒"无需改动状态机。
- **file 模式不进入** `set_tts_active` / `check_interrupt` / 打断递归路径（无 TTS 播放，
  无 barge-in 语义）。
- **唤醒回复可配置**：新增 `wake_greeting: "我在"`（默认保持现状 TTS"我在"；设为空
  字符串则只响提示音不说话）。对两种后端一致生效。
- **启动流程**：`backend = create_backend(config)` → `backend.prepare()`（幂等，Hermes
  自动起 gateway / file 校验路径 / openai 检查 key），随后进入原有组件初始化与循环。
  shutdown 时 `backend.close()`。
- **路径解析**：config 加载改为项目根绝对路径（复用 `models._project_root()` 思路），
  logger 已是基于 `__file__` 的绝对路径；任意 CWD 可运行。
- 启动时打一条后端 banner 日志（`后端 → hermes (http://localhost:8642)`）。

## 5. start.sh 收缩为薄壳

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { echo "尚未安装依赖，请先运行: bash scripts/install.sh"; exit 1; }
exec .venv/bin/python -m voice.main
```

- 不再 `source .venv/bin/activate`——激活只影响裸 `python`/`pip` 的 PATH，对运行无益；
  直接调 `.venv/bin/python`（自带全部依赖）即可。
- 不再承载任何后端逻辑（自动配 Hermes 等全部随 adapter 的 `prepare()` 移入 python）。
- 直接 `python -m voice.main`（或 `.venv/bin/python -m voice.main`）也可行，因 config 路径
  已 CWD 无关；start.sh 保留作带依赖守卫的单入口（install.sh / README 均引用它）。

## 6. 触碰文件

| 文件 | 改动 |
|---|---|
| `voice/backend.py` | **新增**：`Backend` ABC + `create_backend()` 工厂 + `hermes`/`openai`/`file` 三实现（含 hermes 的 gateway prepare） |
| `voice/hermes_client.py` | 逻辑平移泛化进 backend.py 后**移除** |
| `voice/main.py` | 4 处 send 收敛为 `backend.handle` + 无回复分支、启动调 `prepare()`/关闭调 `close()`、`wake_greeting` 配置、config 路径 CWD 无关、错误文案泛化 |
| `config.yaml` | 新增 `backend:` 块 + `wake_greeting`（注释含三种 type 用法） |
| `scripts/start.sh` | 收缩为薄壳（见 §5） |
| `docs/superpowers/specs/2026-09-06-backend-adapter-design.md` | 本文档 |
| `requirements.txt` | 不变 |

暂不触碰：README 措辞、CLAUDE.md 文件表、仓库目录名/remote、logger 名（软改名，功能验收后单独做）。

## 7. 验证清单（无测试框架，手动验收）

1. **file 模式**：`backend.type: file` → 唤醒说"我在…"→ 连说几句，确认逐句追加为一行、
   每句后提示音、无 TTS 回复、安静 30s 回待唤醒；文件内容为简体转写。
2. **file 异常**：把 path 指向不可写目录 → 启动即中止并报清晰错误。
3. **openai + DeepSeek**：设 `DEEPSEEK_API_KEY`，`base_url`/`model` 指向 DeepSeek → 说"你好"
   得到语音回复。不设 key → 启动 warning，请求时 401 → 泛化文案提示。
4. **openai + 本地无 key**（如 Ollama 或本机兼容服务）→ 确认请求未带 Authorization 头、正常对话。
5. **hermes 回归（默认）**：跑 `bash scripts/start.sh` → gateway 被确保运行、唤醒"我在"、
   对话/打断/跟随时窗行为与现状一致。
6. **旧配置向后兼容**：临时去掉 `backend:` 块 → 仍按顶层 hermes_* 正常工作。
7. **`wake_greeting: ""`** → 唤醒无 TTS 只提示音。
8. **任意 CWD**：在项目根以外执行 `.venv/bin/python -m voice.main` → config/log 正常找到。

## 8. 决策记录

| 项 | 决策 | 理由 |
|---|---|---|
| 后端切换粒度 | config 单后端（`backend.type`） | 最简单，符合"改配置即换"心智 |
| 写作交互 | 唤醒词武装后连续听写 | 沿用现有状态机，改动最小且可控 |
| 启动编排归属 | adapter 层 `prepare()`，进 python | 后端种类增多后 bash/sed 撑不住；adapter 同时管协议+启动 |
| start.sh | 保留为薄壳（cd + venv 守卫 + exec） | 保持单入口习惯；不 `source`，直接调 venv 解释器 |
| 本机 vs 云端判定 | 不做 host 启发式 | OpenClaw/Ollama 等都在 localhost，启发式不可靠；由 adapter 类型显式承担 |
| key 存放 | config `api_key` / `api_key_env` / 旧 `HERMES_API_KEY`，空则无鉴权头 | Hermes 是本地口令自动管；云端真 key 走环境变量，不入库 |
| 唤醒回复 | `wake_greeting`（默认"我在"，空=静默） | 写作模式可自行关，默认行为不变 |
| 软改名 | 功能验收后单独做 | 避免与功能改动混在一起难验收 |
