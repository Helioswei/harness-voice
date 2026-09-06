# Harness Voice

macOS 上的唤醒词语音前端——喊一声"小九"唤醒，然后你说的话被转成文字，交给**可插拔的后端**：
要么发给某个 Agent（Hermes / DeepSeek / 任意 OpenAI 兼容服务）并朗读回复，要么作为听写**逐句写入本地文件**。

> **关于名字**：项目身份是 *voice*（语音前端），不绑定任何 Agent——Hermes 只是默认后端之一，
> 现在叫 *harness-voice* 正是要强调"作为 agent harness（Claude Code / Hermes 这类框架）的语音入口"。

## 特性

| 特性 | 说明 |
|------|------|
| **多唤醒词** | 可配置多个，默认"小九""轩轩"。基于 Sherpa-ONNX KWS，<100ms 唤醒延迟 |
| **多引擎 STT** | SenseVoice（默认，ONNX）与 faster-whisper（CTranslate2）两种识别引擎 |
| **可插拔后端** | 一个 `backend:` 配置块切换输出端：`hermes` / `openai`（任意兼容端点）/ `file`（听写落盘） |
| **对话 or 听写** | 后端返回文字 → TTS 朗读回复（多轮、可打断）；后端无回复（file）→ 提示音确认落盘，静默 |
| **AEC 回声消除** | AVAudioEngine 原生 voice processing，TTS 朗读时麦克风继续监听 |
| **TTS 打断（Barge-in）** | 朗读期间说话自动打断响应新指令；咳嗽等短促噪音不误触发 |
| **跟随时窗** | 唤醒后 30 秒内可直接说下一句；每次交互重置计时 |
| **后端自动管理** | `type: hermes` 时启动自动配置/确保 Hermes Gateway 运行；其余类型不需要 |
| **模型自动管理** | KWS / STT 模型按需下载缓存（ModelScope / HuggingFace / 本地） |

## 技术栈与数据流

```
你说: "小九" ──→ Sherpa-ONNX KWS 唤醒 (<100ms)
  │ beep + 可选语音问候 (wake_greeting)
  ▼
你说: "今天天气怎么样" ──→ AVAudioEngine 录音 (AEC) + Silero VAD
  │
  ▼
SenseVoice / faster-whisper STT ──→ backend.handle(text)
  │
  ├─ 返回回复文字（hermes / openai 对话）──→ AVSpeechSynthesizer TTS 朗读
  │      └─ 打断 → 转录打断内容 → 再次 handle（递归）
  └─ 返回 None（file 听写）──→ 追加一行到本地文件 + 提示音
  │
  ▼
30 秒跟随时窗（直接说下一句，无需唤醒词）
```

**组件：**

| 模块 | 作用 |
|------|------|
| `voice/main.py` | 入口 + 状态机（LISTENING / AWAKE），路径基于项目根（任意 CWD 可运行） |
| `voice/av_recorder.py` | AVAudioEngine 录音 + Silero VAD + AEC + 打断检测 |
| `voice/wake_word_engine.py` | Sherpa-ONNX KWS 唤醒词检测 |
| `voice/stt_engine.py` | STT 引擎抽象层（工厂模式） |
| `voice/stt_sensevoice.py` / `voice/stt_whisper.py` | SenseVoice / faster-whisper 识别 |
| `voice/models.py` | 模型下载、缓存、格式转换 |
| `voice/backend.py` | **后端适配层**：`Backend` 抽象 + 工厂；`hermes` / `openai` / `file` 三种实现，各负责协议 + `prepare()` 生命周期 |
| `voice/tts.py` | macOS AVSpeechSynthesizer 中文合成（支持打断回调） |

## 状态机

```
LISTENING ── [检测到唤醒词] ──→ AWAKE
    ▲                               │
    │               [安静超时 / 跟随时窗结束]
    └───────────────────────────────┘
```

- **LISTENING**：KWS 监听唤醒词。命中 → 提示音（+ 可选 `wake_greeting`）→ 录音 → STT → `backend.handle` → 进入 AWAKE。
- **AWAKE**：跟随时窗。每句直接 STT → handle；后端返回回复则朗读并重置计时；file 模式逐句落盘、提示音。安静超时 → 清上下文 → LISTENING。

## 后端配置（核心概念）

`config.yaml` 里一个 `backend:` 块决定"转写文字去哪"，改配置即换：

```yaml
# ① 本机 Hermes Gateway（默认）：启动时自动配置 ~/.hermes/.env 并确保运行
backend:
  type: hermes
  base_url: "http://localhost:8642"
  model: "hermes-agent"
  api_key_env: "HERMES_API_KEY"   # 本地口令，通常由 prepare() 自动处理，无需你填
  timeout: 120

# ② 任意 OpenAI 兼容端点：DeepSeek / Ollama / llama.cpp … 只改 type/base_url/model
# backend:
#   type: openai
#   base_url: "https://api.deepseek.com"     # 只填 origin，不写 /v1
#   model: "deepseek-chat"
#   api_key_env: "DEEPSEEK_API_KEY"          # 云端真 key 走环境变量

# ③ 听写落盘（写作）：无回复、静默 + 提示音
# backend:
#   type: file
#   path: "~/Documents/voice-notes.md"
```

**key 规则**：`backend.api_key` → `backend.api_key_env` 指定的环境变量 → 旧 `HERMES_API_KEY` →
都为空则**请求不带鉴权头**（本机/无鉴权服务）。云端 key 建议放环境变量，勿写进入库的 config。

**向后兼容**：旧配置（无 `backend:` 块，顶层 `hermes_url`/`hermes_api_key`）仍会被当作
`type: hermes` 处理。

其它有用配置项：`wake_greeting`（唤醒语音问候，默认 `"我在"`，留空则只响提示音）、
`kws_keywords`、`stt_engine`、`bargein_*`、`session_timeout`。

## 快速开始

```bash
# 1. 安装依赖（建 .venv）
bash scripts/install.sh

# 2. 启动（默认后端为 hermes；Hermes Gateway 由 daemon 启动时自动确保运行）
bash scripts/start.sh            # 可从任意目录调用（脚本自定位仓库根）

#    直接跑 daemon（无需激活环境；config/日志路径基于项目根）——
#    但需在仓库根目录执行，或用上面的 start.sh 以支持任意目录：
#    .venv/bin/python -m voice.main
```

`start.sh` 只是薄壳（自定位仓库根 → exec venv python），所以**任意 CWD 都能启动**。
Hermes 之外的云端后端，按上面的 `backend:` 示例配好环境变量即可。

## 测试

```bash
.venv/bin/python -m pytest tests/           # 依赖: pip install -r requirements-dev.txt
```

后端适配层（请求形状 / 鉴权头 / key 解析 / 落盘 / env 同步幂等）为无头单测；唤醒/录音/TTS
等需真机手工验收。

## 日志标记

| 标记 | 含义 |
|------|------|
| `唤醒词 →` | KWS 检测到唤醒词 |
| `麦克风→` | 麦克风 → STT 转写结果 |
| `打断→` | TTS 期间用户说话 → 打断转录 |
| `朗读 →` | TTS 正在朗读后端回复 |
| `后端 →` | 启动时选中的后端（name + base_url） |
| `状态机 →` | 状态切换 |
| `TTS 被用户打断` | TTS 被用户语音打断 |

日志文件：`logs/harness-voice.log`

## 依赖

- Python 3.12+，macOS 13+（AVAudioEngine voice processing / AVSpeechSynthesizer）
- Homebrew（portaudio）
- 默认后端 Hermes：`hermes` CLI（不装也启动，运行时按 ConnectionError 提示）

## 项目结构

```
.
├── config.yaml             # 配置（backend 块 / wake_greeting / STT / 打断）
├── requirements.txt        # 运行时依赖
├── requirements-dev.txt    # 测试依赖（pytest）
├── models/                 # 模型文件（自动下载缓存）
├── scripts/
│   ├── install.sh          # 安装依赖
│   └── start.sh            # 薄壳启动（校验 .venv → exec voice.main）
├── docs/                   # 设计/计划文档（含 superpowers spec & plan）
├── tests/test_backend.py   # 后端适配层单测
└── voice/
    ├── main.py             # 守护进程 + 状态机
    ├── backend.py          # Backend 适配层（hermes / openai / file）
    ├── av_recorder.py / wake_word_engine.py / tts.py
    ├── stt_engine.py / stt_sensevoice.py / stt_whisper.py
    └── models.py
```
