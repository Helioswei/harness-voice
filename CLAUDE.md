# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the voice assistant
bash scripts/start.sh            # 薄壳：校验 .venv 后 exec .venv/bin/python -m voice.main
.venv/bin/python -m voice.main   # 直接跑（无需激活环境；config/日志路径基于项目根，任意 CWD 可运行）

# Install dependencies
bash scripts/install.sh

# Activate venv (only needed to use bare `python`/`pip`)
source .venv/bin/activate

# Tests (backend adapter layer — headless)
.venv/bin/python -m pytest tests/      # 先 pip install -r requirements-dev.txt
```

No build system, no linter configured. Test framework: pytest (dev-only).

## Project identity

**Harness Voice** — a macOS wake-word-activated *voice front-end* ("小九" wakes it), not tied to any
single agent. Transcribed text goes to a pluggable **backend**: `hermes` (default, local Hermes
Gateway) / `openai` (any OpenAI-compatible endpoint) / `file` (dictation to a local file). The
logger + log file are `harness-voice`. The repo directory/GitHub name may still say `hermes-voice`
(a not-yet-done rename); that's cosmetic.

Keep the backend concept "Hermes" where it means the backend/gateway (`backend.type: hermes`,
`HERMES_API_KEY`, local handshake key literal `hermes-voice-key`, `~/.hermes`, `hermes gateway`).
Only the project *identity* is "Harness Voice".

## Architecture

Two states, managed in `voice/main.py`:

- **LISTENING** (default) — KWS (Sherpa-ONNX) waits for a wake word. On hit: beep + optional
  `wake_greeting` speech → VAD utterance → STT → route through backend → AWAKE.
- **AWAKE** — 30s follow-up window. Each utterance is transcribed and routed directly. A spoken
  reply resets the timer; silent (file) mode beeps per utterance. Timeout → LISTENING.

### Module Layout (`voice/`)

| File | Role |
|------|------|
| `main.py` | Daemon entry, signal handlers, state machine, config loading |
| `av_recorder.py` | AVAudioEngine input + Silero VAD speech detection + AEC + barge-in detection |
| `wake_word_engine.py` | Sherpa-ONNX KWS wake-word detection (multi-keyword) |
| `stt_engine.py` | STT abstraction + factory (`stt_sensevoice.py`, `stt_whisper.py`) |
| `models.py` | Model download/cache/convert (ModelScope / HuggingFace / local) |
| `backend.py` | **Backend adapter layer**: `Backend` ABC (`prepare`/`handle`/`clear_context`/`close`) + `create_backend(config)` factory; implementations `HermesBackend`, `OpenAIBackend`, `FileBackend` |
| `tts.py` | macOS AVSpeechSynthesizer (zh-CN), block-until-finished, interrupt callback |

### Key Design Details

- **Backend adapter = protocol + lifecycle.** `backend.handle(text) -> str | None`:
  non-None = text to read aloud (chat), None = silent (file dictation → beep + append).
  `backend.prepare()` runs at startup — `hermes` provisions `~/.hermes/.env` and ensures the
  gateway (idempotent); `openai`/`file` mostly no-op. `BackendError` (fatal config, e.g. unwritable
  file path) aborts startup; network/key problems warn and degrade at request time.
- **One `backend:` block in config.yaml** selects the active backend. Legacy config (no block,
  top-level `hermes_url`/`hermes_api_key`) still resolves to `hermes`.
- **Key resolution** (openai family): `backend.api_key` → env named by `backend.api_key_env` →
  `HERMES_API_KEY` → send NO auth header when empty. Cloud keys go in env vars, never in config.
- **Silent branch in main**: `_handle_turn` routes every utterance; on backend failure it speaks a
  neutral message and keeps the AWAKE window open (retry without re-waking); idle timeout returns
  to LISTENING. File write failures speak the backend error text.
- **Wake greeting** `wake_greeting` (default `"我在"`; empty string = beep only) — not a backend property.
- **Path/CWD independence**: config and logs resolve from project root (`_project_root()`), so the
  daemon runs from any working directory.
- **Wake word + STT**: KWS runs in the audio callback thread on chunks; STT runs once per buffered
  utterance delivered via threading.Event.
- **VAD flow**: silero-vad in the audio callback; speech buffers, silence beyond `silence_timeout`
  delivers the utterance.
- **TTS barge-in**: `_speak_and_recover` arms the recorder's interrupt check during speech; short
  pops (coughs < `bargein_duration`) don't trigger; recursive interrupts chain via
  `_process_interruption`.

## Configuration

`config.yaml` — session timeout, samplerate, VAD params, `wake_greeting`, STT engine/model, wake
words, `backend` block (type/base_url/model/api_key_env/timeout or file path), barge-in thresholds.
See README "后端配置" for the three `backend` profiles.

## Dependencies

Python 3.12+, macOS 13+ (AVSpeechSynthesizer), Homebrew (portaudio), default backend Hermes
(`hermes` CLI optional at runtime). Runtime deps in `requirements.txt`; test-only `pytest` in
`requirements-dev.txt` (keep runtime deps out of `requirements-dev.txt`).

## Coding Guidelines

Apply the [Karpathy LLM Coding Guidelines](https://x.com/karpathy/status/2015883857489522876):

1. **Think Before Coding** — State assumptions explicitly before implementing. Surface ambiguities and tradeoffs. Push back on overcomplicated approaches.
2. **Simplicity First** — Minimum code that solves the problem. No speculative features, abstractions, or configurability. If 200 lines can be 50, rewrite it.
3. **Surgical Changes** — Touch only what the task requires. Don't improve adjacent code, refactor what isn't broken, or change style. Clean up orphans your changes create; leave pre-existing dead code alone.
4. **Goal-Driven Execution** — Define verifiable success criteria before starting. For multi-step tasks, state plan as `step → verify: check`. Loop until criteria are met.
