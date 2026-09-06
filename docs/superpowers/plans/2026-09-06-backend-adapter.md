# Backend 输出端适配层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 daemon 的转写文字下游从写死的 `HermesClient` 泛化为可插拔 `Backend` 适配层，支持 `hermes`（本机网关，自动配置）/ `openai`（任意 OpenAI 兼容端点，如 DeepSeek / Ollama）/ `file`（落盘听写）三种后端，启动入口 `bash scripts/start.sh` 保持不变但收缩为薄壳。

**Architecture:** 新增 `voice/backend.py`，仿 `stt_engine.py` 工厂先例：一个 `Backend` ABC（`prepare()` 管启动/准备，`handle(text) -> str|None` 管协议，返回 `None` 即静默不朗读）+ `create_backend(config)` 工厂。OpenAI 兼容协议抽成共享基类，`openai`/`hermes` 复用；`file` 纯落盘。`voice/main.py` 四处置换为 `backend.handle()` 并按返回值分叉（有回复→朗读/打断链；无回复→确认提示音）。Hermes 生命周期（`~/.hermes/.env` 配置、gateway 启停）从 `start.sh` 迁入 `HermesBackend.prepare()`。repo 采用 venv `.venv`，无测试框架，本计划引入 `pytest`（dev-only）。

**Tech Stack:** Python 3.12+，httpx（已依赖），macOS；测试用 `pytest` + `httpx.MockTransport`（无头，不依赖麦克风/AVFoundation）。

**Spec:** `docs/superpowers/specs/2026-09-06-backend-adapter-design.md`

## Global Constraints

- 无新增**运行时**依赖（`requirements.txt` 不变）；测试依赖放 `requirements-dev.txt`。
- `backend.base_url` 一律填 **origin**（`scheme://host[:port]`，不含路径）；请求端点固定为 `<base_url>/v1/chat/completions`。文档/配置示例遵守此约定。
- 日志统一 `logging.getLogger("hermes-voice")`，沿用现有 logger。
- key 解析规则（spec §3.4）：`api_key`(config) → `api_key_env` 指定环境变量 → 旧 `HERMES_API_KEY` 环境变量 → 空则**不带** `Authorization` 头。
- 网络错误以 `ConnectionError` 上抛（main 兜底），文件写盘等本地失败以 `BackendError` 上抛。
- 本计划**不**做软改名（README 措辞 / CLAUDE.md 文件表 / logger 名 / 目录名 / remote）。
- 运行与测试用 `.venv/bin/python`（勿依赖 shell 激活）。
- 分支：执行在 `main`。每个任务独立 commit，遵循仓库现有 `feat:`/`fix:`/`docs:` 风格 + `Co-Authored-By` 尾行。

---

### Task 1: 测试脚手架 + `voice/backend.py`（ABC / key 解析 / OpenAI 兼容核心 + `openai` 后端 + 工厂初版）

**Files:**
- Create: `requirements-dev.txt`
- Create: `voice/backend.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Produces（后续任务依赖的精确签名）：
  - `class Backend` — `name: str`；`prepare() -> None`；`handle(text: str) -> str | None`；`clear_context() -> None`；`close() -> None`。
  - `class BackendError(Exception)`。
  - `_resolve_key(cfg: dict, *, fallback_env: str | None = None, default: str | None = None) -> str | None`。
  - `class OpenAIBackend(Backend)` — `name = "openai"`；`__init__(self, cfg: dict, transport=None)`。
  - `create_backend(config: dict) -> Backend` — 只认 `config["backend"]["type"]`；本任务仅注册 `openai`，未知 type 抛 `ValueError`。

- [ ] **Step 1: 建 dev 依赖文件**

Create `requirements-dev.txt`:
```
pytest>=8
```

- [ ] **Step 2: 安装 dev 依赖**

Run: `.venv/bin/python -m pip install -r requirements-dev.txt`
Expected: 成功；`pytest` 可用。

- [ ] **Step 3: 写失败测试**

Create `tests/test_backend.py`:
```python
import httpx

from voice.backend import OpenAIBackend, _resolve_key, create_backend


def make_handler(status=200, body=None):
    body = body or {"choices": [{"message": {"content": "回复A"}}]}
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["payload"] = httpx.Request(
            request.method, str(request.url), content=request.content
        )  # noqa — 简化用 json 字段
        import json
        captured["payload"] = json.loads(request.content)
        return httpx.Response(status, json=body)

    return handler, captured


def test_handle_posts_to_openai_compat_endpoint():
    handler, cap = make_handler()
    b = OpenAIBackend(
        {"type": "openai", "base_url": "https://api.deepseek.com",
         "model": "deepseek-chat", "api_key": "sk-abc"}, transport=httpx.MockTransport(handler)
    )
    reply = b.handle("你好")
    assert reply == "回复A"
    assert cap["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert cap["payload"]["model"] == "deepseek-chat"
    assert cap["payload"]["messages"] == [{"role": "user", "content": "你好"}]
    assert cap["headers"]["authorization"] == "Bearer sk-abc"


def test_no_key_means_no_auth_header():
    handler, cap = make_handler()
    b = OpenAIBackend(
        {"type": "openai", "base_url": "http://localhost:11434",
         "model": "llama3.2"}, transport=httpx.MockTransport(handler)
    )
    b.handle("hi")
    assert "authorization" not in cap["headers"]


def test_multiturn_context_and_clear():
    handler, cap = make_handler()
    b = OpenAIBackend(
        {"type": "openai", "base_url": "http://x", "model": "m"},
        transport=httpx.MockTransport(handler),
    )
    b.handle("q1")
    b.handle("q2")
    assert [m["role"] for m in cap["payload"]["messages"]] == ["user", "assistant", "user"]
    b.clear_context()
    b.handle("q3")
    assert cap["payload"]["messages"] == [{"role": "user", "content": "q3"}]


def test_network_error_maps_to_connection_error():
    def boom(request):
        raise httpx.ConnectError("refused")

    b = OpenAIBackend(
        {"type": "openai", "base_url": "http://x", "model": "m"},
        transport=httpx.MockTransport(boom),
    )
    try:
        b.handle("hi")
        assert False, "应抛出 ConnectionError"
    except ConnectionError:
        pass
    assert b.messages == []  # 失败时已回滚 user 消息


def test_resolve_key_precedence(monkeypatch):
    # 1) api_key 优先于 env
    assert _resolve_key({"api_key": "cfg", "api_key_env": "KKK"}) == "cfg"
    # 2) api_key_env 读环境变量
    monkeypatch.setenv("KKK", "envval")
    assert _resolve_key({"api_key_env": "KKK"}) == "envval"
    # 3) fallback_env
    monkeypatch.setenv("HERMES_API_KEY", "legacy")
    assert _resolve_key({"api_key_env": "KKK"}, fallback_env="HERMES_API_KEY") == "envval"
    assert _resolve_key({}, fallback_env="HERMES_API_KEY") == "legacy"
    # 4) 都没有 → default / None
    assert _resolve_key({}) is None
    assert _resolve_key({}, default="d") == "d"


def test_openai_no_key_prepare_ok(capsys):
    b = OpenAIBackend({"type": "openai", "base_url": "http://x", "model": "m"})
    b.prepare()  # 不应抛异常，仅 warning


def test_factory_unknown_type_raises():
    try:
        create_backend({"backend": {"type": "nope"}})
        assert False, "应抛 ValueError"
    except ValueError:
        pass
```

- [ ] **Step 4: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: voice.backend`。

- [ ] **Step 5: 实现 `voice/backend.py`（本任务范围：openai）**

Create `voice/backend.py`:
```python
"""Backend adapters — where the transcribed text goes after STT.

Each backend owns both the *protocol* (how text is sent and whether a
spoken reply comes back) and the *lifecycle* (``prepare()`` — how to make
sure its service is ready). Add a new backend by subclassing ``Backend``
and registering it in ``create_backend()``.

Convention: ``base_url`` is the origin only (``scheme://host[:port]``,
no path). Requests go to ``<base_url>/v1/chat/completions``.
"""

import logging
import os

import httpx

logger = logging.getLogger("hermes-voice")


class BackendError(Exception):
    """Recoverable backend failure carrying a user-facing reason."""


class Backend:
    """Abstract backend. See module docstring."""

    name = "backend"

    def prepare(self):
        """Ensure the service is ready. Idempotent; no-op by default."""

    def handle(self, text):
        """Process one transcribed utterance.

        Returns text to read aloud, or ``None`` to stay silent.
        """
        raise NotImplementedError

    def clear_context(self):
        """Reset multi-turn context (no-op for stateless backends)."""

    def close(self):
        """Release resources."""


def _resolve_key(cfg, *, fallback_env=None, default=None):
    """First non-empty of: cfg.api_key, env(cfg.api_key_env), fallback_env, default."""
    if cfg.get("api_key"):
        return cfg["api_key"]
    env_name = cfg.get("api_key_env") or fallback_env
    if env_name:
        val = os.environ.get(env_name)
        if val:
            return val
    return default


class _OpenAIChat(Backend):
    """Shared OpenAI-compatible chat plumbing (multi-turn, Bearer auth)."""

    def __init__(self, base_url, model, api_key=None, timeout=120, transport=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.messages = []
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout), transport=transport
        )

    def handle(self, text):
        self.messages.append({"role": "user", "content": text})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": list(self.messages),
            "stream": False,
        }
        try:
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self.messages.pop()
            raise ConnectionError(f"Backend request failed: {exc}") from exc
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def clear_context(self):
        self.messages.clear()

    def close(self):
        self._client.close()


class OpenAIBackend(_OpenAIChat):
    """Raw OpenAI-compatible endpoint (DeepSeek, Ollama, llama.cpp, ...)."""

    name = "openai"

    def __init__(self, cfg, transport=None):
        key = _resolve_key(cfg)
        super().__init__(
            base_url=cfg.get("base_url", ""),
            model=cfg.get("model", ""),
            api_key=key,
            timeout=cfg.get("timeout", 120),
            transport=transport,
        )

    def prepare(self):
        if not self.api_key:
            logger.warning(
                "openai 后端未配置 key（backend.api_key 或 api_key_env）——将不带鉴权头发送"
            )
        if not self.base_url:
            raise BackendError("openai 后端缺少 base_url")


def create_backend(config):
    """Build the active backend from a full config dict."""
    bcfg = config.get("backend")
    if bcfg is None:
        raise BackendError("config 缺少 backend 块")
    kind = bcfg.get("type")
    if kind == "openai":
        return OpenAIBackend(bcfg)
    raise ValueError(f"Unknown backend type: {kind}")
```

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add requirements-dev.txt voice/backend.py tests/test_backend.py
git commit -m "feat: add OpenAI-compatible backend core (openai type, key resolution, factory)"
```

---

### Task 2: `file` 后端（落盘听写）并注册进工厂

**Files:**
- Modify: `voice/backend.py`（追加 `FileBackend`；工厂加 `file` 分支）
- Test: `tests/test_backend.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `Backend`、`BackendError`。
- Produces: `class FileBackend(Backend)` — `name = "file"`；`__init__(self, cfg)`；`prepare()` 校验路径可写，失败抛 `BackendError`；`handle(text) -> None`（追加一行，失败抛 `BackendError`）。

- [ ] **Step 1: 追加失败测试**

Append to `tests/test_backend.py`:
```python
from voice.backend import BackendError, FileBackend, create_backend


def test_file_handle_appends_line_and_returns_none(tmp_path):
    p = tmp_path / "notes.md"
    b = FileBackend({"type": "file", "path": str(p)})
    b.prepare()
    assert b.handle("第一句") is None
    assert b.handle("第二句") is None
    assert p.read_text(encoding="utf-8") == "第一句\n第二句\n"


def test_file_prepare_rejects_unwritable_dir(tmp_path):
    missing = tmp_path / "no" / "such" / "notes.md"
    b = FileBackend({"type": "file", "path": str(missing)})
    try:
        b.prepare()
        assert False, "应抛 BackendError"
    except BackendError:
        pass


def test_factory_returns_file_backend():
    b = create_backend({"backend": {"type": "file", "path": "/tmp/x.md"}})
    assert b.name == "file"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 新 3 条 FAIL — `AttributeError: module 'voice.backend' has no attribute 'FileBackend'`。

- [ ] **Step 3: 实现 `FileBackend` 并注册**

Append to `voice/backend.py` (imports 已含 `os`):
```python
class FileBackend(Backend):
    """Dictation-to-file: append each utterance as one line, stay silent."""

    name = "file"

    def __init__(self, cfg):
        self.path = os.path.expanduser(cfg.get("path", "~/voice-notes.txt"))

    def prepare(self):
        try:
            open(self.path, "a", encoding="utf-8").close()
        except OSError as exc:
            raise BackendError(f"无法写入听写文件 {self.path}: {exc}")

    def handle(self, text):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")
        except OSError as exc:
            raise BackendError(f"写入听写文件失败 {self.path}: {exc}")
        return None
```

在 `create_backend` 中，`kind == "openai"` 分支之后加：
```python
    if kind == "file":
        return FileBackend(bcfg)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add voice/backend.py tests/test_backend.py
git commit -m "feat: add file backend (dictation to local file, silent reply)"
```

---

### Task 3: `hermes` 后端（生命周期 prepare 迁入）+ 旧配置向后兼容

**Files:**
- Modify: `voice/backend.py`（追加 `HermesBackend` + `prepare()`；工厂加 `hermes` 分支 + `backend` 缺失时的向后兼容合成）
- Test: `tests/test_backend.py`（追加）

**Interfaces:**
- Consumes: Task 1 `_resolve_key`、`_OpenAIChat`、`BackendError`。
- Produces: `class HermesBackend(_OpenAIChat)` — `name = "hermes"`；`__init__(self, cfg, transport=None, hermes_env=None)`；`_sync_env_file() -> bool`（编辑 `~/.hermes/.env`，返回是否改动，纯本地可测）；`prepare()`（幂等：同步 env → 有改动则重启 gateway → 确保运行 → 健康检查；`hermes` CLI 缺失或 gateway 起不来仅 warn，不抛）。

- [ ] **Step 1: 追加失败测试**

Append to `tests/test_backend.py`:
```python
from pathlib import Path

from voice.backend import HermesBackend, _resolve_key, create_backend


def test_hermes_key_defaults_when_nothing_set(monkeypatch):
    for k in ("HERMES_API_KEY", "SOME_OTHER"):
        monkeypatch.delenv(k, raising=False)
    assert _resolve_key({}, fallback_env="HERMES_API_KEY", default="hermes-voice-key") == "hermes-voice-key"


def test_hermes_reads_legacy_env(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "custom-key")
    assert _resolve_key({}, fallback_env="HERMES_API_KEY", default="hermes-voice-key") == "custom-key"


def test_hermes_sync_env_file_writes_defaults(tmp_path):
    env = tmp_path / ".env"
    b = HermesBackend(
        {"type": "hermes", "base_url": "http://localhost:8642"},
        hermes_env=str(env),
    )
    changed = b._sync_env_file()
    assert changed is True
    text = env.read_text(encoding="utf-8")
    assert "API_SERVER_ENABLED=true" in text
    assert "API_SERVER_KEY=hermes-voice-key" in text


def test_hermes_sync_env_file_idempotent(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "API_SERVER_ENABLED=true\nAPI_SERVER_KEY=hermes-voice-key\n", encoding="utf-8"
    )
    b = HermesBackend({"type": "hermes"}, hermes_env=str(env))
    assert b._sync_env_file() is False  # 无改动
    b.api_key = "changed-key"
    assert b._sync_env_file() is True


def test_legacy_config_without_backend_block_defaults_to_hermes(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    # hermes prepare 会碰 CLI；这里只验证 create_backend 合成默认后端类型
    b = create_backend(
        {"hermes_url": "http://localhost:8642", "hermes_api_key": "hermes-voice-key"}
    )
    assert b.name == "hermes"
    assert b.base_url == "http://localhost:8642"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 新 5 条 FAIL — `AttributeError: module 'voice.backend' has no attribute 'HermesBackend'`。

- [ ] **Step 3: 实现 `HermesBackend` 与向后兼容**

追加 `imports`（文件顶部）：把现有
```python
import logging
import os

import httpx
```
改为
```python
import logging
import os
import shutil
import subprocess
from pathlib import Path

import httpx
```
（移除 `os` 下面的空行即可，保留 `logger`）。

Append to `voice/backend.py`:
```python
class HermesBackend(_OpenAIChat):
    """Local Hermes Gateway: OpenAI-compatible chat + lifecycle management.

    ``prepare()`` idempotently provisions ~/.hermes/.env, (re)starts the
    gateway when its config changed, ensures it is running, and health-checks
    it — the logic formerly living in scripts/start.sh.
    """

    name = "hermes"

    def __init__(self, cfg, transport=None, hermes_env=None):
        key = _resolve_key(
            cfg, fallback_env="HERMES_API_KEY", default="hermes-voice-key"
        )
        super().__init__(
            base_url=cfg.get("base_url", "http://localhost:8642"),
            model=cfg.get("model", "hermes-agent"),
            api_key=key,
            timeout=cfg.get("timeout", 120),
            transport=transport,
        )
        self.hermes_env = Path(
            hermes_env or os.path.join(os.path.expanduser("~"), ".hermes", ".env")
        )

    def _sync_env_file(self):
        """Ensure API_SERVER_ENABLED=true and API_SERVER_KEY=self.api_key.

        Returns True if the file changed.
        """
        changed = False
        text = ""
        if self.hermes_env.exists():
            text = self.hermes_env.read_text(encoding="utf-8")
        lines = text.splitlines()
        has_enabled = "API_SERVER_ENABLED=true" in text
        key_line = f"API_SERVER_KEY={self.api_key}"
        if not has_enabled:
            lines.append("")
            lines.append("# Hermes API Server (auto-configured by voice backend)")
            lines.append("API_SERVER_ENABLED=true")
            changed = True
        if key_line not in text:
            lines = [key_line if l.startswith("API_SERVER_KEY=") else l for l in lines]
            if key_line not in lines:
                lines.append(key_line)
            changed = True
        if changed:
            self.hermes_env.parent.mkdir(parents=True, exist_ok=True)
            self.hermes_env.write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")
        return changed

    def _hermes_cli(self, args):
        if not shutil.which("hermes"):
            return None
        try:
            return subprocess.run(
                ["hermes", *args],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            ).returncode
        except Exception:
            return None

    def prepare(self):
        if self._sync_env_file():
            logger.info("Hermes API Server 配置已更新，重启 gateway …")
            self._hermes_cli(["gateway", "restart"])
        if self._hermes_cli(["gateway", "status"]) != 0:
            logger.info("启动 Hermes Gateway …")
            self._hermes_cli(["gateway", "start"])
        # 健康检查（尽力而为，失败仅告警，交由运行时 ConnectionError 兜底）
        try:
            r = self._client.get(f"{self.base_url}/v1/health", timeout=5)
            if r.status_code == 200:
                logger.info("Hermes API Server 正常")
            else:
                logger.warning("Hermes API Server 健康检查非 200 (%s)", r.status_code)
        except httpx.HTTPError:
            logger.warning("Hermes API Server 无响应（可稍后随请求重试）")
```

在 `create_backend` 中，把开头"config 缺 backend 块即抛错"替换为向后兼容合成：
```python
def create_backend(config):
    """Build the active backend from a full config dict."""
    bcfg = config.get("backend")
    if bcfg is None:
        # 向后兼容：无 backend 块 → 顶层 hermes_* 老配置
        bcfg = {
            "type": "hermes",
            "base_url": config.get("hermes_url", "http://localhost:8642"),
            "api_key": config.get("hermes_api_key", ""),
            "timeout": config.get("hermes_timeout", 120),
        }
    kind = bcfg.get("type", "hermes")
    if kind == "hermes":
        return HermesBackend(bcfg)
    if kind == "openai":
        return OpenAIBackend(bcfg)
    if kind == "file":
        return FileBackend(bcfg)
    raise ValueError(f"Unknown backend type: {kind}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add voice/backend.py tests/test_backend.py
git commit -m "feat: add hermes backend with lifecycle prepare; legacy config fallback"
```

---

### Task 4: `main.py` 收敛到 `backend.handle()` + 静默分支 + 启动/关闭接入 + `wake_greeting`

**Files:**
- Modify: `voice/main.py`
- Delete: `voice/hermes_client.py`

**Interfaces:**
- Consumes: `create_backend(config)`、`BackendError`（来自 `.backend`）。
- Produces: 无（仅内部重构）。main 行为约定：唤醒默认 TTS"我在"（`wake_greeting` 可配置/空即静默）；每句走 `_handle_turn`，有回复→朗读/打断，无回复→`play_beep()` 确认；启动调 `backend.prepare()`（失败仅 warn），关闭调 `backend.close()`。

- [ ] **Step 1: 改导入与路径加载**

在 `voice/main.py`：
1. 顶部 import：`from .hermes_client import HermesClient` → `from .backend import BackendError, create_backend`
2. 在 `load_config` 上方加模块级助手：
```python
def _project_root():
    """项目根目录（voice/ 的上一级），保证任意 CWD 可运行。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```
3. `load_config` 改为路径缺省解析到项目根：
```python
def load_config(path=None):
    if path is None:
        path = os.path.join(_project_root(), "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("HERMES_API_KEY") or cfg.get("hermes_api_key", "")
    cfg["hermes_api_key"] = api_key
    return cfg
```
（保留 env 合并是为向后兼容顶层 `hermes_api_key`；`backend` 块存在时该字段不再被读取。）

- [ ] **Step 2: 收紧 `validate_config`**

把 `validate_config` 中 `required` 字典改为只保留与会话无关后端必填项：
```python
def validate_config(cfg):
    """Validate required config keys and their types."""
    required = {
        "samplerate": (int, float),
        "silence_timeout": (int, float),
    }
    for key, expected_type in required.items():
        if key not in cfg:
            raise ValueError(f"配置缺少必要字段: {key}")
        if not isinstance(cfg[key], expected_type):
            expected_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else " or ".join(t.__name__ for t in expected_type)
            )
            raise TypeError(
                f"配置字段 {key} 类型错误: "
                f"期望 {expected_name}, 实际 {type(cfg[key]).__name__}"
            )
```
（删除原 `hermes_url`/`hermes_api_key` 必填项；backend 结构交给 `create_backend()` 校验。）

- [ ] **Step 3: main() 初始化——建后端 + prepare**

把 main() 中
```python
    hermes = HermesClient(
        base_url=config.get("hermes_url", "http://localhost:8642"),
        api_key=config["hermes_api_key"],
        timeout=config.get("hermes_timeout", 120),
    )
    tts = TTSEngine()
```
替换为
```python
    backend = create_backend(config)
    try:
        backend.prepare()
    except BackendError as exc:
        # 配置级致命错误（如 file 路径不可写 / openai 缺 base_url）：中止
        logger.critical("后端准备失败: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("后端 prepare 异常，继续以当前配置启动")
    logger.info("后端 → %s (%s)", backend.name, getattr(backend, "base_url", ""))
    tts = TTSEngine()
```

- [ ] **Step 4: 新增 `_handle_turn` 并更新两个打断/续答助手**

把现有 `_process_interruption`、`_renew_if_pending` 整体替换为以下三块（保持它们在 `_speak_and_recover` 定义之后、主循环之前）：

```python
    def _handle_turn(recorder, stt, backend, text):
        """Route one transcribed utterance through the active backend.

        Returns interrupted audio (speech captured during TTS barge-in) if a
        spoken reply was given and cut short, else None.
        """
        try:
            reply = backend.handle(text)
        except ConnectionError as exc:
            logger.warning("后端不可达 (%s)", exc)
            tts.speak("后端服务不可达，请检查配置或网络")
            return None
        except BackendError as exc:
            logger.error("后端处理失败: %s", exc)
            tts.speak("处理失败，请查看日志")
            return None

        reply = _renew_if_pending(recorder, stt, backend, reply)

        if reply:
            logger.info("朗读 → %s", reply)
            return _speak_and_recover(recorder, tts, reply)
        # 无文字回复（file 听写）：响一声确认，不朗读
        play_beep()
        return None

    def _process_interruption(recorder, stt, backend, tts, interrupted_audio):
        """Transcribe and respond to audio captured during TTS barge-in.

        Uses the already-captured *interrupted_audio* directly. Handles
        recursive interrupts — if the reply TTS is also barged in, the
        chain continues naturally.
        """
        if interrupted_audio is None:
            return

        text = _to_simplified(stt.transcribe(interrupted_audio))
        if not _is_valid_speech(text):
            return

        logger.info("打断→ %s", text)
        more_audio = _handle_turn(recorder, stt, backend, text)
        if more_audio is not None:
            _process_interruption(recorder, stt, backend, tts, more_audio)

    def _renew_if_pending(recorder, stt, backend, current_reply):
        """If VAD captured speech during last blocking call, re-request.

        Consumes audio VAD stored while ``backend.handle()`` blocked (slow
        LLM / tool calls), transcribes it, and makes a fresh request so the
        assistant answers what the user *actually* just said.
        """
        audio = recorder.consume_utterance()
        if audio is None:
            return current_reply
        text = _to_simplified(stt.transcribe(audio))
        if not text or not text.strip():
            return current_reply
        logger.info("等待期间→ %s", text)
        try:
            return backend.handle(text)
        except ConnectionError:
            return current_reply
```

> 注：`_handle_turn` 引用的 `_speak_and_recover`、`_renew_if_pending` 需先定义；`_process_interruption` 依赖 `_handle_turn`。按 `_speak_and_recover` → `_handle_turn` → `_process_interruption` → `_renew_if_pending` 顺序放置即可（Python 运行时才解析名字，顺序只影响可读性）。

- [ ] **Step 5: 唤醒回复可配置**

把 main() 中
```python
                play_beep()
                tts.speak("我在")
```
替换为
```python
                play_beep()
                greeting = config.get("wake_greeting", "我在")
                if greeting:
                    tts.speak(greeting)
```

- [ ] **Step 6: 替换 LISTENING 分支的发送块**

把
```python
                reply = hermes.send(text)
                reply = _renew_if_pending(recorder, stt, hermes, reply)
                logger.info("朗读 → %s", reply)
                interrupted = _speak_and_recover(recorder, tts, reply)
                state = "AWAKE"
                recorder.start()
                logger.info("状态机 → 进入跟随时窗 (%.0f秒)",
                            session_timeout)
                _process_interruption(recorder, stt, hermes, tts, interrupted)
```
替换为
```python
                interrupted = _handle_turn(recorder, stt, backend, text)
                state = "AWAKE"
                recorder.start()
                logger.info("状态机 → 进入跟随时窗 (%.0f秒)",
                            session_timeout)
                _process_interruption(recorder, stt, backend, tts, interrupted)
```

- [ ] **Step 7: 替换 AWAKE 分支的发送块**

把
```python
                reply = hermes.send(text)
                reply = _renew_if_pending(recorder, stt, hermes, reply)
                logger.info("朗读 → %s", reply)
                interrupted = _speak_and_recover(recorder, tts, reply)
                _process_interruption(recorder, stt, hermes, tts, interrupted)
```
替换为
```python
                interrupted = _handle_turn(recorder, stt, backend, text)
                _process_interruption(recorder, stt, backend, tts, interrupted)
```

- [ ] **Step 8: 兜底文案泛化 + 上下文/关闭接线**

1. AWAKE 超时块：`hermes.clear_context()` → `backend.clear_context()`。
2. shutdown handler 与函数尾：`hermes.close()` → `backend.close()`（共 2 处）。
3. 外层 `except ConnectionError`：`tts.speak("请先启动 Hermes 服务")` → `tts.speak("后端服务不可达，请检查配置或网络")`。

- [ ] **Step 9: 删除旧 client，编译/导入自检**

Run:
```bash
rm voice/hermes_client.py
.venv/bin/python -m compileall voice/main.py voice/backend.py
.venv/bin/python -c "from voice.backend import create_backend; import yaml; c=yaml.safe_load(open('config.yaml')); b=create_backend(c); print('backend:', b.name)"
```
Expected: 无语法错；最后一行用**当前旧 config.yaml**（无 `backend` 块）打印 `backend: hermes`（验证向后兼容 + 不 import AVFoundation 也能建后端）。

- [ ] **Step 10: 全量单测仍绿 + 冒烟**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 全部 PASS。

- [ ] **Step 11: Commit**

```bash
git add -A voice
git commit -m "refactor: route utterances through backend adapter in main; drop HermesClient"
```

---

### Task 5: 新 config.yaml（backend 块 + wake_greeting）与薄壳 start.sh

**Files:**
- Modify: `config.yaml`（整体替换）
- Modify: `scripts/start.sh`（整体替换）

- [ ] **Step 1: 重写 `config.yaml`**

把整个文件替换为：
```yaml
# ── 会话 ──────────────────────────────────────
session_timeout: 30          # 跟随时窗秒数
samplerate: 16000            # 音频采样率
silence_timeout: 1.5         # VAD 静音判定秒数
max_record_sec: 15           # 单次录音上限

# ── 唤醒 ──────────────────────────────────────
wake_greeting: "我在"        # 唤醒后朗读的话；留空 "" 则只响提示音

# ── STT 语音转文字引擎 ────────────────────────
stt_engine: sensevoice
stt_model_size: small        # tiny / base / small / medium（whisper 引擎用）
stt_model_source: modelscope # huggingface / modelscope / local
stt_model_path: ""           # 本地模型路径（source=local 时使用）

# ── 唤醒词 ─────────────────────────────────────
kws_keywords:                # 自定义唤醒词（可多个）
  - "小九"
  - "轩轩"
kws_threshold: 0.25          # 唤醒阈值（0.1-0.9，越高越严格）

# ── 后端（输出端：转写文字去哪）────────────────
# type: hermes | openai | file
#   hermes 本机 Hermes Gateway——prepare() 自动配置/确保运行（默认）
#   openai 任意 OpenAI 兼容端点（DeepSeek/Ollama/llama.cpp…）；base_url 填 origin，协议固定 /v1/chat/completions
#   file   落盘听写（写作）：每句追加一行，无回复朗读
backend:
  type: hermes
  base_url: "http://localhost:8642"
  model: "hermes-agent"
  api_key_env: "HERMES_API_KEY"   # 从环境变量读 key；解析不到则不带鉴权头
  timeout: 120                    # HTTP 请求超时（秒），工具调用可能耗时较长

# 切换示例：把上面整个 backend 块替换为以下之一 ──
#
# # DeepSeek（云端，真 key 走环境变量，勿入库）
# backend:
#   type: openai
#   base_url: "https://api.deepseek.com"
#   model: "deepseek-chat"
#   api_key_env: "DEEPSEEK_API_KEY"
#
# # 本地 Ollama（OpenAI 兼容，无需 key）
# backend:
#   type: openai
#   base_url: "http://localhost:11434"
#   model: "llama3.2"
#
# # 写作听写（落盘）
# backend:
#   type: file
#   path: "~/Documents/voice-notes.md"

# ── TTS 打断（Barge-in） ────────────────────────
bargein_threshold: 0.55      # TTS 期间 VAD 阈值（0.0-1.0，越高越严格）
bargein_duration: 0.5        # 打断确认时长（秒，咳嗽 <500ms 会被过滤）
```

- [ ] **Step 2: 重写 `scripts/start.sh` 为薄壳**

把整个文件替换为：
```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "尚未安装依赖，请先运行: bash scripts/install.sh"
    exit 1
fi

exec .venv/bin/python -m voice.main
```
说明：不再 `source` 激活（激活只影响裸 `python`/`pip`，直接调 `.venv/bin/python` 自带依赖）；Hermes gateway 的自动配置/启停已由 `HermesBackend.prepare()` 承担，脚本不再承载任何后端逻辑。

- [ ] **Step 3: 语法校验**

Run: `bash -n scripts/start.sh && .venv/bin/python -c "import yaml; yaml.safe_load(open('config.yaml')); print('config ok')"`
Expected: 无输出错误；打印 `config ok`。

- [ ] **Step 4: 后端工厂对新 config 冒烟**

Run:
```bash
.venv/bin/python - <<'PY'
import yaml
from voice.backend import create_backend
c = yaml.safe_load(open("config.yaml"))
b = create_backend(c)
print("active backend:", b.name, b.base_url, "model:", b.model)
PY
```
Expected: `active backend: hermes http://localhost:8642 model: hermes-agent`

- [ ] **Step 5: 全量单测仍绿**

Run: `.venv/bin/python -m pytest tests/test_backend.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add config.yaml scripts/start.sh
git commit -m "feat: configurable backend block + wake_greeting; start.sh as thin launcher"
```

---

### 手工验收（需麦克风/真机，代替 end-to-end 自动化）

在完成全部任务后、宣称完成前逐条执行（对应 spec §7）：

1. **file 模式**：把 config 换成 file 后端 → `bash scripts/start.sh` → 唤醒说话几句 → 确认逐句追加为一行、每句后提示音、无 TTS 回复、安静 30s 回待唤醒；文件为简体。
2. **file 异常**：path 指向不可写目录 → 启动即中止/强烈告警（见 `FileBackend.prepare`）。
3. **openai + DeepSeek**：设 `DEEPSEEK_API_KEY`，backend 指向 deepseek → 说"你好"得到语音回复；不设 key → 启动 warning、请求 401 → 泛化提示。
4. **openai + 本地无 key**（如 Ollama）→ 请求不带 Authorization 头、正常对话。
5. **hermes 回归（默认）**：跑 `bash scripts/start.sh` → gateway 自动确保、唤醒"我在"、对话/打断/跟随时窗与现状一致。
6. **旧配置兼容**：临时把 config 的 `backend:` 块删掉 → 仍按顶层 `hermes_url`/`hermes_api_key` 工作（见 Task 4 Step 9 已部分覆盖）。
7. **`wake_greeting: ""`** → 唤醒无 TTS 只提示音。
8. **任意 CWD**：项目根之外执行 `.venv/bin/python -m voice.main` → config/log 正常找到。
