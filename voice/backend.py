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
import shutil
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger("harness-voice")


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
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            self.messages.pop()
            raise ConnectionError(f"Backend request failed: {exc}") from exc
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
        key_line = f"API_SERVER_KEY={self.api_key}"
        lines = []
        found_enabled = found_key = False
        for line in text.splitlines():
            if line.startswith("API_SERVER_ENABLED="):
                lines.append("API_SERVER_ENABLED=true")
                if line != "API_SERVER_ENABLED=true":
                    changed = True
                found_enabled = True
            elif line.startswith("API_SERVER_KEY="):
                lines.append(key_line)
                if line != key_line:
                    changed = True
                found_key = True
            else:
                lines.append(line)
        if not found_enabled:
            lines.append("API_SERVER_ENABLED=true")
            changed = True
        if not found_key:
            lines.append(key_line)
            changed = True
        if changed:
            self.hermes_env.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(lines)
            if content:
                content += "\n"
            self.hermes_env.write_text(content, encoding="utf-8")
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
