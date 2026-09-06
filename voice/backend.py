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
