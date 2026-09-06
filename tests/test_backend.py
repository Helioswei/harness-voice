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
