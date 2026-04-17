import json

from llm2cmd.ollama_client import (
    parse_json_response,
    resolve_tool_mode,
    supports_tool_calling,
    _transform_messages_for_json,
)


def test_supports_tool_calling_known_models():
    assert supports_tool_calling("llama3.1:8b") is True
    assert supports_tool_calling("llama3.2:3b") is True
    assert supports_tool_calling("qwen2.5:7b") is True
    assert supports_tool_calling("mistral-nemo:latest") is True


def test_supports_tool_calling_unknown_models():
    assert supports_tool_calling("gemma4:latest") is False
    assert supports_tool_calling("gemma3:4b") is False
    assert supports_tool_calling("phi3:mini") is False
    assert supports_tool_calling("llama2:7b") is False


def test_resolve_tool_mode_auto():
    assert resolve_tool_mode("llama3.1:8b", "auto") == "tools"
    assert resolve_tool_mode("gemma4:latest", "auto") == "json"


def test_resolve_tool_mode_override():
    assert resolve_tool_mode("gemma4:latest", "tools") == "tools"
    assert resolve_tool_mode("llama3.1:8b", "json") == "json"


def test_parse_json_response_run_action():
    raw = json.dumps({"action": "run", "command": "ls -la", "explanation": "llista"})
    msg = parse_json_response(raw)
    assert msg["role"] == "assistant"
    assert msg["tool_calls"]
    call = msg["tool_calls"][0]
    assert call["function"]["name"] == "run_shell_command"
    assert call["function"]["arguments"]["command"] == "ls -la"
    assert call["function"]["arguments"]["explanation"] == "llista"


def test_parse_json_response_reply_action():
    raw = json.dumps({"action": "reply", "content": "hola"})
    msg = parse_json_response(raw)
    assert msg["content"] == "hola"
    assert "tool_calls" not in msg


def test_parse_json_response_with_surrounding_garbage():
    raw = "Aquí va el JSON: " + json.dumps(
        {"action": "run", "command": "pwd", "explanation": "x"}
    ) + "\n(fi)"
    msg = parse_json_response(raw)
    assert msg["tool_calls"]
    assert msg["tool_calls"][0]["function"]["arguments"]["command"] == "pwd"


def test_parse_json_response_invalid_falls_back_to_text():
    msg = parse_json_response("això no és JSON de cap manera")
    assert "tool_calls" not in msg
    assert msg["content"] == "això no és JSON de cap manera"


def test_parse_json_response_empty():
    msg = parse_json_response("")
    assert msg == {"role": "assistant", "content": ""}


def test_transform_messages_tool_becomes_user():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "fes X"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_shell_command", "arguments": {"command": "ls", "explanation": "y"}}}
            ],
        },
        {"role": "tool", "content": '{"stdout": "a.txt"}'},
    ]
    out = _transform_messages_for_json(msgs)
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert out[2]["role"] == "assistant"
    # Assistant sense tool_calls, content reconstruït amb JSON.
    assert "tool_calls" not in out[2]
    assert "ls" in out[2]["content"]
    assert out[3]["role"] == "user"
    assert out[3]["content"].startswith("[RESULTAT]")
