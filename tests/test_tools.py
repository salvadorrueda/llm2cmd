import pytest

from llm2cmd.tools import RUN_SHELL_TOOL, TOOLS, validate_run_shell_args


def test_schema_shape():
    assert RUN_SHELL_TOOL["type"] == "function"
    fn = RUN_SHELL_TOOL["function"]
    assert fn["name"] == "run_shell_command"
    props = fn["parameters"]["properties"]
    assert "command" in props and "explanation" in props
    assert fn["parameters"]["required"] == ["command", "explanation"]


def test_tools_list_contains_run_shell():
    assert RUN_SHELL_TOOL in TOOLS


def test_validate_ok():
    cmd, exp = validate_run_shell_args({"command": " ls -la ", "explanation": "Llista fitxers"})
    assert cmd == "ls -la"
    assert exp == "Llista fitxers"


def test_validate_missing_command():
    with pytest.raises(ValueError):
        validate_run_shell_args({"explanation": "x"})


def test_validate_empty_command():
    with pytest.raises(ValueError):
        validate_run_shell_args({"command": "   ", "explanation": "x"})


def test_validate_non_dict():
    with pytest.raises(ValueError):
        validate_run_shell_args("ls -la")  # type: ignore[arg-type]


def test_validate_missing_explanation_is_ok():
    cmd, exp = validate_run_shell_args({"command": "ls"})
    assert cmd == "ls"
    assert exp == ""
