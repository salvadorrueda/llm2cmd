from llm2cmd.executor import run


def test_success_exit_code_zero():
    result = run("echo hola", timeout=5, max_output_chars=1000)
    assert result.returncode == 0
    assert "hola" in result.stdout
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.truncated is False


def test_non_zero_exit_code_captured():
    result = run("false", timeout=5, max_output_chars=1000)
    assert result.returncode != 0
    assert result.timed_out is False


def test_timeout_marks_timed_out():
    result = run("sleep 2", timeout=0.2, max_output_chars=1000)
    assert result.timed_out is True
    assert result.returncode == -1
    assert "cancel" in (result.error or "").lower()


def test_output_truncation():
    # 3000 bytes of 'x' then a newline, limit 500 -> should truncate
    result = run("python3 -c \"print('x'*3000)\"", timeout=5, max_output_chars=500)
    assert result.returncode == 0
    assert result.truncated is True
    assert "truncated" in result.stdout
    assert len(result.stdout) <= 500 + 100  # marker adds a few chars


def test_stderr_captured():
    result = run("python3 -c \"import sys; sys.stderr.write('boom')\"", timeout=5, max_output_chars=1000)
    assert result.returncode == 0
    assert "boom" in result.stderr


def test_to_dict_has_expected_keys():
    result = run("echo ok", timeout=5, max_output_chars=1000)
    d = result.to_dict()
    for key in ("command", "returncode", "stdout", "stderr", "truncated", "timed_out", "error"):
        assert key in d
