from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass


@dataclass
class ExecutionResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    truncated: bool
    timed_out: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    head_len = max_chars // 2
    tail_len = max_chars - head_len - 32
    marker = f"\n...[truncated {len(text) - max_chars} chars]...\n"
    return text[:head_len] + marker + text[-tail_len:], True


def run(command: str, timeout: float, max_output_chars: int) -> ExecutionResult:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_t, t1 = _truncate(stdout, max_output_chars)
        stderr_t, t2 = _truncate(stderr, max_output_chars)
        return ExecutionResult(
            command=command,
            returncode=-1,
            stdout=stdout_t,
            stderr=stderr_t,
            truncated=t1 or t2,
            timed_out=True,
            error=f"Comanda cancel·lada després de {timeout}s",
        )
    except Exception as exc:
        return ExecutionResult(
            command=command,
            returncode=-1,
            stdout="",
            stderr="",
            truncated=False,
            timed_out=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    stdout_t, t1 = _truncate(proc.stdout or "", max_output_chars)
    stderr_t, t2 = _truncate(proc.stderr or "", max_output_chars)
    return ExecutionResult(
        command=command,
        returncode=proc.returncode,
        stdout=stdout_t,
        stderr=stderr_t,
        truncated=t1 or t2,
        timed_out=False,
    )
