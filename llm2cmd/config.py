from __future__ import annotations

import os
from dataclasses import dataclass


TOOL_MODES = ("auto", "tools", "json")


@dataclass
class Config:
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    timeout: float = 30.0
    max_output_chars: int = 4000
    temperature: float = 0.2
    tool_mode: str = "auto"  # auto | tools | json

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model=os.environ.get("LLM2CMD_MODEL", cls.model),
            host=os.environ.get("OLLAMA_HOST", cls.host),
            timeout=float(os.environ.get("LLM2CMD_TIMEOUT", cls.timeout)),
            max_output_chars=int(
                os.environ.get("LLM2CMD_MAX_OUTPUT", cls.max_output_chars)
            ),
            temperature=float(
                os.environ.get("LLM2CMD_TEMPERATURE", cls.temperature)
            ),
            tool_mode=os.environ.get("LLM2CMD_TOOL_MODE", cls.tool_mode),
        )
