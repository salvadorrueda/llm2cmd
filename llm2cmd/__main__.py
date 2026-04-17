from __future__ import annotations

import argparse
import sys

from .config import Config, TOOL_MODES
from .repl import Repl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm2cmd",
        description="Pont entre llenguatge natural i comandes del sistema via Ollama.",
    )
    defaults = Config.from_env()
    parser.add_argument("--model", default=defaults.model, help=f"Model d'Ollama (default: {defaults.model})")
    parser.add_argument("--host", default=defaults.host, help=f"URL del servidor Ollama (default: {defaults.host})")
    parser.add_argument("--timeout", type=float, default=defaults.timeout, help="Timeout per comanda en segons")
    parser.add_argument(
        "--max-output",
        type=int,
        default=defaults.max_output_chars,
        help="Caràcters màxims de stdout/stderr tornats al model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help="Temperatura del model",
    )
    parser.add_argument(
        "--tool-mode",
        choices=TOOL_MODES,
        default=defaults.tool_mode,
        help="Mecanisme de crida del tool: auto (detecta pel model), tools (natiu Ollama), json (fallback)",
    )
    args = parser.parse_args(argv)

    config = Config(
        model=args.model,
        host=args.host,
        timeout=args.timeout,
        max_output_chars=args.max_output,
        temperature=args.temperature,
        tool_mode=args.tool_mode,
    )
    Repl(config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
