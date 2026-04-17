from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from .config import Config
from .executor import ExecutionResult, run
from rich.table import Table

from .ollama_client import OllamaClient, OllamaError
from .tools import validate_run_shell_args


HELP_TEXT = """**Comandes disponibles**

- `/help` — mostra aquesta ajuda
- `/clear` — esborra l'historial de conversa
- `/model <nom>` — canvia el model d'Ollama en calent (sense arg mostra l'actual)
- `/models` — llista els models disponibles al servidor Ollama
- `/history` — mostra els missatges de la conversa actual
- `/exit` o `/quit` — surt (Ctrl-D també funciona)

**Execució directa**

- `! <comanda>` — executa la comanda al shell sense passar-la pel LLM.
  El resultat queda registrat a l'historial perquè el model el pugui veure.

Escriu qualsevol altra cosa per demanar una acció al sistema."""


class Repl:
    def __init__(self, config: Config, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.client = OllamaClient(config)
        self.messages: list[dict[str, Any]] = [self.client.system_message()]

    # ---- Public loop -----------------------------------------------------

    def run(self) -> None:
        self._print_banner()
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]llm2cmd[/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_meta(user_input):
                    continue
                return  # /exit returns False which means: stop

            if user_input.startswith("!"):
                self._execute_direct(user_input[1:])
                continue

            self.messages.append({"role": "user", "content": user_input})
            self._process_assistant_turn()

    # ---- Meta commands ---------------------------------------------------

    def _handle_meta(self, line: str) -> bool:
        """Return True to keep looping, False to exit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            self.console.print(Markdown(HELP_TEXT))
            return True
        if cmd == "/clear":
            self.messages = [self.client.system_message()]
            self.console.print("[dim]Historial esborrat.[/dim]")
            return True
        if cmd == "/model":
            if not arg:
                self.console.print(
                    f"[dim]Model actual:[/dim] [bold]{self.config.model}[/bold] "
                    f"[dim](mode:[/dim] {self.client.mode}[dim])[/dim]"
                )
            else:
                new_model = arg.strip()
                self.client.set_model(new_model)
                self.messages = [self.client.system_message()]
                self.console.print(
                    f"[dim]Model canviat a:[/dim] [bold]{self.config.model}[/bold] "
                    f"[dim](mode:[/dim] {self.client.mode}[dim]). Historial reiniciat.[/dim]"
                )
                self._maybe_warn_json_mode()
            return True
        if cmd == "/models":
            self._print_models()
            return True
        if cmd == "/history":
            self._print_history()
            return True
        self.console.print(f"[yellow]Comanda desconeguda:[/yellow] {cmd}. Prova /help.")
        return True

    def _print_models(self) -> None:
        try:
            names = self.client.list_models()
        except OllamaError as exc:
            self.console.print(f"[red]{exc}[/red]")
            return
        if not names:
            self.console.print("[dim]No hi ha cap model instal·lat a Ollama. Prova `ollama pull <nom>`.[/dim]")
            return
        table = Table(title=f"Models a {self.config.host}")
        table.add_column("", width=2)
        table.add_column("model")
        for name in names:
            marker = "[green]*[/green]" if name == self.config.model else ""
            table.add_row(marker, name)
        self.console.print(table)
        self.console.print("[dim]Canvia amb `/model <nom>`.[/dim]")

    def _print_history(self) -> None:
        for i, msg in enumerate(self.messages):
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            preview = content if len(content) < 200 else content[:200] + "..."
            self.console.print(f"[dim][{i}] {role}[/dim] {preview}")
            if msg.get("tool_calls"):
                self.console.print(f"[dim]    tool_calls: {len(msg['tool_calls'])}[/dim]")

    # ---- Direct execution ------------------------------------------------

    def _execute_direct(self, raw: str) -> None:
        command = raw.strip()
        if not command:
            self.console.print("[dim]Ús: `! <comanda>` per executar directament.[/dim]")
            return
        self._print_command_proposal(command, "execució directa (sense LLM)")
        result = run(command, self.config.timeout, self.config.max_output_chars)
        self._print_result(result)
        payload = json.dumps(result.to_dict(), ensure_ascii=False)
        self.messages.append({
            "role": "user",
            "content": f"(Comanda executada directament amb `!`: `{command}`)\n[RESULTAT]\n{payload}",
        })

    # ---- Assistant turn --------------------------------------------------

    def _process_assistant_turn(self) -> None:
        while True:
            try:
                with self.console.status("[dim]Pensant...[/dim]", spinner="dots"):
                    message = self.client.chat(self.messages)
            except OllamaError as exc:
                self.console.print(f"[red]{exc}[/red]")
                return

            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = (message.get("content") or "").strip()
                if content:
                    self.console.print(Panel(Markdown(content), title="assistent", border_style="green"))
                return

            all_done = True
            for call in tool_calls:
                handled = self._handle_tool_call(call)
                if not handled:
                    all_done = False
                    break

            if not all_done:
                return
            # Si hi ha hagut tool calls, tornem a demanar al model perquè elabori la resposta final.

    def _handle_tool_call(self, call: dict[str, Any]) -> bool:
        fn = call.get("function") or {}
        name = fn.get("name")
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args or {}

        if name != "run_shell_command":
            self._append_tool_result(call, {"error": f"Tool desconegut: {name}"})
            return True

        try:
            command, explanation = validate_run_shell_args(args)
        except ValueError as exc:
            self._append_tool_result(call, {"error": str(exc)})
            return True

        self._print_command_proposal(command, explanation)
        choice = self._ask_confirmation()
        if choice == "n":
            self._append_tool_result(
                call, {"status": "rejected", "message": "L'usuari ha rebutjat l'execució."}
            )
            return True
        if choice == "e":
            edited = Prompt.ask("[yellow]Comanda editada[/yellow]", default=command).strip()
            if not edited:
                self._append_tool_result(
                    call, {"status": "rejected", "message": "Edició cancel·lada."}
                )
                return True
            command = edited

        result = run(command, self.config.timeout, self.config.max_output_chars)
        self._print_result(result)
        self._append_tool_result(call, result.to_dict())
        return True

    # ---- UI helpers ------------------------------------------------------

    def _print_command_proposal(self, command: str, explanation: str) -> None:
        self.console.print(
            Panel(
                Syntax(command, "bash", theme="ansi_dark", word_wrap=True),
                title="comanda proposada",
                border_style="yellow",
            )
        )
        if explanation:
            self.console.print(f"[dim]{explanation}[/dim]")

    def _ask_confirmation(self) -> str:
        choice = Prompt.ask(
            "[bold]Executar?[/bold]", choices=["y", "n", "e"], default="n"
        )
        return choice

    def _print_result(self, result: ExecutionResult) -> None:
        title = f"exit={result.returncode}"
        if result.timed_out:
            title += " (timeout)"
        border = "green" if result.returncode == 0 and not result.error else "red"
        body_parts: list[str] = []
        if result.error:
            body_parts.append(f"[red]{result.error}[/red]")
        if result.stdout:
            body_parts.append(result.stdout.rstrip())
        if result.stderr:
            body_parts.append(f"[dim]stderr:[/dim]\n{result.stderr.rstrip()}")
        body = "\n\n".join(body_parts) if body_parts else "[dim](sense sortida)[/dim]"
        self.console.print(Panel(body, title=title, border_style=border))

    def _append_tool_result(self, call: dict[str, Any], result: dict[str, Any]) -> None:
        msg: dict[str, Any] = {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
        }
        tool_call_id = call.get("id")
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self.messages.append(msg)

    def _print_banner(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold]llm2cmd[/bold] — model: [cyan]{self.config.model}[/cyan]  "
                f"host: [cyan]{self.config.host}[/cyan]  mode: [cyan]{self.client.mode}[/cyan]\n"
                "Escriu la teva petició en llenguatge natural, `! <cmd>` per executar directe, "
                "/help per ajuda, /exit per sortir.",
                border_style="magenta",
            )
        )
        self._maybe_warn_json_mode()

    def _maybe_warn_json_mode(self) -> None:
        if self.client.mode != "json":
            return
        self.console.print(
            Panel(
                f"El model [bold]{self.config.model}[/bold] no té suport conegut de "
                "tool calling natiu a Ollama. S'utilitzarà el [bold]fallback JSON[/bold]: "
                "el model ha de retornar estrictament un objecte JSON amb la comanda.\n\n"
                "[dim]Pots forçar el mode natiu amb [bold]--tool-mode tools[/bold] "
                "(pot fallar si el model realment no ho suporta), o canviar a un model "
                "compatible (p.ex. llama3.1, qwen2.5, mistral-nemo).[/dim]",
                title="⚠ mode fallback",
                border_style="yellow",
            )
        )
