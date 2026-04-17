from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import ollama

from .config import Config
from .tools import TOOLS


SYSTEM_PROMPT_TOOLS = """Ets llm2cmd, un assistent que tradueix peticions en llenguatge natural (català, castellà o anglès) a comandes shell POSIX per a sistemes Linux.

Regles:
- Per executar qualsevol acció al sistema de l'usuari, usa SEMPRE el tool `run_shell_command`. Mai no descriguis una comanda sense cridar el tool.
- Proposa una sola comanda per torn. Si la tasca requereix diversos passos, executa'ls un per un i espera el resultat abans del següent.
- Prefereix comandes POSIX estàndard (coreutils, find, grep, awk, sed) abans que eines específiques de distribució.
- Quan necessitis paràmetres que no et dona l'usuari, demana-los en llenguatge natural en lloc d'inventar-los.
- Si l'usuari només fa una pregunta conceptual (sense acció), respon amb text sense cridar cap tool.
- Quan rebis el resultat d'un tool, resumeix-lo breument en llenguatge natural per a l'usuari.
- Mai escriguis comandes destructives (rm -rf /, dd, mkfs) sense preguntar confirmació explícita a l'usuari primer.
"""


SYSTEM_PROMPT_JSON = """Ets llm2cmd, un assistent que tradueix peticions en llenguatge natural (català, castellà o anglès) a comandes shell POSIX per a Linux.

IMPORTANT: Has de respondre SEMPRE amb un únic objecte JSON vàlid i res més. Sense text abans ni després, sense backticks, sense ```json, sense comentaris.

Dues formes possibles:

(A) Per executar una comanda shell:
{"action": "run", "command": "<comanda exacta>", "explanation": "<explicació breu>"}

(B) Per respondre text a l'usuari (preguntes conceptuals, resums de resultats, aclariments):
{"action": "reply", "content": "<el teu missatge en text pla>"}

Regles:
- Una sola acció per resposta.
- Si un missatge de l'usuari comença amb "[RESULTAT]" és la sortida (JSON amb stdout/stderr) d'una comanda executada; resumeix-la amb action="reply".
- Proposa una sola comanda per torn; si la tasca en requereix diverses, emet-les una a una esperant el resultat.
- Prefereix comandes POSIX (coreutils, find, grep, awk, sed).
- Mai escriguis comandes destructives (rm -rf /, dd, mkfs) sense demanar confirmació amb action="reply" primer.
"""


# Famílies de models conegudes amb suport de tool calling natiu a Ollama.
# Llista conservadora: qualsevol model fora d'aquí cau al fallback JSON.
_TOOL_CAPABLE_PREFIXES = (
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "llama4",
    "qwen2.5",
    "qwen3",
    "mistral-nemo",
    "mistral-small",
    "mistral-large",
    "mixtral",
    "command-r",
    "firefunction",
    "hermes3",
    "granite3",
    "nemotron",
)


def supports_tool_calling(model: str) -> bool:
    base = model.lower().split(":", 1)[0]
    return any(base.startswith(p) for p in _TOOL_CAPABLE_PREFIXES)


def resolve_tool_mode(model: str, override: str) -> str:
    if override == "tools":
        return "tools"
    if override == "json":
        return "json"
    # auto
    return "tools" if supports_tool_calling(model) else "json"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = ollama.Client(host=config.host)
        self._mode = resolve_tool_mode(config.model, config.tool_mode)
        self.last_timing: dict[str, Any] | None = None

    @property
    def mode(self) -> str:
        return self._mode

    def set_model(self, model: str) -> str:
        self.config.model = model
        self._mode = resolve_tool_mode(model, self.config.tool_mode)
        return self._mode

    def system_message(self) -> dict[str, str]:
        prompt = SYSTEM_PROMPT_TOOLS if self._mode == "tools" else SYSTEM_PROMPT_JSON
        return {"role": "system", "content": prompt}

    def list_models(self) -> list[str]:
        try:
            response = self._client.list()
        except Exception as exc:
            raise OllamaError(
                f"No s'han pogut llistar els models a {self.config.host}: {exc}"
            ) from exc
        models = response.get("models") if isinstance(response, dict) else getattr(response, "models", [])
        names: list[str] = []
        for m in models or []:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if not name:
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
            if name:
                names.append(name)
        return sorted(names)

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self._mode == "tools":
            return self._chat_tools(messages)
        return self._chat_json(messages)

    # ---- Native tool-calling path ---------------------------------------

    def _chat_tools(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                tools=TOOLS,
                options={"temperature": self.config.temperature},
            )
        except ollama.ResponseError as exc:
            self.last_timing = {"elapsed_s": time.perf_counter() - t0}
            raise OllamaError(f"Ollama ha retornat error: {exc}") from exc
        except Exception as exc:
            self.last_timing = {"elapsed_s": time.perf_counter() - t0}
            raise OllamaError(
                f"No s'ha pogut connectar amb Ollama a {self.config.host}: {exc}"
            ) from exc

        self.last_timing = extract_timing(response, time.perf_counter() - t0)
        message = response.get("message") if isinstance(response, dict) else response.message
        if message is None:
            raise OllamaError("Resposta d'Ollama sense camp 'message'.")
        return _to_dict(message)

    # ---- JSON fallback path ---------------------------------------------

    def _chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        transformed = _transform_messages_for_json(messages)
        # Substituïm el primer missatge system pel de mode JSON (si existeix).
        if transformed and transformed[0].get("role") == "system":
            transformed[0] = {"role": "system", "content": SYSTEM_PROMPT_JSON}
        else:
            transformed.insert(0, {"role": "system", "content": SYSTEM_PROMPT_JSON})

        t0 = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.config.model,
                messages=transformed,
                options={"temperature": self.config.temperature},
                format="json",
            )
        except ollama.ResponseError as exc:
            self.last_timing = {"elapsed_s": time.perf_counter() - t0}
            raise OllamaError(f"Ollama ha retornat error: {exc}") from exc
        except Exception as exc:
            self.last_timing = {"elapsed_s": time.perf_counter() - t0}
            raise OllamaError(
                f"No s'ha pogut connectar amb Ollama a {self.config.host}: {exc}"
            ) from exc

        self.last_timing = extract_timing(response, time.perf_counter() - t0)
        message = response.get("message") if isinstance(response, dict) else response.message
        if message is None:
            raise OllamaError("Resposta d'Ollama sense camp 'message'.")
        msg = _to_dict(message)
        raw_content = (msg.get("content") or "").strip()
        return parse_json_response(raw_content)


# ---- Helpers -----------------------------------------------------------


def extract_timing(response: Any, elapsed_s: float) -> dict[str, Any]:
    """Extreu els temps i recompte de tokens reportats per Ollama, si existeixen.
    `total_duration` i companys venen en nanosegons; els convertim a segons."""

    def get(key: str) -> Any:
        if isinstance(response, dict):
            return response.get(key)
        return getattr(response, key, None)

    def ns_to_s(val: Any) -> float | None:
        if val is None:
            return None
        try:
            return float(val) / 1e9
        except (TypeError, ValueError):
            return None

    eval_s = ns_to_s(get("eval_duration"))
    gen_tokens = get("eval_count")
    tps: float | None = None
    if gen_tokens and eval_s and eval_s > 0:
        try:
            tps = float(gen_tokens) / eval_s
        except (TypeError, ValueError, ZeroDivisionError):
            tps = None

    return {
        "elapsed_s": elapsed_s,
        "total_s": ns_to_s(get("total_duration")),
        "load_s": ns_to_s(get("load_duration")),
        "prompt_eval_s": ns_to_s(get("prompt_eval_duration")),
        "eval_s": eval_s,
        "prompt_tokens": get("prompt_eval_count"),
        "gen_tokens": gen_tokens,
        "tokens_per_second": tps,
    }


def _to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "dict"):
        return message.dict()
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _transform_messages_for_json(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapta l'historial (role=tool, tool_calls, etc.) a la conversa plana que
    necessita un model sense tool calling."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            out.append({
                "role": "user",
                "content": "[RESULTAT]\n" + (msg.get("content") or ""),
            })
            continue
        if role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            if not content and tool_calls:
                # Reconstruïm un JSON coherent perquè el model vegi el seu propi torn.
                call = tool_calls[0]
                fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
                args = (fn or {}).get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content = json.dumps(
                    {
                        "action": "run",
                        "command": (args or {}).get("command", ""),
                        "explanation": (args or {}).get("explanation", ""),
                    },
                    ensure_ascii=False,
                )
            out.append({"role": "assistant", "content": content})
            continue
        # system / user i qualsevol altre: passa tal qual (sense tool_calls).
        clean = {"role": role, "content": msg.get("content") or ""}
        out.append(clean)
    return out


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Converteix la resposta textual del model (JSON esperat) a l'estructura
    de missatge assistant que consumeix el REPL. Tolerant a brossa al voltant."""
    if not raw:
        return {"role": "assistant", "content": ""}

    data: Any = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(raw)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        # No s'ha pogut parsejar: tractem-ho com a text lliure.
        return {"role": "assistant", "content": raw}

    action = data.get("action")
    if action == "run" and isinstance(data.get("command"), str):
        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "function": {
                "name": "run_shell_command",
                "arguments": {
                    "command": data["command"],
                    "explanation": data.get("explanation", "") or "",
                },
            },
        }
        return {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "run",
                    "command": data["command"],
                    "explanation": data.get("explanation", "") or "",
                },
                ensure_ascii=False,
            ),
            "tool_calls": [tool_call],
        }

    if action == "reply" and isinstance(data.get("content"), str):
        return {"role": "assistant", "content": data["content"]}

    # JSON vàlid però format inesperat: retornem el contingut cru al usuari.
    return {"role": "assistant", "content": raw}
