# llm2cmd

Pont entre llenguatge natural i comandes del sistema mitjançant [Ollama](https://ollama.com).

Escrius una petició en llenguatge natural (català, castellà o anglès) i un LLM local proposa la comanda shell equivalent. Tu confirmes abans d'executar.

## Requisits

- Python 3.10+
- [Ollama](https://ollama.com/download) instal·lat i en execució
- Un model amb suport de *tool calling* (per defecte `llama3.1:8b`)

```bash
ollama pull llama3.1:8b
```

## Instal·lació

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Ús

Llança el REPL:

```bash
python -m llm2cmd
# o si has instal·lat el paquet:
llm2cmd
```

Exemple de sessió:

```
llm2cmd> mostra el directori actual
╭── comanda proposada ────────╮
│ pwd                         │
╰─────────────────────────────╯
Retorna la ruta absoluta del directori de treball.
Executar? [y/n/e] (n): y
╭── exit=0 ───────────────────╮
│ /home/user/project          │
╰─────────────────────────────╯

llm2cmd> ara llista els 3 fitxers més grans d'aquí
...
```

A la confirmació:
- `y` — executa la comanda proposada
- `n` — rebutja (el model rep la negativa i pot proposar una alternativa)
- `e` — edita la comanda abans d'executar

## Meta-comandes

- `/help` — mostra l'ajuda
- `/clear` — esborra l'historial de la conversa
- `/model <nom>` — canvia el model en calent
- `/history` — mostra els missatges de la sessió
- `/exit` o `Ctrl-D` — sortir

## Configuració

Paràmetres per CLI o variables d'entorn:

| Flag              | Env var               | Default                   |
|-------------------|-----------------------|---------------------------|
| `--model`         | `LLM2CMD_MODEL`       | `llama3.1:8b`             |
| `--host`          | `OLLAMA_HOST`         | `http://localhost:11434`  |
| `--timeout`       | `LLM2CMD_TIMEOUT`     | `30` (segons)             |
| `--max-output`    | `LLM2CMD_MAX_OUTPUT`  | `4000` (caràcters)        |
| `--temperature`   | `LLM2CMD_TEMPERATURE` | `0.2`                     |
| `--tool-mode`     | `LLM2CMD_TOOL_MODE`   | `auto` (`auto`/`tools`/`json`) |

## Models i tool calling

L'app prefereix el **tool calling natiu** d'Ollama (més fiable). Si el model triat no té suport conegut (p.ex. `gemma*`, `llama2`, `phi3`), cau automàticament al mode **fallback JSON**: el model ha de retornar un objecte JSON estricte que l'app interpreta.

- `--tool-mode auto` (default): detecta pel nom del model.
- `--tool-mode tools`: força el mode natiu (pot fallar si el model no ho suporta).
- `--tool-mode json`: força el fallback.

Models amb tool calling natiu coneguts: `llama3.1`, `llama3.2`, `llama3.3`, `qwen2.5`, `qwen3`, `mistral-nemo`, `mistral-small/large`, `mixtral`, `command-r`, `firefunction`, `hermes3`, `granite3`, `nemotron`.

## Tests

```bash
pip install -e '.[dev]'
pytest
```

## Seguretat

Cada comanda requereix confirmació humana explícita abans d'executar-se. No hi ha sandboxing: la comanda s'executa amb els privilegis de l'usuari que llança `llm2cmd`. Revisa sempre la comanda proposada abans de prémer `y`.
