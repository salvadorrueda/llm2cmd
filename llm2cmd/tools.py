from __future__ import annotations

RUN_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell_command",
        "description": (
            "Executa una comanda shell POSIX al sistema Linux de l'usuari. "
            "L'usuari sempre confirmarà abans d'executar. "
            "Retorna stdout, stderr i el codi de sortida."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "La comanda shell exacta a executar.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Explicació breu en llenguatge natural del què fa la comanda i per què l'has escollit.",
                },
            },
            "required": ["command", "explanation"],
        },
    },
}


TOOLS = [RUN_SHELL_TOOL]


def validate_run_shell_args(args: dict) -> tuple[str, str]:
    if not isinstance(args, dict):
        raise ValueError("Els arguments del tool han de ser un objecte JSON.")
    command = args.get("command")
    explanation = args.get("explanation", "")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("L'argument 'command' és obligatori i ha de ser un string no buit.")
    if not isinstance(explanation, str):
        raise ValueError("L'argument 'explanation' ha de ser un string.")
    return command.strip(), explanation.strip()
