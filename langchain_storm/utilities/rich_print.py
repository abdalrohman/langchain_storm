import sys
from threading import local
from typing import Any

from rich.console import Console

_thread_locals = local()


def get_console():
    if not hasattr(_thread_locals, "console"):
        _thread_locals.console = Console()
    return _thread_locals.console


def print(*args: Any, **kwargs: Any):
    # ensure running builtin print inside jupyter notebook
    if "ipykernel" in sys.modules:
        __builtins__.print(*args, **kwargs)
    else:
        kwargs["style"] = "bold cyan"
        console = get_console()
        console.print(*args, **kwargs)
