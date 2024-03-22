from typing import Iterator, List, Union

from rich.console import Console

console = Console()


def get_user_input(prompt: str) -> str:
    return console.input(prompt)


def print_markdown(msg: Union[List[str], str, Iterator[str]]):
    from rich.markdown import Markdown

    if isinstance(msg, str):
        markdown = Markdown(msg)
        console.print(markdown, style="bold cyan")
    elif isinstance(msg, list):
        markdown = Markdown("\n".join(msg))
        console.print(markdown, style="bold cyan")
    elif isinstance(msg, Iterator):
        for token in msg:
            # markdown = Markdown(token)
            console.print(token, style="bold cyan", end="")
    else:
        raise TypeError("msg must be a string, a list of strings, or an Iterator")
