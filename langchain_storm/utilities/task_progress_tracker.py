import queue
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner


class TaskProgressDisplay(ABC):
    @abstractmethod
    def add_output_message(self, text):
        pass

    @abstractmethod
    def show_task_results(self, function_name: str, start_time: datetime):
        pass


class ConsoleTaskProgressDisplay(TaskProgressDisplay):
    def __init__(self):
        self.console = Console()
        self.output_queue = queue.Queue()

    def add_output_message(self, text):
        self.output_queue.put(text)

    def show_task_results(self, function_name: str, start_time: datetime):
        execution_time = datetime.now() - start_time
        output = ""
        while not self.output_queue.empty():
            output += self.output_queue.get()
        if output:
            self.console.print(
                Panel(
                    f"\n[bold]Output of {function_name}:[/bold]\n{output}", expand=False
                )
            )
        self.console.print(
            f"[green]✓ Execution of {function_name} succeeded in {execution_time.total_seconds():.2f} seconds[/green]"
        )


class TaskExecutor:
    def __init__(self, progress_display: TaskProgressDisplay):
        self.progress_display = progress_display

    def execute_task(self, func, *args, **kwargs):
        function_name = func.__name__
        spinner = Spinner("dots", text=f"Executing {function_name}...")
        start_time = datetime.now()

        def target():
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.progress_display.add_output_message(str(e))

        task_thread = threading.Thread(target=target)
        task_thread.start()

        with Live(
            spinner,
            console=self.progress_display.console,
            refresh_per_second=10,
            transient=True,
        ):
            while task_thread.is_alive():
                time.sleep(0.1)
                while not self.progress_display.output_queue.empty():
                    self.progress_display.console.print(
                        Panel(self.progress_display.output_queue.get(), expand=False)
                    )

        task_thread.join()
        self.progress_display.show_task_results(function_name, start_time)


def task_execution_tracker(func):
    def wrapper(*args, **kwargs):
        progress_display = ConsoleTaskProgressDisplay()
        executor = TaskExecutor(progress_display)
        executor.execute_task(func, *args, **kwargs)

    return wrapper
