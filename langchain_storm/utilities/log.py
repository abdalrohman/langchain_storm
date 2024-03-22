import io
import logging
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.logging import RichHandler


class CustomFilters(logging.Filter):
    def __init__(self, filters: Union[str | List[str]]):
        super().__init__()
        if isinstance(filters, str):
            filters = [filters]
        if isinstance(filters, list):
            filters = filters
        if not all(isinstance(f, str) for f in filters):
            raise TypeError("'filters' must be a string or list of strings")
        self.filters = filters

    def filter(self, record):
        # return all(f not in record.getMessage() for f in self.filters)
        message = record.getMessage()
        return any(c in message for c in self.filters)


class Log(BaseModel):
    """
    A class used to initialize the log file.
    Attributes:
        log_file_path (Path): The path to the log file.

    Log Example:
        from langchain_storm.log import Log

        log = Log(log_file_path='test.log', level='DEBUG')
        logger = log.get_logger()
        logger.debug('This is a debug message')
    """

    log_file_path: Union[Path, io.StringIO, str] = Field(
        default=Path("logging.log"), description="The file to write to."
    )
    level: str = Field(default="INFO", description="The log level.")
    logger_name: str = Field(default="__file__", description="The name of the logger.")
    # filters: List[str] = Field(default=[], description="The list of filters to skip from recording.")
    filters: Union[str | List[str]] = Field(
        default=["watchdog.observers.inotify_buffer"],
        description="The list of filters to skip from recording.",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s",
        description="The log format.",
    )
    datefmt: str = Field(default="[%X]", description="The date format.")
    clean_log_file: bool = Field(
        default=False, description="Whether to clean the log file."
    )

    @field_validator("level", mode="before")
    @classmethod
    def check_log_level(cls, value):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]
        if value not in valid_levels:
            raise ValueError(
                f"Invalid log level: {value}. Valid options are: {valid_levels}"
            )
        return value

    @field_validator("log_file_path", mode="before")
    @classmethod
    def validate_log_file_path(cls, v: Union[Path, io.StringIO, str]) -> Path:
        if isinstance(v, (str, io.StringIO)):
            return Path(v.getvalue() if isinstance(v, io.StringIO) else v)

        return v

    @property
    def console(self) -> Console:
        return Console(
            force_terminal=True,
            color_system="auto",
        )

    @property
    def handler(self) -> RichHandler:
        return RichHandler(
            console=self.console,
            enable_link_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )

    def get_logger(self) -> logging.Logger:
        if self.clean_log_file and self.log_file_path.exists():  # type: ignore
            self.log_file_path.unlink()  # type: ignore
        file_handler = logging.FileHandler(filename=self.log_file_path)  # type: ignore
        file_handler.setFormatter(logging.Formatter(self.format, self.datefmt))
        logging.basicConfig(
            level=self.level,
            format="%(name)s - %(message)s",
            datefmt=self.datefmt,
            handlers=[self.handler, file_handler],
        )
        logger = logging.getLogger(self.logger_name)

        # Set the logger's level
        logger.setLevel(self.level)

        # TODO fix adding filters to the logger
        logger.addFilter(CustomFilters(self.filters))

        # remove existing handler
        logger.handlers.clear()

        return logger

    class Config:
        arbitrary_types_allowed = True
