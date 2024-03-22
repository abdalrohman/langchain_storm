# test_rich_logger.py

import logging
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from langchain_storm.utilities.log import Log


def test_validate_log_file_path_str():
    log = Log(log_file_path="test.log")
    assert log.log_file_path == Path("test.log")


def test_validate_log_file_path_stringio():
    stringio = StringIO()
    log = Log(log_file_path=stringio)
    assert log.log_file_path == Path(stringio.getvalue())


def test_get_logger_cleans_file():
    with patch.object(Path, "exists", return_value=True), patch.object(
        Path, "unlink"
    ) as mock_unlink, patch("logging.FileHandler") as mock_file_handler:
        log = Log(log_file_path="test.log", clean_log_file=True)
        log.get_logger()
        mock_unlink.assert_called_once()


def test_get_logger_returns_logger():
    with patch("logging.FileHandler"):
        log = Log()
        logger = log.get_logger()
        assert isinstance(logger, logging.Logger)


def test_log_level():
    log = Log(level="INFO")
    logger = log.get_logger()
    assert logger.level == logging.getLevelName("INFO")

    log = Log(level="DEBUG")
    logger = log.get_logger()
    assert logger.level == logging.getLevelName("DEBUG")

    log = Log(level="WARNING")
    logger = log.get_logger()
    assert logger.level == logging.getLevelName("WARNING")

    log = Log(level="ERROR")
    logger = log.get_logger()
    assert logger.level == logging.getLevelName("ERROR")

    log = Log(level="CRITICAL")
    logger = log.get_logger()
    assert logger.level == logging.getLevelName("CRITICAL")

    log = Log(level="NOTSET")
    logger = log.get_logger()
    assert logger.level == logging.getLevelName("NOTSET")
