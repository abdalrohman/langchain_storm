import pytest

from langchain_storm.config import PROJECT_PATH, ProjectConfiguration

# Constants for tests
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]
INVALID_LOG_LEVEL = "NOTALOGLEVEL"
EXISTING_FILE_PATH = PROJECT_PATH / "config.yaml"
NON_EXISTING_FILE_PATH = PROJECT_PATH / "non_existing_config.yaml"


def test_check_log_level_valid():
    for level in VALID_LOG_LEVELS:
        assert ProjectConfiguration.check_log_level(level) == level


def test_check_log_level_invalid():
    with pytest.raises(ValueError) as exc_info:
        ProjectConfiguration.check_log_level(INVALID_LOG_LEVEL)
    assert (
        str(exc_info.value)
        == f"Invalid log level: NOTALOGLEVEL. Valid options are: {VALID_LOG_LEVELS}"
    )


def test_load_config_existing():
    config = ProjectConfiguration.load_config(EXISTING_FILE_PATH)
    assert isinstance(config, ProjectConfiguration)


def test_load_config_non_existing():
    config = ProjectConfiguration.load_config(NON_EXISTING_FILE_PATH)
    assert (
        isinstance(config, ProjectConfiguration)
        and config.project_name == "langchain_storm"
    )


def test_create_default_config_file_non_existing(tmp_path):
    non_existing_file_path = tmp_path / "non_existing_config.yaml"
    ProjectConfiguration.create_default_config_file(non_existing_file_path)
    assert non_existing_file_path.exists()
