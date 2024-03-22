from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_PATH = Path(__file__).parent.parent
LOG_PATH = PROJECT_PATH / "logs"

if not LOG_PATH.exists():
    LOG_PATH.mkdir()


# Define a class for configuration using Pydantic
class ProjectConfiguration(BaseModel):
    project_path: str = Field(default=str(PROJECT_PATH), env="PROJECT_PATH")
    project_name: str = Field(default="langchain_storm", env="PROJECT_NAME")
    project_version: str = Field(default="0.1.0", env="PROJECT_VERSION")
    log_path: str = Field(default=str(LOG_PATH), env="LOG_PATH")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    @field_validator("log_level", mode="before")
    @classmethod
    def check_log_level(cls, value):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]
        if value not in valid_levels:
            raise ValueError(
                f"Invalid log level: {value}. Valid options are: {valid_levels}"
            )
        return value

    @classmethod
    def load_config(cls, file_path: Path):
        if file_path.exists():
            with open(file_path, "r") as file:
                return cls(**yaml.safe_load(file))
        else:
            return cls()

    @classmethod
    def create_default_config_file(cls, file_path: Path):
        if not file_path.exists():
            default_config = cls().dict()
            with open(file_path, "w") as file:
                yaml.dump(default_config, file, default_flow_style=False)


# Define the path to the configuration file
config_file_path = PROJECT_PATH / "config.yaml"

# Write the default YAML file if it does not exist
ProjectConfiguration.create_default_config_file(config_file_path)

# Load the configuration
config = ProjectConfiguration.load_config(config_file_path)
