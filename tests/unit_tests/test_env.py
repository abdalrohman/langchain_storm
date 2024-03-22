import os

import pytest

from langchain_storm.utilities.env import EnvironmentLoader


# Test to verify that the .env file exists
def test_env_file_exists():
    env_loader = EnvironmentLoader()
    assert (
        env_loader.env_file_path.exists()
    ), f"Failed to find .env file at {env_loader.env_file_path}"


# Test to check that all required environment variables are present
def test_required_vars_present():
    required_vars = ["FIREWORKS_API_KEY", "ANTHROPIC_API_KEY"]
    env_loader = EnvironmentLoader(required_vars=required_vars)
    env_loader.load_envs()
    for var in required_vars:
        assert var in os.environ, f"Required environment variable {var} not found"


# Test to ensure that an exception is raised if any required variables are missing
def test_missing_required_vars():
    required_vars = ["FIREWORKS_API_KEY", "MISSING_VAR"]
    env_loader = EnvironmentLoader(required_vars=required_vars)
    with pytest.raises(ValueError) as excinfo:
        env_loader.load_envs()
    assert "Required environment variables not found" in str(excinfo.value)
