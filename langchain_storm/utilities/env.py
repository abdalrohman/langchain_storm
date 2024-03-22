import logging
import os
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """
    Example usage:
        from langchain_storm.env import EnvironmentLoader

        env_loader = EnvironmentLoader(required_vars=["FIREWORKS_API_KEY"])
        env_loader.load_envs()
    """

    def __init__(
        self, env_file_path: Union[str, Path] = ".env", required_vars: List[str] = None
    ):
        self.env_file_path = Path(env_file_path)
        self.required_vars = required_vars

    def load_envs(self) -> None:
        """
        Load environment variables from a file.
        :return: None
        """
        logger.info(f"Loading environment variables from {self.env_file_path}...")

        if not self.env_file_path.exists():
            raise FileNotFoundError(f"Failed to load .env file at {self.env_file_path}")

        load_dotenv(dotenv_path=self.env_file_path, verbose=False, override=True)

        if self.required_vars:
            self._check_required_vars()

    def _check_required_vars(self):
        missing_vars = [var for var in self.required_vars if var not in os.environ]
        if missing_vars:
            raise ValueError(
                f"Required environment variables not found: {missing_vars}"
            )
        invalid_vars = [
            var for var in self.required_vars if "your_key" in os.environ.get(var)
        ]
        if invalid_vars:
            raise ValueError(
                f"Add your key to the environmental variable: {invalid_vars}"
            )
