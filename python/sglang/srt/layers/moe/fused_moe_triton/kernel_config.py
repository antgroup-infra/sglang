import json
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Optional

from sglang.srt.utils import get_current_device_name

logger = logging.getLogger(__name__)


class KernelConfigs(ABC):

    kernel_name: str = "unknown_kernel"

    @classmethod
    def _get_config_dir_path(cls) -> str:
        """
        Helper method to construct the directory path where kernel configurations are stored.
        """
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "all_kernel_configs",
            cls.kernel_name,
        )

    @classmethod
    def get_config_file_name(cls, params: Dict[str, Any]) -> str:
        """
        Generates a unique filename based on the given parameters and the current device name.
        """
        json_str = json.dumps(params, sort_keys=True)
        json_str = (
            json_str.replace(" ", "")
            .replace("\n", "")
            .replace('"', "")
            .replace(":", "=")
        )
        device_name = get_current_device_name().replace(" ", "_")
        return f"{json_str}_{device_name}.json"

    @classmethod
    @lru_cache(maxsize=None)
    def get_the_config(cls, params: Dict[str, Any]) -> Optional[dict]:
        """
        Retrieves the configuration from the file system based on the provided parameters.
        Returns None if the configuration file doesn't exist.
        """
        assert cls != KernelConfigs, "Base class cannot call this class method."

        json_file_name = cls.get_config_file_name(params)
        config_dir_path = cls._get_config_dir_path()
        config_file_path = os.path.join(config_dir_path, json_file_name)

        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, mode="r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON from the config file {config_file_path}"
                )
                return None
        else:
            logger.warning(
                f"Config file not found: {config_file_path}. Using default kernel settings."
            )
            return None

    @classmethod
    def store_config(cls, params: Dict[str, Any], dest_json: dict) -> None:
        """
        Stores the given configuration as a JSON file based on the provided parameters.
        """
        assert cls != KernelConfigs, "Base class cannot call this class method."

        json_file_name = cls.get_config_file_name(params)
        config_dir_path = cls._get_config_dir_path()

        # Create the directory if it does not exist
        os.makedirs(config_dir_path, exist_ok=True)

        config_file_path = os.path.join(config_dir_path, json_file_name)

        try:
            with open(config_file_path, mode="w") as file:
                json.dump(dest_json, file)
        except (OSError, IOError) as e:
            logger.error(f"Error saving config file {config_file_path}: {e}")
            raise

    @classmethod
    @abstractmethod
    def try_to_get_optimal_kernel_config(cls, *args, **kwargs) -> dict:
        """
        Abstract method that should be implemented by subclasses to return the best configuration.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def save_config(cls, *args, **kwargs) -> None:
        """
        Abstract method that should be implemented by subclasses to save a configuration.
        """
        raise NotImplementedError()
