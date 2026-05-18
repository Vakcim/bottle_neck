from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tinvest_token: str
    tinvest_sandbox: bool = True
    data_dir: str = "./data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()


def get_settings() -> AppSettings:
    load_dotenv()
    return AppSettings()
