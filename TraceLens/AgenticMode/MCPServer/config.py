"""Server configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=os.getenv("TRACELENS_HOST", "0.0.0.0"),
            port=int(os.getenv("TRACELENS_PORT", "8000")),
        )


config = Config.from_env()
