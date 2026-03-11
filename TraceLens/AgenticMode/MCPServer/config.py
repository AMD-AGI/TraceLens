"""Server configuration loaded from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    trace_ttl_hours: int = 24
    max_cached_traces: int = 20

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=os.getenv("TRACELENS_HOST", "0.0.0.0"),
            port=int(os.getenv("TRACELENS_PORT", "8000")),
            trace_ttl_hours=int(os.getenv("TRACELENS_TTL_HOURS", "24")),
            max_cached_traces=int(os.getenv("TRACELENS_MAX_CACHED", "20")),
        )


config = Config.from_env()
