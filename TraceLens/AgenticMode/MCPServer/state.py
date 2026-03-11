"""In-memory trace cache with TTL eviction and LRU ordering."""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    analyzer: object
    metadata: dict
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        self.last_accessed = time.time()

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


class TraceCache:
    """Thread-safe trace cache with TTL and LRU eviction."""

    def __init__(self, max_size: int = 20, ttl_hours: int = 24):
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl_hours = ttl_hours

    @staticmethod
    def make_trace_id(trace_path: str, platform: Optional[str] = None,
                      trace_type: str = "pytorch",
                      enable_pseudo_ops: bool = True) -> str:
        key = f"{trace_path}|{platform}|{trace_type}|{enable_pseudo_ops}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def store(self, trace_id: str, analyzer: object, metadata: dict):
        with self._lock:
            self._evict_expired_locked()
            if len(self._entries) >= self._max_size and trace_id not in self._entries:
                self._evict_lru_locked()
            self._entries[trace_id] = _CacheEntry(analyzer=analyzer, metadata=metadata)
        logger.info("Cached trace %s (total: %d)", trace_id, len(self._entries))

    def get_analyzer(self, trace_id: str):
        with self._lock:
            entry = self._entries.get(trace_id)
            if entry is None:
                return None
            entry.touch()
            return entry.analyzer

    def get_metadata(self, trace_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._entries.get(trace_id)
            if entry is None:
                return None
            entry.touch()
            return entry.metadata

    def has(self, trace_id: str) -> bool:
        with self._lock:
            return trace_id in self._entries

    def evict(self, trace_id: str) -> bool:
        with self._lock:
            if trace_id in self._entries:
                del self._entries[trace_id]
                logger.info("Evicted trace %s", trace_id)
                return True
            return False

    def list_traces(self) -> list:
        with self._lock:
            return [
                {"trace_id": tid, "age_hours": round(e.age_hours(), 2), **e.metadata}
                for tid, e in self._entries.items()
            ]

    def cleanup(self):
        """Evict all expired entries."""
        with self._lock:
            self._evict_expired_locked()

    def _evict_expired_locked(self):
        expired = [
            tid for tid, e in self._entries.items()
            if e.age_hours() > self._ttl_hours
        ]
        for tid in expired:
            del self._entries[tid]
            logger.info("TTL-evicted trace %s", tid)

    def _evict_lru_locked(self):
        if not self._entries:
            return
        lru_id = min(self._entries, key=lambda tid: self._entries[tid].last_accessed)
        del self._entries[lru_id]
        logger.info("LRU-evicted trace %s", lru_id)
