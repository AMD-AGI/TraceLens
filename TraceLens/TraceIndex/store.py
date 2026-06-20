###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Storage interface for TraceIndex backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from TraceLens.TraceIndex.models import SearchHit, TraceRecord, TraceReport


class TraceIndexStore(ABC):
    """Persistence boundary for TraceIndex.

    New backends should implement this interface without changing scanner,
    importer, or CLI workflow code.
    """

    @abstractmethod
    def init_schema(self) -> None:
        pass

    @abstractmethod
    def upsert_trace(self, trace: TraceRecord) -> int:
        pass

    @abstractmethod
    def import_report(self, trace_id: int, report: TraceReport) -> None:
        pass

    @abstractmethod
    def search(self, terms: str, limit: int = 50) -> List[SearchHit]:
        pass

    @abstractmethod
    def execute_read_query(
        self,
        sql: str,
        params: Optional[Sequence[Any]] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        pass

    def close(self) -> None:
        pass
