from datetime import datetime
"""
core/models.py
Strict Dataclass Definitions for the V3 Agent Flow.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

class ClarificationStatus(Enum):
    WAITING_FOR_USER = "WAITING_FOR_USER"
    RESOLVED = "RESOLVED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

@dataclass
class ClarificationOption:
    entity_id: str
    label: str

@dataclass
class PendingClarification:
    clarification_id: str
    original_utterance: str
    proposed_schema_expr: str
    unresolved_slot_name: str
    candidates: List[ClarificationOption]
    status: ClarificationStatus = ClarificationStatus.WAITING_FOR_USER
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "clarification_id": self.clarification_id,
            "original_utterance": self.original_utterance,
            "proposed_schema_expr": self.proposed_schema_expr,
            "unresolved_slot_name": self.unresolved_slot_name,
            "candidates": [{"entity_id": c.entity_id, "label": c.label} for c in self.candidates],
            "status": self.status.value,
            "created_at": self.created_at
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> "PendingClarification":
        return cls(
            clarification_id=data["clarification_id"],
            original_utterance=data["original_utterance"],
            proposed_schema_expr=data["proposed_schema_expr"],
            unresolved_slot_name=data["unresolved_slot_name"],
            candidates=[ClarificationOption(**c) for c in data["candidates"]],
            status=ClarificationStatus(data["status"]),
            created_at=data["created_at"]
        )
