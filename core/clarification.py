import datetime
"""
core/clarification.py
State manager for Pending Clarifications (Uncertainty-Aware Dialogue Management).
"""
import json
import os
import uuid
from typing import Optional, List
from core.models import PendingClarification, ClarificationOption, ClarificationStatus

class ClarificationManager:
    """
    Manages the lifecycle of Pending Clarifications.
    In a production system, this would be backed by SQLite/Postgres.
    For this implementation, we use a local JSON store to persist state across turns.
    """
    STORE_FILE = ".clarifications.json"
    
    @classmethod
    def _load_store(cls) -> dict:
        if not os.path.exists(cls.STORE_FILE):
            return {}
        try:
            with open(cls.STORE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
            
    @classmethod
    def _save_store(cls, store: dict):
        with open(cls.STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)

    @classmethod
    def create_clarification(cls, utterance: str, schema_expr: str, slot_name: str, candidates: List[ClarificationOption]) -> PendingClarification:
        """Creates a new pending clarification ticket and saves it to the DB."""
        ticket = PendingClarification(
            clarification_id=f"pending_{uuid.uuid4().hex[:8]}",
            original_utterance=utterance,
            proposed_schema_expr=schema_expr,
            unresolved_slot_name=slot_name,
            candidates=candidates
        )
        store = cls._load_store()
        store[ticket.clarification_id] = ticket.to_dict()
        cls._save_store(store)
        return ticket

    @classmethod
    def get_active_clarification(cls) -> Optional[PendingClarification]:
        """Retrieves the most recent WAITING_FOR_USER clarification."""
        store = cls._load_store()
        # Sort by created_at descending
        active_tickets = [
            PendingClarification.from_dict(v) 
            for v in store.values() 
            if v["status"] == ClarificationStatus.WAITING_FOR_USER.value
        ]
        if not active_tickets:
            return None
        active_tickets.sort(key=lambda t: t.created_at, reverse=True)
        return active_tickets[0]

    @classmethod
    def resolve_clarification(cls, clarification_id: str, selected_entity_id: str) -> str:
        """
        Resolves a ticket, substitutes the selected entity into the schema,
        and returns the completed schema expression ready for the Validator.
        """
        store = cls._load_store()
        if clarification_id not in store:
            raise ValueError(f"Clarification {clarification_id} not found.")
            
        ticket_data = store[clarification_id]
        if ticket_data["status"] != ClarificationStatus.WAITING_FOR_USER.value:
            raise ValueError(f"Clarification {clarification_id} is already {ticket_data['status']}.")
            
        # Verify entity exists in candidates
        if not any(c["entity_id"] == selected_entity_id for c in ticket_data["candidates"]):
            raise ValueError("Selected entity ID is not a valid candidate for this clarification.")
            
        # Complete the schema
        completed_schema = ticket_data["proposed_schema_expr"].replace(
            ticket_data["unresolved_slot_name"], 
            selected_entity_id
        )
        
        # Mark as resolved
        ticket_data["status"] = ClarificationStatus.RESOLVED.value
        cls._save_store(store)
        
        return completed_schema

    @classmethod
    def clear_all(cls):
        """FIX 4: Wipes all clarification state. Called on agent init to prevent
        stale tickets from previous sessions freezing the state machine."""
        if os.path.exists(cls.STORE_FILE):
            os.remove(cls.STORE_FILE)

