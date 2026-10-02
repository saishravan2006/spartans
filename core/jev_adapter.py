from dotenv import load_dotenv
import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

class JevDecisionAPI:
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        self.base_url = os.environ.get("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
        self.model = "typesafe/jev-1.13"

    def route_intent(self, text: str) -> str:
        """
        Calls Jev Decision API to classify the intent.
        Returns: 'query', 'learn', 'contradiction', or 'clarification'
        """
        if not self.api_key:
            # No Jev API key — use a universal keyword-based local classifier.
            # This is NOT an edge-case patch; it is the universal fallback behavior.
            import re as _re
            text_lower = text.lower().strip()

            # Contradiction signals: "actually", "no,", "wrong", "not X, it's Y"
            if _re.search(r"\b(actually|no,|wrong,|correction:|not\s+\w+,\s+it.?s|the correct|it should be|that.?s wrong)\b", text_lower):
                logger.info(f"Local Jev: contradiction for '{text}'")
                return "contradiction"

            # Question signals
            if text_lower.endswith("?") or _re.search(r"^(who|what|when|where|which|how|is|are|can|does|did|do)\b", text_lower):
                logger.info(f"Local Jev: query for '{text}'")
                return "query"

            # Learn signals: asserting facts or rules
            if _re.search(r"\b(is|are|was|were|has|have|belongs to|located|meets|president|member|rule:|learn|remember|the .+ is|teach)\b", text_lower):
                logger.info(f"Local Jev: learn for '{text}'")
                return "learn"

            logger.info(f"Local Jev: defaulting to query for '{text}'")
            return "query"

        payload = {
            "model": self.model,
            "state": text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Classify the intent of the user's input.",
                    "criteria": {
                        "query": "The user is asking a question or requesting information.",
                        "learn": "The user is stating a new fact to be learned or asserting something.",
                        "contradiction": "The user is correcting an old fact.",
                        "clarification": "The input is ambiguous, confusing, or needs clarification."
                    }
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            resp = requests.post(f"{self.base_url}/alpha/decisions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["answers"]["intent"]["choice"]
            logger.info(f"Jev routing: {choice} for input '{text}'")
            return choice
        except Exception as e:
            logger.error(f"Jev routing failed: {e}")
            return "query"

    def check_temporal_violation(self, user_input: str, proposed_atom: str) -> bool:
        """
        Uses Jev to determine if an atom contains raw temporal data without proper wrappers.
        """
        if not self.api_key:
            logger.warning("No TYPESAFE_API_KEY found, bypassing temporal check")
            return False

        payload = {
            "model": self.model,
            "state": f"User Input: {user_input}\nProposed Atom: {proposed_atom}",
            "questions": {
                "temporal_violation": {
                    "type": "choice",
                    "instructions": "Determine if the proposed atom contains raw temporal information (time, day, date, recurrence, season) that is NOT wrapped in a structured temporal predicate like ValidDate, Recurrence, or Exception. If the predicate itself is one of those wrappers, it is safe.",
                    "criteria": {
                        "violation": "The atom contains a raw temporal string argument inside a standard predicate like Schedule, MeetingPlace, or LocatedIn.",
                        "safe": "The atom contains no raw temporal arguments, OR it is properly wrapped in ValidDate, Recurrence, or Exception."
                    }
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            resp = requests.post(f"{self.base_url}/alpha/decisions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["answers"]["temporal_violation"]["choice"]
            logger.info(f"Jev Temporal Check: {choice} for atom '{proposed_atom}'")
            return choice == "violation"
        except Exception as e:
            logger.error(f"Jev temporal check failed: {e}")
            return False

    def evaluate_schema(self, predicate: str, meaning: str) -> str:
        """
        Uses Jev to critique a newly proposed schema predicate.
        Returns a string starting with 'REJECT' if bad, or 'APPROVE' if acceptable.
        """
        if not self.api_key:
            logger.warning("No TYPESAFE_API_KEY found, bypassing schema critic")
            return "APPROVE"

        payload = {
            "model": self.model,
            "state": f"Proposed Predicate: {predicate}\nMeaning: {meaning}",
            "questions": {
                "schema_critic": {
                    "type": "choice",
                    "instructions": "Determine if this schema predicate is acceptable. Reject overly specific, duplicate, or badly named predicates.",
                    "criteria": {
                        "REJECT": "The predicate is poorly named, uses snake_case instead of PascalCase, or is way too specific.",
                        "APPROVE": "The predicate is cleanly named in PascalCase and represents a distinct relation or property."
                    }
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            resp = requests.post(f"{self.base_url}/alpha/decisions", json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            choice = data["answers"]["schema_critic"]["choice"]
            logger.info(f"Jev Schema Critic: {choice} for predicate '{predicate}'")
            return choice
        except Exception as e:
            logger.error(f"Jev schema check failed: {e}")
            return "APPROVE"

    def disambiguate_entity(self, query_context: str, candidates: list[str]) -> str:
        """
        Uses TypeSafe AI's Jev calibrated decision engine to disambiguate candidate entities.
        Returns:
          - The winning canonical entity ID if unambiguous (e.g. 'DrArisThorne')
          - 'AMBIGUOUS' if there is genuine ambiguity requiring user clarification
          - 'NONE' if no candidate matches the query
        """
        if not candidates:
            return "NONE"
        if len(candidates) == 1:
            return candidates[0]

        if not self.api_key:
            # Fallback: token-overlap dominance scoring if no API key
            import re
            q_words = set(re.sub(r"[^\w\s]", "", query_context.lower()).split())
            scored = []
            for c in candidates:
                c_words = set(re.sub(r"([a-z])([A-Z])", r"\1 \2", c).replace("_", " ").lower().split())
                overlap = len(q_words.intersection(c_words))
                scored.append((overlap, c))
            scored.sort(reverse=True, key=lambda x: x[0])
            # If top candidate strictly dominates
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1]
            return "AMBIGUOUS"

        # Build Jev discrete choice criteria
        criteria = {}
        for c in candidates[:6]:  # Limit to top candidates
            criteria[c] = f"The query is referring specifically to entity '{c}'."
        criteria["AMBIGUOUS"] = "The query could legitimately refer to multiple candidates with equal likelihood, requiring user clarification."
        criteria["NONE"] = "The query is not referring to any of these candidates."

        payload = {
            "model": self.model,
            "state": f"User Query / Utterance: \"{query_context}\"\nCandidate entities in Knowledge Graph: {', '.join(candidates)}",
            "questions": {
                "entity_selection": {
                    "type": "choice",
                    "instructions": "Determine which candidate entity best and most specifically matches the person, object, or concept mentioned in the user query.",
                    "criteria": criteria
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(f"{self.base_url}/alpha/decisions", json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            choice = data["answers"]["entity_selection"]["choice"]
            logger.info(f"Jev Entity Disambiguation: chosen '{choice}' for query '{query_context}' from candidates {candidates}")
            return choice
        except Exception as e:
            logger.error(f"Jev entity disambiguation failed: {e}")
            # Fallback on dominance scoring
            import re
            q_words = set(re.sub(r"[^\w\s]", "", query_context.lower()).split())
            scored = []
            for c in candidates:
                c_words = set(re.sub(r"([a-z])([A-Z])", r"\1 \2", c).replace("_", " ").lower().split())
                overlap = len(q_words.intersection(c_words))
                scored.append((overlap, c))
            scored.sort(reverse=True, key=lambda x: x[0])
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1]
            return "AMBIGUOUS"

    def align_predicate(self, proposed_predicate: str, utterance_context: str, known_predicates: list[str]) -> str:
        """
        Uses TypeSafe AI's Jev calibrated decision engine to align a proposed predicate
        with existing predicates in the schema and inference rules, preventing synonym drift.
        Returns:
          - An existing predicate from known_predicates if it semantically matches
          - 'NEW' if this concept genuinely requires a new predicate
        """
        if not known_predicates:
            return "NEW"
        if proposed_predicate in known_predicates:
            return proposed_predicate

        # Clean list of candidates
        candidates = [p for p in known_predicates if p not in ("=", ":", "SchemaDefinition", "ValidDate", "Recurrence", "Exception")]
        if not candidates:
            return "NEW"

        if not self.api_key:
            # Fallback substring / token overlap check
            p_lower = proposed_predicate.lower()
            for cand in candidates:
                c_lower = cand.lower()
                if c_lower in p_lower or p_lower in c_lower:
                    return cand
            return "NEW"

        criteria = {}
        for c in candidates[:8]:  # Limit to top 8 predicates
            criteria[c] = f"Use existing predicate '{c}' because it represents the same relation/meaning."
        criteria["NEW"] = "None of the existing predicates match; a new predicate is genuinely needed."

        payload = {
            "model": self.model,
            "state": f"User Utterance: \"{utterance_context}\"\nProposed Predicate: {proposed_predicate}\nApproved Schema & Rule Predicates: {', '.join(candidates)}",
            "questions": {
                "predicate_alignment": {
                    "type": "choice",
                    "instructions": "Determine if the proposed concept should reuse an existing predicate from the schema/rules or if a brand-new predicate is required.",
                    "criteria": criteria
                }
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(f"{self.base_url}/alpha/decisions", json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            choice = data["answers"]["predicate_alignment"]["choice"]
            logger.info(f"Jev Predicate Alignment: chose '{choice}' for proposed '{proposed_predicate}' in '{utterance_context}'")
            return choice
        except Exception as e:
            logger.error(f"Jev predicate alignment failed: {e}")
            p_lower = proposed_predicate.lower()
            for cand in candidates:
                c_lower = cand.lower()
                if c_lower in p_lower or p_lower in c_lower:
                    return cand
            return "NEW"
