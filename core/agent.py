import os
import json
import logging
from openai import OpenAI
from core.metta_engine import MeTTaEngine, _tokenize_expr, _parse_ast_expr, _ast_to_string
from core.catalogue import EntityCatalogue
from core.persistence import PersistenceManager
from core.jev_adapter import JevDecisionAPI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are Super Memory, a Neuro-Symbolic AI assistant managing an extensible knowledge base.
You must be helpful, conversational, insightful, and precise.
You are powered by a MeTTa Knowledge Graph (AtomSpace). You do NOT know facts inherently—you MUST use your tools to query the graph.

# ReAct Cognitive Loop (Reason -> Act -> Observe)
For every user utterance, you must follow a deliberate ReAct reasoning workflow:
1. Reason:
   - Identify the user's intent: Is this a new fact, a rule, a state update/replacement, or a query?
   - If an entity stepped down, was replaced, changed role, or moved location: What is the existing state in the graph that must be superseded?
2. Act:
   - Always call `get_schema` first to see approved predicates.
   - For queries: Call `search_entities` to get canonical IDs, then `query_knowledge` to inspect the graph.
   - For state replacements/updates: If someone stepped down or a role changed, identify the old atom, call `retract_fact` to remove the outdated atom, then call `learn_fact` with the new atom.
   - For learning: Call `learn_fact` to assert verified facts or rules.
3. Observe:
   - Inspect the returned tool observations. If a query succeeded, synthesize the exact bindings. If a tool threw an error, adjust your plan.

# Rules for State Updates, Replacements & Belief Revision
When someone steps down, is replaced, or an assignment changes (e.g. "X stepped down, and Y took over as Head"):
- You MUST remove the outdated record: Call `retract_fact` for the old occupant.
- Then call `learn_fact` for the new occupant.
- NEVER allow two conflicting occupants to coexist for a single-occupant role or state.

# Historical State & Predecessor Queries (Past / Previous / Successor)
When the user asks who PREVIOUSLY held a role, what changed before, who came before X, or asks about historical events/claims:
- The active AtomSpace only contains the CURRENT truth.
- Historical records are preserved in the permanent evidence ledger!
- You MUST call `query_history` with the entity name, role, or predicate (e.g. `query_history(entity_or_keyword='Nadia Volkov')` or `query_history(predicate='HeadOfDepartment')`).
- Inspect the returned historical claims, operations ('add' vs 'remove'), and original context utterances to deduce who previously occupied the role, when they stepped down, and who succeeded whom.
- Answer the user's historical question accurately based on the returned evidence ledger.

# Rules for Tool Use
1. Call `get_schema` ONCE at the start of any learn or query turn to see approved predicates.
2. If the schema is empty OR no relevant predicate exists, directly use a clean PascalCase predicate in `learn_fact` WITHOUT calling `propose_schema` first.
3. When LEARNING: do NOT call `search_entities` first. Use the name as-is from the user. When QUERYING: call `search_entities` to find the canonical ID before querying.
4. If a predicate already exists in the schema for that meaning, YOU MUST reuse it. Do not invent synonyms.
5. Use `query_knowledge` to find answers. Use `learn_fact` to store knowledge.
6. Complete the FULL workflow in a single turn: get_schema -> learn_fact (or query_knowledge) -> reply to user. Do NOT stop after schema lookup.
7. When answering from `query_knowledge`, strictly limit your reply to the returned `answer_bindings`. Never hallucinate.
8. If a tool returns an error, DO NOT stay silent. Explain the issue and ask the user for help.

# Temporal Exceptions & Recurrences (Stage D)
When the user mentions dates, recurrence, or exceptions, YOU MUST wrap the base fact in a structured temporal record:
- For a specific date: `(ValidDate (MeetingPlace RoboticsClub Lab5) "2026-10-02")`
- For a recurring event: `(Recurrence (MeetingPlace RoboticsClub Lab5) Weekly Friday)`
- For a one-time exception to a recurring event: `(Exception (MeetingPlace RoboticsClub Lab5) "2026-10-09" (MeetingPlace RoboticsClub RoomB))`
NEVER pass raw temporal text (like "next Friday") inside a normal predicate.

# MeTTa Rule Syntax
If you need to learn a conditional rule for backward chaining, you MUST use strict MeTTa syntax:
You MUST start the rule exactly with an equals sign `=`. Example:
`(= (ConclusionPredicate $var) (and (ConditionPredicate1 $var) (ConditionPredicate2 $var)))`
NEVER use Prolog syntax like `:-` or `&`. Always wrap the entire rule in an outer `(= ...)`.


# Conversational Tone & Layman Presentation (The Hood Down Directive)
You are speaking to regular human beings, authors, and decision-makers--NOT software engineers.
1. NEVER expose internal plumbing or code in your final user-facing replies:
   - STRICTLY FORBIDDEN: Do not mention S-expressions, parentheses like '(= ...)', variables like $x, 'AtomSpace', 'SQLite', 'backward chaining', 'unification', 'AST', 'bindings', or 'predicates'.
2. Translate all symbolic inferences and proofs into crisp, natural English:
   - INSTEAD OF: 'Queried (CapitalCity Nation_075 $cap) and retrieved City_075.'
   - SAY: 'The capital of Nation 075 is City 075.'
   - INSTEAD OF: 'Proved via backward chaining rule (= (SubjectToJurisdiction $sub $country) ...) that Subsidiary_03_2 is subject to Nation_010.'
   - SAY: 'Yes, Subsidiary 03-2 is under the jurisdiction of Nation 010 because its parent corporation, MegaCorp 03, is headquartered there.'
3. When asking for clarification or missing information, sound like a thoughtful, helpful human partner, never like a database error message.
Do NOT fabricate knowledge. If `query_knowledge` returns no results or NO_SUPPORTING_CLAIM, tell the user you don't know and ask them to teach you.
"""

class NeuroSymbolicAgent:
    def __init__(self, engine: MeTTaEngine, catalogue: EntityCatalogue, pm: PersistenceManager):
        self.engine = engine
        self.catalogue = catalogue
        self.pm = pm
        self.jev = JevDecisionAPI()
        
        from dotenv import load_dotenv
        load_dotenv(override=True)
        # Force switch to ASI Cloud
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://llm.c.singularitynet.io/v1")
        model = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._schema_checked = False

        # FIX 4: Clear any stale clarification state from previous sessions.
        # Prevents frozen state across database wipes or server restarts.
        try:
            from core.clarification import ClarificationManager
            ClarificationManager.clear_all()
        except Exception:
            pass

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_entities",
                    "description": "Fuzzy search the knowledge graph for entities matching a keyword (e.g. 'hackathon' or 'SecHack'). Returns canonical IDs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"}
                        },
                        "required": ["keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Retrieves the list of all predicates and rules currently defined in the AtomSpace. Use this to prevent hallucinating new predicates.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_schema",
                    "description": "Propose a new schema predicate when you encounter a concept that cannot be expressed using the existing schema. This explicitly defines the predicate's meaning and arguments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "predicate_name": {"type": "string", "description": "PascalCase name of the new predicate (e.g. Loves)"},
                            "argument_types": {"type": "array", "items": {"type": "string"}, "description": "List of expected arguments (e.g. ['Person', 'Person'])"},
                            "meaning": {"type": "string", "description": "A clear definition of what this predicate represents."}
                        },
                        "required": ["predicate_name", "argument_types", "meaning"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge",
                    "description": "Executes a MeTTa query against the AtomSpace. Example: (Location AISecurityHackathon $x)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metta_query": {"type": "string"}
                        },
                        "required": ["metta_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "learn_fact",
                    "description": "Asserts a new fact into the AtomSpace. Example: (Location AISecurityHackathon BuildingC)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metta_atom": {"type": "string"}
                        },
                        "required": ["metta_atom"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "request_clarification",
                    "description": "Call this when the user's utterance is ambiguous and you have multiple entity candidates for a slot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "proposed_schema_expr": {"type": "string"},
                            "unresolved_slot_name": {"type": "string"},
                            "candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "entity_id": {"type": "string"},
                                        "label": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["proposed_schema_expr", "unresolved_slot_name", "candidates"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "resolve_pending_clarification",
                    "description": "Call this to resolve an active clarification ticket once the user specifies their choice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "clarification_id": {"type": "string"},
                            "selected_entity_id": {"type": "string"}
                        },
                        "required": ["clarification_id", "selected_entity_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_history",
                    "description": "Queries the permanent evidence ledger for historical claims, predecessors, and past states that were added or retracted/removed. Use this when the user asks who previously held a role, what changed before, or about past history.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_or_keyword": {
                                "type": "string",
                                "description": "Entity name, keyword, or role to search historical ledger for (e.g. 'DrNadiaVolkov', 'Cybersecurity', or 'MarcusSterling')."
                            },
                            "predicate": {
                                "type": "string",
                                "description": "Optional predicate filter (e.g. 'HeadOfDepartment')."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retract_fact",
                    "description": "Removes a fact from the AtomSpace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metta_atom": {"type": "string"}
                        },
                        "required": ["metta_atom"]
                    }
                }
            }
        ]


    def _handle_tool_call(self, name: str, args: dict, user_input: str) -> str:
        """Executes the tool and applies Jev guardrails where necessary."""
        logger.info(f"Tool Call: {name}({args})")
        
        if name == "search_entities":
            keyword = args.get("keyword", "")
            cands = self.catalogue.search(keyword)
            if not cands:
                return json.dumps({"status": "error", "message": "No entities found."})

            cand_ids = [c.canonical_id for c in cands]
            
            # --- JEV CALIBRATED DISAMBIGUATION ---
            # If multiple candidates found, use Jev to calibrate whether there's a dominant match
            # or if genuine ambiguity exists. This prevents spurious title collisions (e.g. Dr).
            if len(cand_ids) > 1:
                query_context = self.pm.get_utterance_text(user_input) or keyword
                decision = self.jev.disambiguate_entity(query_context, cand_ids)
                if decision in cand_ids:
                    logger.info(f"Jev calibrated selection: '{decision}' from candidates {cand_ids}")
                    return json.dumps({
                        "candidates": [decision],
                        "disambiguation_note": f"Jev calibrated match: '{decision}'"
                    })
                elif decision == "NONE":
                    return json.dumps({"status": "error", "message": "No relevant entities matched the query."})

            return json.dumps({"candidates": cand_ids})
            
        elif name == "get_schema":
            self._schema_checked = True
            predicates = set()

            def walk_ast_preds(node):
                if isinstance(node, list) and node:
                    fn = node[0]
                    if isinstance(fn, str) and not fn.startswith("$"):
                        if fn == "SchemaDefinition" and len(node) > 1 and isinstance(node[1], str):
                            predicates.add(node[1])
                        elif fn not in ("=", ":", "and", "or", ",", "Rule", "ValidDate", "Recurrence", "Exception"):
                            predicates.add(fn)
                    for child in node[1:]:
                        walk_ast_preds(child)

            all_exprs = list(self.engine.get_all_atoms()) + list(getattr(self.engine, "_rules", []))
            for a in all_exprs:
                try:
                    ast, _ = _parse_ast_expr(_tokenize_expr(a))
                    walk_ast_preds(ast)
                except Exception:
                    pass

            if not predicates:
                return json.dumps({"status": "success", "schema": "Schema is currently empty. You may define new predicates."})
            return json.dumps({"status": "success", "schema": sorted(list(predicates))})
        elif name == "propose_schema":
            pred = args.get("predicate_name")
            args_types = args.get("argument_types", [])
            meaning = args.get("meaning")
            
            if not pred or not pred[0].isupper() or not pred.isalnum():
                return json.dumps({"status": "error", "message": "Predicate must be valid PascalCase (start with a capital letter and contain only alphanumeric characters)."})

            # --- JEV SCHEMA CRITIC ---
            try:
                critic_eval = self.jev.evaluate_schema(pred, meaning)
                if critic_eval and critic_eval.strip().upper().startswith("REJECT"):
                    logger.warning(f"Schema Critic Rejected {pred}: {critic_eval}")
                    return json.dumps({
                        "status": "error", 
                        "message": f"SCHEMA CRITIC REJECTED YOUR PROPOSAL: {critic_eval}"
                    })
            except Exception as e:
                logger.error(f"Critic evaluation failed: {e}")
                pass
                
            # Write a SchemaDefinition atom to the DB
            types_str = " ".join(args_types) if args_types else "Entity"
            schema_atom = f'(SchemaDefinition {pred} "{types_str}" "{meaning}")'
            
            import hashlib
            op_id = "op_schema_" + hashlib.sha256(f"{user_input}:{schema_atom}".encode()).hexdigest()[:12]
            
            if self.pm.operation_exists(op_id):
                return json.dumps({"status": "success", "message": f"Duplicate schema operation skipped."})
                
            self.engine.add_atom(schema_atom)
            self.pm.record_modification(self.engine, user_input, schema_atom, "add", op_id=op_id)
            
            return json.dumps({
                "status": "success", 
                "message": f"Successfully registered new predicate '{pred}'. You may now use it in learn_fact."
            })
            
        elif name == "query_knowledge":
            query = args.get("metta_query", "")
            template = query
            
            result = self.engine.query(query, template)
            used_bc = False
            if result.is_empty:
                import re as _re
                bc_expr = query
                m = _re.match(r"\(match\s+&self\s+(.+?)\s+\$\w+\)$", bc_expr)
                if m:
                    bc_expr = m.group(1)
                result = self.engine.backward_chain(bc_expr)
                used_bc = True
                
            if result.is_empty:
                return json.dumps({"status": "NO_SUPPORTING_CLAIM", "results": []})
            
            # Build derivation-indexed lookup for BC results
            derivation_map = {}
            if used_bc:
                for drec in result.derivations:
                    derivation_map[drec.answer] = drec
            
            evidence_packages = []
            for res_str in result.raw:
                c_id = self.pm.get_claim_by_atom(res_str)
                if c_id:
                    evidence_packages.append({
                        "status": "ANSWERED",
                        "answer_bindings": res_str,
                        "supporting_claim_ids": [c_id],
                        "origin": "reported"
                    })
                elif res_str in derivation_map:
                    drec = derivation_map[res_str]
                    premise_claim_ids = []
                    for p in drec.premises:
                        pid = self.pm.get_claim_by_atom(p)
                        if pid:
                            premise_claim_ids.append(pid)
                    evidence_packages.append({
                        "status": "ANSWERED",
                        "answer_bindings": res_str,
                        "origin": "derived",
                        "applied_rule": drec.rule_text,
                        "premise_claim_ids": premise_claim_ids,
                        "premises": drec.premises
                    })
                else:
                    # No claim ID and no derivation record = sync error
                    evidence_packages.append({
                        "status": "UNVERIFIED",
                        "answer_bindings": res_str,
                        "origin": "unknown",
                        "warning": "No claim ID or derivation record found. Possible sync error."
                    })
                
            return json.dumps({"status": "success", "evidence": evidence_packages})
            
        elif name == "learn_fact":
            from core.clarification import ClarificationManager
            if ClarificationManager.get_active_clarification():
                return json.dumps({
                    "status": "error",
                    "message": "FATAL ERROR: There is an active clarification ticket. You CANNOT guess or learn facts right now. You MUST ask the user which option they meant."
                })
            
            atom = args.get("metta_atom", "")
            
            if not self._schema_checked:
                return json.dumps({
                    "status": "error", 
                    "message": "VALIDATION FAILED: You must call get_schema before learning a fact to prevent predicate hallucination."
                })
                
            # ── Universal Adaptive AST S-Expression Parser (Zero Regex) ──
            try:
                tokens = _tokenize_expr(atom)
                ast, remainder = _parse_ast_expr(tokens)
                if not isinstance(ast, list) or not ast:
                    return json.dumps({"status": "error", "message": "VALIDATION FAILED: Malformed atom. Must be a valid parenthesized S-expression."})
            except Exception as e:
                return json.dumps({"status": "error", "message": f"VALIDATION FAILED: S-expression parse error: {e}"})

            functor = ast[0]

            # 1. Rule Handling: Canonicalize (= Head Body) or (Rule Head Body) into MeTTa equality rules
            if functor in ("=", "Rule", ":-"):
                if len(ast) < 3:
                    return json.dumps({"status": "error", "message": "VALIDATION FAILED: Rule must have at least head and body: (= (Head) (Body))"})
                atom = f"(= {_ast_to_string(ast[1])} {_ast_to_string(ast[2])})"
                predicate = "="
            else:
                # Direct ground fact or temporal wrapper
                # Collect all known predicates from engine + rules
                known_preds = set()
                def collect_preds(node):
                    if isinstance(node, list) and node:
                        fn = node[0]
                        if isinstance(fn, str) and not fn.startswith("$"):
                            if fn == "SchemaDefinition" and len(node) > 1 and isinstance(node[1], str):
                                known_preds.add(node[1])
                            elif fn not in ("=", ":", "and", "or", ",", "Rule", "ValidDate", "Recurrence", "Exception"):
                                known_preds.add(fn)
                        for child in node[1:]:
                            collect_preds(child)

                for a in list(self.engine.get_all_atoms()) + list(getattr(self.engine, "_rules", [])):
                    try:
                        e_ast, _ = _parse_ast_expr(_tokenize_expr(a))
                        collect_preds(e_ast)
                    except Exception:
                        pass

                utterance_txt = self.pm.get_utterance_text(user_input) or atom

                # JEV PREDICATE ALIGNMENT: Check if predicate or inner wrapped predicate needs alignment
                if functor in ("ValidDate", "Recurrence", "Exception"):
                    predicate = functor
                    if len(ast) > 1 and isinstance(ast[1], list) and ast[1]:
                        target_pred = ast[1][0]
                        if target_pred not in known_preds and known_preds:
                            aligned = self.jev.align_predicate(target_pred, utterance_txt, list(known_preds))
                            if aligned in known_preds:
                                logger.info(f"Jev aligned wrapped predicate '{target_pred}' -> '{aligned}'")
                                ast[1][0] = aligned
                    atom = _ast_to_string(ast)
                else:
                    target_pred = functor
                    if target_pred not in known_preds and known_preds:
                        aligned = self.jev.align_predicate(target_pred, utterance_txt, list(known_preds))
                        if aligned in known_preds:
                            logger.info(f"Jev aligned predicate '{target_pred}' -> '{aligned}'")
                            ast[0] = aligned
                            target_pred = aligned
                    predicate = ast[0]
                    atom = _ast_to_string(ast)

                    # Auto-register if brand-new
                    if known_preds and predicate not in known_preds:
                        if predicate[0].isupper() and predicate.isalnum():
                            self.engine.add_atom(f'(SchemaDefinition {predicate} "Entity Entity" "Auto-registered relation")')
                        else:
                            return json.dumps({
                                "status": "error",
                                "message": f"VALIDATION FAILED: Predicate '{predicate}' must be valid PascalCase."
                            })

            # Temporal boundaries: check both atom AND original utterance
            import re
            utterance_text = self.pm.get_utterance_text(user_input) or ""
            combined_text = atom + " " + utterance_text
            
            is_temporal_wrapper = predicate in ("ValidDate", "Recurrence", "Exception")
            
            if not is_temporal_wrapper and re.search(r"\b(tomorrow|next\s+\w+day|next\s+week|20\d\d|\d{1,2}/\d{1,2}/\d{2,4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", combined_text, re.IGNORECASE):
                return json.dumps({
                    "status": "error",
                    "message": "VALIDATION FAILED: Temporal scope detected in utterance or atom, but you did not use a structured temporal wrapper. You MUST wrap the fact in (ValidDate ...), (Recurrence ...), or (Exception ...). Do not put raw dates in standard predicates."
                })
            
            # Idempotent write: generate operation ID, check for duplicate
            import hashlib
            op_id = "op_" + hashlib.sha256(f"{user_input}:{atom}:add".encode()).hexdigest()[:12]
            if self.pm.operation_exists(op_id):
                return json.dumps({"status": "success", "message": f"Duplicate operation {op_id} already committed. Skipped."})
                
            self.engine.add_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            self.pm.record_modification(self.engine, user_input, atom, "add", op_id=op_id)
            
            self._schema_checked = False
            return json.dumps({"status": "success", "message": f"Learned {atom}", "operation_id": op_id})
            
        elif name == "query_history":
            entity_or_keyword = args.get("entity_or_keyword", "")
            predicate = args.get("predicate", "")
            
            # Clean entity keyword if passed as natural language
            clean_kw = entity_or_keyword.replace(" ", "").replace(".", "").replace('"', '').replace("'", "")
            
            records = self.pm.query_history(entity_or_keyword=clean_kw, predicate=predicate)
            if not records and entity_or_keyword != clean_kw:
                records = self.pm.query_history(entity_or_keyword=entity_or_keyword, predicate=predicate)
                
            if not records:
                return json.dumps({
                    "status": "success",
                    "history": [],
                    "message": f"No historical records found for '{entity_or_keyword}'."
                })
                
            return json.dumps({
                "status": "success",
                "history": records,
                "message": f"Retrieved {len(records)} historical records from evidence ledger."
            })

        elif name == "retract_fact":
            atom = args.get("metta_atom", "")
            # FIX 5: Verify atom exists before retracting. Silent success on missing
            # atoms causes State Hijack — LLM thinks old fact was removed when it wasn't.
            existing_atoms = self.engine.get_all_atoms()
            if atom not in existing_atoms:
                return json.dumps({
                    "status": "error",
                    "message": f"RETRACT FAILED: Atom '{atom}' does not exist in the AtomSpace. Use get_schema or query_knowledge to find the exact atom string before retracting."
                })
            self.engine.remove_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            self.pm.record_modification(self.engine, user_input, atom, "remove")
            return json.dumps({"status": "success", "message": f"Retracted {atom}"})
            
        elif name == "request_clarification":
            from core.clarification import ClarificationManager
            from core.models import ClarificationOption
            schema_expr = args.get("proposed_schema_expr", "")
            slot_name = args.get("unresolved_slot_name", "")
            cands = args.get("candidates", [])

            # UNIVERSAL GUARD: Clarification only makes sense with 2+ distinct candidates.
            # With 0 or 1 candidate, there is nothing to disambiguate — proceed directly.
            if len(cands) <= 1:
                return json.dumps({
                    "status": "skipped",
                    "message": f"No ambiguity detected (only {len(cands)} candidate). Clarification not needed. Proceed with learn_fact or query_knowledge directly."
                })

            options = [ClarificationOption(entity_id=c.get("entity_id"), label=c.get("label")) for c in cands]
            
            try:
                ticket = ClarificationManager.create_clarification(user_input, schema_expr, slot_name, options)
                return json.dumps({
                    "status": "success",
                    "message": f"STATE FROZEN. Ticket ID: {ticket.clarification_id}. DO NOT GUESS. DO NOT CALL LEARN_FACT. You MUST immediately stop using tools and ask the user a polite List-Based question asking them to choose between the candidates."
                })
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})
                
        elif name == "resolve_pending_clarification":
            from core.clarification import ClarificationManager
            ticket_id = args.get("clarification_id")
            entity_id = args.get("selected_entity_id")
            try:
                completed_schema = ClarificationManager.resolve_clarification(ticket_id, entity_id)
                return json.dumps({
                    "status": "success",
                    "message": f"Ticket resolved. You may now call learn_fact with atom: {completed_schema}"
                })
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        return json.dumps({"status": "error", "message": "Unknown tool"})

    def chat(self, messages: list, user_input: str) -> str:
        """
        Executes the conversational loop with Jev intent routing and tool calling.
        messages should include the system prompt and conversation history.
        """
        # Fix 1: Reset schema guard per-turn, not per-instance
        self._schema_checked = False

        eval_input = user_input
        messages.append({"role": "user", "content": eval_input})
        u_id = self.pm.record_utterance(eval_input)

        # 1. Fast Intent Routing via Jev
        intent = self.jev.route_intent(eval_input)

        # Fix 2: Inject JEV intent into the user message, NOT as a mid-conversation
        # system message. OpenAI spec only allows system role at position 0.
        # Sent clean to UI; inject intent only into windowed copy for LLM

        # 3. Slow Brain executes with tools
        for _ in range(10):
            # Fix 3: Rolling window — always send system prompt + last 20 messages only.
            # Prevents context blowout and 429 amplification from unbounded message lists.
            system_msg = messages[0]
            history = messages[1:]
            history_copy = [dict(m) for m in history]
            for m in reversed(history_copy):
                if m.get("role") == "user":
                    m["content"] = f"[JEV INTENT: {intent.upper()}]\n{m['content']}"
                    break
            windowed = [system_msg] + history_copy[-20:]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=windowed,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            msg = response.choices[0].message
            # Append dict to prevent Streamlit session state serialization crash
            msg_dict = msg.model_dump(exclude_unset=True)
            messages.append(msg_dict)
            
            if not msg.tool_calls:
                return msg.content or "I couldn't find an answer in my knowledge graph. Could you rephrase?"
                
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}
                    
                tool_result_str = self._handle_tool_call(fn_name, args, u_id)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": tool_result_str
                })
                
        return "I had to stop thinking because I used too many tools. Could you rephrase your question?"
