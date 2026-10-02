"""
core/metta_engine.py
MeTTa / Hyperon AtomSpace wrapper with Equivalence Traversal & Belief Revision.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from hyperon import MeTTa as _HyperonMeTTa
    HYPERON_AVAILABLE = True
except ImportError:
    HYPERON_AVAILABLE = False
    logger.warning("hyperon package not found — running in MOCK mode.")


def validate_sexpr(expr: str) -> bool:
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0 and len(expr.strip()) > 0


def sanitize_sexpr(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```\w*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    # Strip any stray quotes that LLMs might hallucinate inside expressions
    cleaned = cleaned.replace('"', '').replace("'", "")
    cleaned = cleaned.strip()

    if not validate_sexpr(cleaned):
        raise MeTTaSyntaxError(f"Unbalanced parentheses in expression: {cleaned!r}")
    return cleaned


class MeTTaSyntaxError(Exception):
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock AtomSpace with Native Equivalence & Belief Revision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Predicates where an entity can only have one value at a time
SINGLE_VALUED_PREDICATES = {
    "location", "time", "date", "status", "manager", "room", 
    "clearance", "hasclearance", "role", "hasrole",
    "hascredential", "requirescredential",
    "headofdepartment", "president", "director", "chiefaiarchitect",
    "chair", "lead", "dean", "provost", "head"
}


# V3 NOTE: _MockSpace and _MockMeTTa are intentional dev-mode fallbacks.
# They activate automatically when the Hyperon package is not installed.
# In production, install hyperon and these classes are bypassed entirely.
class _MockSpace:
    def __init__(self) -> None:
        self._atoms: list[str] = []

    def get_aliases(self, entity: str) -> set[str]:
        """Traverse (= (A) (B)) atoms bidirectionally to find all synonyms."""
        aliases = {entity}
        changed = True
        while changed:
            changed = False
            for atom in self._atoms:
                # Match (= (TermA) (TermB)) or (= TermA TermB) natively
                m = re.match(r"^\(=\s+(?:\((.+?)\)|(\S+))\s+(?:\((.+?)\)|(\S+))\)$", atom.strip())
                if m:
                    a = (m.group(1) or m.group(2)).strip()
                    b = (m.group(3) or m.group(4)).strip()
                    if a in aliases and b not in aliases:
                        aliases.add(b)
                        changed = True
                    elif b in aliases and a not in aliases:
                        aliases.add(a)
                        changed = True
        return aliases

    def add_atom(self, expr: str) -> None:
        normalised = " ".join(expr.split())
        tokens = self._tokenise(normalised)

        # Belief revision: If (Predicate Subject Value), retract prior values for Subject (and aliases)
        if len(tokens) >= 5 and tokens[0] == "(" and tokens[-1] == ")":
            pred = tokens[1].lower()
            subject = tokens[2]
            if pred in SINGLE_VALUED_PREDICATES:
                aliases = self.get_aliases(subject)
                # Find and remove old conflicting atoms
                to_remove = []
                for existing in self._atoms:
                    e_tokens = self._tokenise(existing)
                    if len(e_tokens) == len(tokens):
                        if e_tokens[1].lower() == pred and e_tokens[2] in aliases:
                            to_remove.append(existing)
                for old in to_remove:
                    self._atoms.remove(old)
                    logger.info("Belief revision: retracted outdated atom %s", old)

        if normalised not in self._atoms:
            self._atoms.append(normalised)

    def remove_atom(self, expr: str) -> bool:
        normalised = " ".join(expr.split())
        if normalised in self._atoms:
            self._atoms.remove(normalised)
            return True
        return False

    def get_atoms(self) -> list[str]:
        return list(self._atoms)

    def _base_query(self, pattern: str) -> list[str]:
        """Single-hop base pattern matching."""
        results: list[str] = []
        pat_tokens = self._tokenise(pattern)

        for atom in self._atoms:
            atom_tokens = self._tokenise(atom)
            if self._match(pat_tokens, atom_tokens):
                results.append(atom)
        return results

    def _split_chained_pattern(self, pattern: str) -> list[str]:
        """Splits `(, (A) (B))` into `['(A)', '(B)']`."""
        inner = pattern.strip()[2:-1].strip()
        if inner.startswith(","): inner = inner[1:].strip()
        forms = []
        current = ""
        depth = 0
        for ch in inner:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
                if depth == 0:
                    forms.append(current.strip())
                    current = ""
            elif depth == 0 and ch.isspace():
                continue
            else:
                current += ch
        if current.strip():
            forms.append(current.strip())
        return forms

    def _evaluate_chain(self, sub_patterns: list[str], bindings: dict[str, str]) -> list[list[str]]:
        if not sub_patterns:
            return [[]]
            
        current_pat = sub_patterns[0]
        # Substitute bindings
        tokens = self._tokenise(current_pat)
        resolved_tokens = [bindings.get(t, t) for t in tokens]
        current_pat_sub = " ".join(resolved_tokens)
            
        matches = self._base_query(current_pat_sub)
        results = []
        
        for match in matches:
            new_bindings = dict(bindings)
            pat_toks = self._tokenise(current_pat_sub)
            match_toks = self._tokenise(match)
            for p, m in zip(pat_toks, match_toks):
                if p.startswith("$"):
                    new_bindings[p] = m
                    
            rest_chains = self._evaluate_chain(sub_patterns[1:], new_bindings)
            for rc in rest_chains:
                results.append([match] + rc)
                
        return results

    def query(self, pattern: str) -> list[str]:
        pattern = pattern.strip()
        if pattern.startswith("(,") or pattern.startswith("( ,"):
            sub_patterns = self._split_chained_pattern(pattern)
            chains = self._evaluate_chain(sub_patterns, {})
            return [f"(, {' '.join(chain)})" for chain in chains]
        else:
            return self._base_query(pattern)

    def _tokenise(self, expr: str) -> list[str]:
        return expr.replace("(", " ( ").replace(")", " ) ").split()

    def _match(self, pattern_tokens: list[str], atom_tokens: list[str]) -> bool:
        if len(pattern_tokens) != len(atom_tokens):
            return False

        for p, a in zip(pattern_tokens, atom_tokens):
            if p.startswith("$"):
                continue
            if p == a:
                continue
            # Equivalence match: check if p and a are registered synonyms
            p_aliases = self.get_aliases(p)
            if a in p_aliases:
                continue
            return False

        return True


class _MockMeTTa:
    def __init__(self) -> None:
        self._space = _MockSpace()

    def run(self, program: str) -> list[list[Any]]:
        program = program.strip()
        results: list[list[Any]] = []

        for line in self._split_top_level(program):
            line = line.strip()
            if not line:
                continue
            r = self._exec_one(line)
            results.append(r)

        return results

    def space(self) -> _MockSpace:
        return self._space

    def _exec_one(self, line: str) -> list[Any]:
        # !(add-atom &self (...))
        m = re.match(r"^!\(\s*add-atom\s+&self\s+(.+)\)$", line, re.DOTALL)
        if m:
            atom_expr = m.group(1).strip()
            self._space.add_atom(atom_expr)
            return [atom_expr]

        # !(remove-atom &self (...))
        m = re.match(r"^!\(\s*remove-atom\s+&self\s+(.+)\)$", line, re.DOTALL)
        if m:
            atom_expr = m.group(1).strip()
            ok = self._space.remove_atom(atom_expr)
            return [ok]

        # !(match &self (pattern) (template))
        m = re.match(r"^!\(\s*match\s+&self\s+(.+?)\s+(\S+|\(.+?\))\s*\)$", line, re.DOTALL)
        if m:
            pattern = m.group(1).strip()
            template = m.group(2).strip()
            matches = self._space.query(pattern)
            if not matches:
                return []
            return self._extract_bindings(pattern, template, matches)

        # !(get-atoms &self)
        if re.match(r"^!\(\s*get-atoms\s+&self\s*\)$", line):
            return list(self._space.get_atoms())

        # Bare atom -> auto-add
        if line.startswith("(") and not line.startswith("!"):
            self._space.add_atom(line)
            return [line]

        return []

    def _extract_bindings(self, pattern: str, template: str, matches: list[str]) -> list[str]:
        pat_tokens = self._space._tokenise(pattern)
        results = []

        for atom in matches:
            atom_tokens = self._space._tokenise(atom)
            bindings: dict[str, str] = {}
            for p, a in zip(pat_tokens, atom_tokens):
                if p.startswith("$"):
                    bindings[p] = a

            result = template
            for var, val in bindings.items():
                result = result.replace(var, val)
            results.append(result)

        return results

    @staticmethod
    def _split_top_level(program: str) -> list[str]:
        forms: list[str] = []
        current = ""
        depth = 0
        i = 0
        while i < len(program):
            ch = program[i]
            if ch == "!" and depth == 0 and not current.strip():
                current += ch
            elif ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
                if depth == 0:
                    forms.append(current.strip())
                    current = ""
            else:
                current += ch
            i += 1
        remainder = current.strip()
        if remainder:
            forms.append(remainder)
        return forms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified MeTTa Engine Wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DerivationRecord:
    """Proof trace for a single inferred answer."""
    answer: str                   # fully-bound result atom
    rule_text: str                # the rule that fired (original text)
    premises: list[str]           # ground facts that satisfied the body
    origin: str = "derived"       # always "derived" for backward-chained results

@dataclass
class QueryResult:
    raw: list[Any]
    is_empty: bool
    formatted: str
    derivations: list[DerivationRecord] = field(default_factory=list)

    @staticmethod
    def from_raw(raw_results: list[list[Any]]) -> "QueryResult":
        flat = []
        for group in raw_results:
            for item in group:
                flat.append(str(item))

        return QueryResult(
            raw=flat,
            is_empty=len(flat) == 0,
            formatted=", ".join(flat) if flat else "∅ (no results)",
        )


def _tokenize_expr(expr: str) -> list[str]:
    tokens = []
    current = ""
    for ch in expr:
        if ch in "() \n\t":
            if current:
                tokens.append(current)
                current = ""
            if ch in "()":
                tokens.append(ch)
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens

def _parse_ast_expr(tokens: list[str]):
    if not tokens:
        return None, []
    t = tokens.pop(0)
    if t == "(":
        lst = []
        while tokens and tokens[0] != ")":
            item, tokens = _parse_ast_expr(tokens)
            lst.append(item)
        if tokens:
            tokens.pop(0)
        return lst, tokens
    elif t == ")":
        raise ValueError("Unexpected ')'")
    else:
        return t, tokens

def _ast_to_string(ast):
    if isinstance(ast, str):
        return ast
    return "(" + " ".join(_ast_to_string(child) for child in ast) + ")"

@dataclass
class MeTTaEngine:
    _metta: Any = field(init=False, repr=False)
    _using_mock: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._rules: list[str] = []  # Python-side rule registry (preserves variable names)
        if HYPERON_AVAILABLE:
            self._metta = _HyperonMeTTa()
            self._using_mock = False
            logger.info("Hyperon MeTTa engine initialised (native)")
        else:
            self._metta = _MockMeTTa()
            self._using_mock = True
            logger.info("Mock MeTTa engine initialised (dev mode)")

    @property
    def is_mock(self) -> bool:
        return self._using_mock

    def get_aliases(self, entity: str) -> set[str]:
        """Universal alias resolution using native graph queries."""
        aliases = {entity}
        changed = True
        while changed:
            changed = False
            # Check both directions of the equivalence
            for term in list(aliases):
                # Find what this term is equal to
                res1 = self.query(f"(= ({term}) $x)")
                if not res1.is_empty:
                    for match in res1.raw:
                        if match not in aliases:
                            aliases.add(match)
                            changed = True
                
                # Find what is equal to this term
                res2 = self.query(f"(= $x ({term}))")
                if not res2.is_empty:
                    for match in res2.raw:
                        if match not in aliases:
                            aliases.add(match)
                            changed = True
        return aliases

    def add_atom(self, expr: str) -> str:
        """Inject an atom, enforcing belief revision for single-valued predicates."""
        expr = sanitize_sexpr(expr)
        
        forms = self._split_forms(expr)
        if len(forms) > 1:
            for form in forms:
                self.add_atom(form)
            return expr

        tokens = expr.replace("(", " ( ").replace(")", " ) ").split()

        # Belief Revision Interception
        if len(tokens) >= 5 and tokens[0] == "(" and tokens[-1] == ")":
            pred = tokens[1]
            subject = tokens[2]
            
            if pred.lower() in SINGLE_VALUED_PREDICATES:
                aliases = self.get_aliases(subject)
                
                # Query the engine for existing conflicting atoms and retract them
                for alias in aliases:
                    # Query pattern: (Predicate Alias $val)
                    pattern = f"({pred} {alias} $val)"
                    existing = self.query(pattern, template="$val")
                    
                    if not existing.is_empty:
                        for old_val in existing.raw:
                            # Construct the exact old atom and remove it
                            old_atom = f"({pred} {alias} {old_val})"
                            self.remove_atom(old_atom)
                            logger.info("Middleware retracted outdated state: %s", old_atom)

        # AST Rule Normalization: accept both (= Head Body) and (Rule Head Body)
        rule_target = expr
        if expr.startswith("(Rule "):
            # Rewrite (Rule Head Body) -> (= Head Body)
            inner = expr[6:-1].strip() if expr.endswith(")") else expr[6:].strip()
            # Split into Head and Body AST
            tokens_rule = _tokenize_expr(expr)
            ast_rule, _ = _parse_ast_expr(tokens_rule)
            if isinstance(ast_rule, list) and len(ast_rule) == 3 and ast_rule[0] == "Rule":
                rule_target = f"(= {_ast_to_string(ast_rule[1])} {_ast_to_string(ast_rule[2])})"
                logger.info("Normalized semantic Rule to MeTTa: %s", rule_target)
                expr = rule_target

        # Register rules in Python-side registry (Hyperon v0.2.x strips variables from str())
        if expr.startswith("(= "):
            if expr not in self._rules:
                self._rules.append(expr)
                logger.info("Rule registered in BC registry: %s", expr)

        # Finally, execute the actual addition to the underlying engine
        cmd = f"!(add-atom &self {expr})"
        logger.debug("add_atom: %s", cmd)
        self._metta.run(cmd)
        return expr

    def add_equivalence(self, term_a: str, term_b: str) -> str:
        """Inject a bidirectional equivalence rule: (= (TermA) (TermB))."""
        expr = f"(= ({term_a}) ({term_b}))"
        return self.add_atom(expr)

    def remove_atom(self, expr: str) -> bool:
        expr = sanitize_sexpr(expr)
        # Remove from rule registry if present
        if expr in self._rules:
            self._rules.remove(expr)
        cmd = f"!(remove-atom &self {expr})"
        result = self._metta.run(cmd)
        return bool(result)

    def query(self, pattern: str, template: str | None = None) -> QueryResult:
        pattern = sanitize_sexpr(pattern)
        if template is None:
            template = "$result"
        cmd = f"!(match &self {pattern} {template})"
        raw = self._metta.run(cmd)
        return QueryResult.from_raw(raw)

    def run_raw(self, program: str) -> list[list[Any]]:
        program = sanitize_sexpr(program)
        return self._metta.run(program)

    def load_program(self, program: str) -> None:
        for form in self._split_forms(program):
            form = form.strip()
            if not form or form.startswith(";"):
                continue
            try:
                sanitize_sexpr(form)
                self._metta.run(form)
            except MeTTaSyntaxError as e:
                logger.warning("Skipping malformed form on load: %s — %s", form, e)

    def get_all_atoms(self) -> list[str]:
        if self._using_mock:
            return self._metta.space().get_atoms()
        try:
            results = self._metta.run("!(get-atoms &self)")
            atoms = []
            for group in results:
                for atom in group:
                    # Strip Hyperon's internal scope IDs (e.g. $var#1234 -> $var) to prevent diff noise
                    cleaned_atom = re.sub(r"(\$[a-zA-Z0-9_]+)#\d+", r"\1", str(atom))
                    atoms.append(cleaned_atom)
            return atoms
        except Exception as e:
            logger.warning("get-atoms failed (%s), falling back to empty", e)
            return []

    def get_state_snapshot(self) -> str:
        atoms = sorted(self.get_all_atoms())
        return "\n".join(atoms)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Python-Level Backward Chainer
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def backward_chain(self, goal: str) -> QueryResult:
        """
        Backward chaining inference engine via AST unification.
        Handles shared bindings across AND clauses and full backtracking.
        """
        goal = sanitize_sexpr(goal)
        
        def tokenize(expr: str) -> list[str]:
            tokens = []
            current = ""
            for ch in expr:
                if ch in "() \n\t":
                    if current:
                        tokens.append(current)
                        current = ""
                    if ch in "()":
                        tokens.append(ch)
                else:
                    current += ch
            if current:
                tokens.append(current)
            return tokens

        def parse_ast(tokens: list[str]):
            if not tokens:
                return None, []
            t = tokens.pop(0)
            if t == '(':
                lst = []
                while tokens and tokens[0] != ')':
                    item, tokens = parse_ast(tokens)
                    lst.append(item)
                if tokens:
                    tokens.pop(0)
                return lst, tokens
            elif t == ')':
                raise ValueError("Unexpected ')'")
            else:
                return t, tokens

        def substitute(ast, bindings):
            if isinstance(ast, str) and ast.startswith('$'):
                if ast in bindings:
                    return substitute(bindings[ast], bindings)
                return ast
            if isinstance(ast, list):
                return [substitute(child, bindings) for child in ast]
            return ast

        def to_string(ast):
            if isinstance(ast, str):
                return ast
            return "(" + " ".join(to_string(child) for child in ast) + ")"

        def unify(ast1, ast2, bindings):
            bindings = bindings.copy()
            if isinstance(ast1, str) and ast1.startswith('$'):
                if ast1 in bindings:
                    return unify(bindings[ast1], ast2, bindings)
                else:
                    bindings[ast1] = ast2
                    return bindings
            if isinstance(ast2, str) and ast2.startswith('$'):
                if ast2 in bindings:
                    return unify(ast1, bindings[ast2], bindings)
                else:
                    bindings[ast2] = ast1
                    return bindings
            
            if isinstance(ast1, str) and isinstance(ast2, str):
                if ast1 == ast2:
                    return bindings
                return None
            
            if isinstance(ast1, list) and isinstance(ast2, list):
                if len(ast1) != len(ast2):
                    return None
                for c1, c2 in zip(ast1, ast2):
                    bindings = unify(c1, c2, bindings)
                    if bindings is None:
                        return None
                return bindings
            return None

        # Parse rules and facts
        facts = []
        rules = []
        
        # Facts come from the engine
        for atom in self.get_all_atoms():
            ast, _ = _parse_ast_expr(_tokenize_expr(atom))
            if isinstance(ast, list) and len(ast) == 3 and ast[0] == "=":
                # Engine might strip variables, so we ignore rules here
                pass
            elif ast is not None:
                facts.append(ast)
                # Unwrap base fact from temporal wrapper (ValidDate BaseFact Date)
                if isinstance(ast, list) and len(ast) == 3 and ast[0] in ("ValidDate", "Recurrence") and isinstance(ast[1], list):
                    facts.append(ast[1])
                
        # Rules come from the Python registry to preserve variables
        for rule in getattr(self, "_rules", []):
            ast, _ = parse_ast(tokenize(rule))
            if isinstance(ast, list) and len(ast) == 3 and ast[0] == "=":
                rules.append((ast[1], ast[2]))
        # Keep original rule text for derivation records
        rule_texts = list(getattr(self, "_rules", []))

        def solve_and(goals, bindings, depth, trace):
            if not goals:
                yield bindings, trace
                return
            first = goals[0]
            rest = goals[1:]
            for new_bindings, new_trace in solve(first, bindings, depth, trace):
                yield from solve_and(rest, new_bindings, depth, new_trace)

        def solve(goal_ast, bindings, depth, trace):
            if depth > 10:
                return
            goal_ast = substitute(goal_ast, bindings)

            # AND Logic
            if isinstance(goal_ast, list) and len(goal_ast) > 0 and goal_ast[0] in (',', 'and'):
                yield from solve_and(goal_ast[1:], bindings, depth, trace)
                return

            # Temporal Resolution Logic (Stage D)
            if isinstance(goal_ast, list) and len(goal_ast) == 3 and goal_ast[0] == 'ValidDate':
                base_fact_ast = goal_ast[1]
                date_ast = goal_ast[2]
                
                # 1. Check for Exceptions where alt_fact unifies with our query fact
                # (Exception $orig_fact "YYYY-MM-DD" $query_fact)
                exc_goal = ['Exception', '$orig_fact_var', date_ast, base_fact_ast]
                for exc_bindings, exc_trace in solve(exc_goal, bindings, depth + 1, trace):
                    yield exc_bindings, exc_trace + [("rule", "Temporal Exception Override")]
                    
                # 2. Check for Recurrences
                if isinstance(date_ast, str) and not date_ast.startswith('$'):
                    date_str = date_ast.strip('"')
                    try:
                        import datetime
                        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        day_name = dt.strftime("%A")
                        
                        rec_goal = ['Recurrence', base_fact_ast, 'Weekly', day_name]
                        for rec_bindings, rec_trace in solve(rec_goal, bindings, depth + 1, trace):
                            # Ensure no exception overrides this specific recurrence date
                            ground_base_fact = substitute(base_fact_ast, rec_bindings)
                            conflict_goal = ['Exception', ground_base_fact, date_ast, '$any_alt']
                            has_exception = False
                            for _ in solve(conflict_goal, rec_bindings, depth + 1, []):
                                has_exception = True
                                break
                                
                            if not has_exception:
                                yield rec_bindings, rec_trace + [("rule", "Temporal Recurrence Resolution")]
                    except ValueError:
                        pass

            # Literal facts
            for fact in facts:
                new_bindings = unify(goal_ast, fact, bindings)
                if new_bindings is not None:
                    ground_atom = to_string(substitute(goal_ast, new_bindings))
                    yield new_bindings, trace + [("fact", ground_atom)]

            # Rules
            for idx, (rule_head, rule_body) in enumerate(rules):
                def refresh(ast, _depth=depth):
                    if isinstance(ast, str) and ast.startswith('$'):
                        return f"{ast}_{_depth}"
                    if isinstance(ast, list):
                        return [refresh(c, _depth) for c in ast]
                    return ast
                r_head = refresh(rule_head)
                r_body = refresh(rule_body)

                new_bindings = unify(goal_ast, r_head, bindings)
                if new_bindings is not None:
                    rule_src = rule_texts[idx] if idx < len(rule_texts) else "<unknown rule>"
                    for final_bindings, sub_trace in solve(r_body, new_bindings, depth + 1, trace):
                        yield final_bindings, sub_trace + [("rule", rule_src)]

        goal_ast, _ = parse_ast(tokenize(goal))
        results = []
        derivation_records = []
        seen = set()
        for b, trace in solve(goal_ast, {}, 0, []):
            res_ast = substitute(goal_ast, b)
            answer_str = to_string(res_ast)
            if answer_str in seen:
                continue
            seen.add(answer_str)
            results.append(answer_str)

            # Build derivation record from trace
            premise_atoms = [item for kind, item in trace if kind == "fact"]
            applied_rules = [item for kind, item in trace if kind == "rule"]
            if applied_rules:
                derivation_records.append(DerivationRecord(
                    answer=answer_str,
                    rule_text=applied_rules[0],
                    premises=premise_atoms,
                    origin="derived"
                ))

        return QueryResult(
            raw=results,
            is_empty=len(results) == 0,
            formatted=", ".join(results) if results else "\u2205 (no results)",
            derivations=derivation_records
        )
    @staticmethod
    def _split_forms(program: str) -> list[str]:
        # Strip MeTTa comments before parsing to prevent ghost word injection
        cleaned_lines = []
        for line in program.split("\n"):
            stripped = line.strip()
            if stripped.startswith(";"):
                continue  # skip comment lines entirely
            cleaned_lines.append(line)
        program = "\n".join(cleaned_lines)

        forms: list[str] = []
        current = ""
        depth = 0
        for ch in program:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
                if depth == 0:
                    forms.append(current.strip())
                    current = ""
            elif ch == "\n" and depth == 0:
                stripped = current.strip()
                if stripped:
                    forms.append(stripped)
                current = ""
            else:
                current += ch
        remainder = current.strip()
        if remainder:
            forms.append(remainder)
        return forms

