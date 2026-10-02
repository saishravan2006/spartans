import re
from typing import Optional

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

class Candidate:
    def __init__(self, canonical_id: str, is_predicate: bool = False):
        self.canonical_id = canonical_id
        self.is_predicate = is_predicate
        self.aliases = set()
        
        # Display label: "AISecurityHackathon" -> "AI Security Hackathon"
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1 \2', canonical_id)
        self.display_label = re.sub(r"([a-z\d])([A-Z])", r'\1 \2', s).replace("_", " ")

class EntityCatalogue:
    def __init__(self):
        self.candidates: dict[str, Candidate] = {}
        # Maps lowercase string (alias or canonical) to canonical IDs
        self._lookup: dict[str, set[str]] = {}

    def sync(self, atoms: list[str]):
        """Rebuilds the catalogue from the current AtomSpace via universal AST walking."""
        self.candidates = {}
        self._lookup = {}

        INTERNAL_OPS = {"=", ":", "and", "or", ",", "Rule", "ValidDate", "Recurrence", "Exception", "SchemaDefinition"}

        def walk(node):
            if isinstance(node, list) and node:
                fn = node[0]
                if isinstance(fn, str) and not fn.startswith("$"):
                    if fn == "SchemaDefinition" and len(node) > 1 and isinstance(node[1], str):
                        self._add_predicate(node[1])
                    elif fn not in INTERNAL_OPS:
                        self._add_predicate(fn)
                for child in node[1:]:
                    walk(child)
            elif isinstance(node, str):
                # Universal entity symbol: non-variable, non-quoted, non-operator
                if not node.startswith("$") and not (node.startswith('\"') or node.startswith("'")) and node not in INTERNAL_OPS:
                    self._add_entity(node)

        for atom in atoms:
            try:
                ast, _ = parse_ast(tokenize(atom))
                walk(ast)
            except Exception:
                pass

        # Second pass: attach aliases
        for atom in atoms:
            try:
                ast, _ = parse_ast(tokenize(atom))
                if isinstance(ast, list) and len(ast) == 3 and ast[0] == "=":
                    head = ast[1]
                    if isinstance(head, list) and len(head) == 2 and head[0] == "Alias":
                        alias = head[1]
                        canonical = ast[2]
                        if isinstance(canonical, str) and canonical in self.candidates:
                            self.candidates[canonical].aliases.add(alias)
                            self._add_lookup(alias, canonical)
            except Exception:
                pass

    def _add_predicate(self, pred: str):
        if pred not in self.candidates:
            self.candidates[pred] = Candidate(pred, is_predicate=True)
            self._add_lookup(pred, pred)

    def _add_entity(self, entity: str):
        if entity not in self.candidates:
            self.candidates[entity] = Candidate(entity, is_predicate=False)
            self._add_lookup(entity, entity)

    def _add_lookup(self, key: str, canonical: str):
        lower_key = key.lower()
        if lower_key not in self._lookup:
            self._lookup[lower_key] = set()
        self._lookup[lower_key].add(canonical)

    def search(self, text: str, entities_only: bool = True) -> list[Candidate]:
        """Find candidates relevant to the text. Splits PascalCase into words for matching.
        FIX 3: entities_only=True by default — predicates are not returned by search_entities.
        This prevents predicate names from being confused as entity candidates in clarification.
        """
        words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
        
        matches = set()
        for lower_key, canonicals in self._lookup.items():
            key_parts = set(re.sub(r'([a-z])([A-Z])', r'\1 \2', lower_key).replace('_', ' ').lower().split())
            key_parts.add(lower_key)
            
            if key_parts.intersection(words) or any(w in lower_key for w in words if len(w) > 3):
                matches.update(canonicals)
                
        results = [self.candidates[c] for c in matches if c in self.candidates]
        if entities_only:
            results = [c for c in results if not c.is_predicate]
        return results

    def resolve_exact(self, name: str) -> tuple[Optional[str], Optional[list[str]]]:
        """
        Attempts to resolve an exact entity name.
        Returns: (canonical_id, None) if unique.
                 (None, list_of_collisions) if ambiguous.
                 (None, None) if unknown.
        """
        lower = name.lower()
        if lower in self._lookup:
            canonicals = list(self._lookup[lower])
            if len(canonicals) == 1:
                return canonicals[0], None
            else:
                return None, canonicals
        return None, None
