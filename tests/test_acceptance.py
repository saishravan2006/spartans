"""
tests/test_acceptance.py
Deterministic acceptance tests for Super Memory V3 Agent.
Tests the actual execution path: backward_chain, catalogue, belief revision,
temporal resolution, and historical ledger queries.
All tests use the MeTTa engine directly without LLM calls.
"""
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metta_engine import MeTTaEngine
from core.catalogue import EntityCatalogue
from core.persistence import PersistenceManager

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  OK {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: {detail}")


# --- Test 1: Backward Chainer - Mismatched clearance ---
print("\n=== TEST 1: Mismatched clearance -> no proof ===")
engine = MeTTaEngine()
engine.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
engine.add_atom("(RequiresClearance LabA Level3)")
engine.add_atom("(HasClearance Anush Level2)")
result = engine.backward_chain("(CanAccess Anush LabA)")
check("Mismatch Level3/Level2 -> empty", result.is_empty, f"Got: {result.raw}")


# --- Test 2: Backward Chainer - Matching clearance ---
print("\n=== TEST 2: Matching clearance -> proof ===")
engine2 = MeTTaEngine()
engine2.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
engine2.add_atom("(RequiresClearance LabA Level2)")
engine2.add_atom("(HasClearance Anush Level2)")
result2 = engine2.backward_chain("(CanAccess Anush LabA)")
check("Match Level2/Level2 -> non-empty", not result2.is_empty, f"Got: {result2.raw}")
if result2.derivations:
    check("Derivation has premises", len(result2.derivations[0].premises) > 0)


# --- Test 3: Belief Revision - Single-valued predicates ---
print("\n=== TEST 3: Belief Revision - HeadOfDepartment ===")
engine3 = MeTTaEngine()
engine3.add_atom("(HeadOfDepartment Cybersecurity MarcusSterling)")
check("Initial head", "(HeadOfDepartment Cybersecurity MarcusSterling)" in engine3.get_all_atoms())
engine3.add_atom("(HeadOfDepartment Cybersecurity NadiaVolkov)")
atoms3 = engine3.get_all_atoms()
check("New head present", "(HeadOfDepartment Cybersecurity NadiaVolkov)" in atoms3)
check("Old head removed", "(HeadOfDepartment Cybersecurity MarcusSterling)" not in atoms3, f"Still present: {atoms3}")


# --- Test 4: Catalogue - Entity Discovery & Search ---
print("\n=== TEST 4: Catalogue entity sync and search ===")
engine4 = MeTTaEngine()
engine4.add_atom("(HeadOfDepartment Cybersecurity DrNadiaVolkov)")
engine4.add_atom("(Member RoboticsClub AlexTurner)")
cat4 = EntityCatalogue()
cat4.sync(engine4.get_all_atoms())
results_nadia = cat4.search("Nadia")
check("Search finds DrNadiaVolkov", any(c.canonical_id == "DrNadiaVolkov" for c in results_nadia))
results_alex = cat4.search("Alex")
check("Search finds AlexTurner", any(c.canonical_id == "AlexTurner" for c in results_alex))


# --- Test 5: Symmetric Rule (Collaborator) ---
print("\n=== TEST 5: Symmetric rule backward chaining ===")
engine5 = MeTTaEngine()
engine5.add_atom("(= (CollaboratorOf $B $A) (CollaboratorOf $A $B))")
engine5.add_atom("(CollaboratorOf AgentSmith AgentJones)")
res5 = engine5.backward_chain("(CollaboratorOf AgentJones AgentSmith)")
check("Symmetric deduction succeeds", not res5.is_empty, f"Got: {res5.raw}")
res5_neg = engine5.backward_chain("(CollaboratorOf AgentSmith AgentVance)")
check("Unprovable query returns empty", res5_neg.is_empty, f"Got: {res5_neg.raw}")


# --- Test 6: Temporal - Recurrence & Exceptions ---
# Oct 16, 2026 = Friday, Oct 23, 2026 = Friday, Oct 10 = Saturday
print("\n=== TEST 6: Temporal recurrence with exception override ===")
engine6 = MeTTaEngine()
engine6.add_atom("(Recurrence (MeetingPlace RoboticsClub Lab5) Weekly Friday)")
engine6.add_atom("(Exception (MeetingPlace RoboticsClub Lab5) 2026-10-16 (MeetingPlace RoboticsClub RoomB))")

# Oct 16 is Friday with exception -> should get RoomB
res6_exc = engine6.backward_chain("(ValidDate (MeetingPlace RoboticsClub RoomB) 2026-10-16)")
check("Exception override on Oct 16 (Friday)", not res6_exc.is_empty, f"Got: {res6_exc.raw}")

# Oct 23 is also Friday - no exception, normal recurrence
res6_normal = engine6.backward_chain("(ValidDate (MeetingPlace RoboticsClub Lab5) 2026-10-23)")
check("Normal recurrence on Oct 23 (Friday)", not res6_normal.is_empty, f"Got: {res6_normal.raw}")

# Saturday should NOT match weekly Friday recurrence
res6_sat = engine6.backward_chain("(ValidDate (MeetingPlace RoboticsClub Lab5) 2026-10-10)")
check("Saturday Oct 10 -> no match", res6_sat.is_empty, f"Got: {res6_sat.raw}")


# --- Test 7: Persistence - Evidence Ledger & History ---
print("\n=== TEST 7: Persistence and historical query ===")
test_dir = Path("test_data_acceptance")
if test_dir.exists():
    shutil.rmtree(test_dir, ignore_errors=True)
test_dir.mkdir(exist_ok=True)

engine7 = MeTTaEngine()
pm7 = PersistenceManager(data_dir=test_dir)

uid1 = pm7.record_utterance("Marcus Sterling is Head of Cybersecurity.")
engine7.add_atom("(HeadOfDepartment Cybersecurity MarcusSterling)")
pm7.record_modification(engine7, uid1, "(HeadOfDepartment Cybersecurity MarcusSterling)", "add")

uid2 = pm7.record_utterance("Sterling stepped down, Nadia took over.")
engine7.remove_atom("(HeadOfDepartment Cybersecurity MarcusSterling)")
pm7.record_modification(engine7, uid2, "(HeadOfDepartment Cybersecurity MarcusSterling)", "remove")
engine7.add_atom("(HeadOfDepartment Cybersecurity NadiaVolkov)")
pm7.record_modification(engine7, uid2, "(HeadOfDepartment Cybersecurity NadiaVolkov)", "add")

atoms7 = engine7.get_all_atoms()
check("Active state: only Nadia", "(HeadOfDepartment Cybersecurity NadiaVolkov)" in atoms7)
check("Active state: no Marcus", "(HeadOfDepartment Cybersecurity MarcusSterling)" not in atoms7)

history = pm7.query_history(entity_or_keyword="MarcusSterling")
check("Historical ledger has Marcus records", len(history) > 0, f"Got: {history}")
has_remove = any(h["operation"] == "remove" for h in history)
check("Historical ledger shows removal operation", has_remove)

# Close DB connection before cleanup
del pm7
shutil.rmtree(test_dir, ignore_errors=True)


# --- Test 8: Multi-hop backward chaining (3 conjuncts) ---
print("\n=== TEST 8: Multi-hop backward chaining ===")
engine8 = MeTTaEngine()
engine8.add_atom("(= (CanEnter $x QuantumVault) (and (SecurityClearance $x TopSecret) (CompletedTraining $x BioResilience) (EquippedWith $x BiometricPass)))")
engine8.add_atom("(SecurityClearance OperativeAlpha TopSecret)")
engine8.add_atom("(CompletedTraining OperativeAlpha BioResilience)")
engine8.add_atom("(EquippedWith OperativeAlpha BiometricPass)")
res8 = engine8.backward_chain("(CanEnter OperativeAlpha QuantumVault)")
check("3-conjunct multi-hop succeeds", not res8.is_empty, f"Got: {res8.raw}")

engine8b = MeTTaEngine()
engine8b.add_atom("(= (CanEnter $x QuantumVault) (and (SecurityClearance $x TopSecret) (CompletedTraining $x BioResilience) (EquippedWith $x BiometricPass)))")
engine8b.add_atom("(SecurityClearance OperativeAlpha TopSecret)")
engine8b.add_atom("(CompletedTraining OperativeAlpha BioResilience)")
res8b = engine8b.backward_chain("(CanEnter OperativeAlpha QuantumVault)")
check("Missing conjunct -> empty", res8b.is_empty, f"Got: {res8b.raw}")


# --- Summary ---
print(f"\n{'='*50}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} total")
print(f"{'='*50}")

if FAIL > 0:
    sys.exit(1)
