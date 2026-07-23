"""Static, provable safety invariants — not documentation claims, actual
checks that fail the build if violated. `safety/governor.py`'s docstring
asserts "nothing in the learning/consolidation loop is permitted to
import or write to this file's rules." This module verifies that
mechanically, by parsing the actual source (via `ast`, not just grepping
text) rather than trusting the convention to hold as the codebase grows.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "machine_brain"

# Modules whose behavior must never be reachable from a learning/consolidation
# code path — the actual hard limits, not the guard *orchestration* around them.
PROTECTED_MODULES = {"machine_brain.safety.governor"}

# Modules that implement learning/consolidation and must not import the
# protected modules above.
LEARNING_MODULES = [SRC / "learning" / "reviewed_learning.py"]


def _imported_module_names(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_learning_modules_never_import_the_safety_governor():
    for module_path in LEARNING_MODULES:
        imported = _imported_module_names(module_path)
        overlap = imported & PROTECTED_MODULES
        assert not overlap, f"{module_path} imports protected safety module(s): {overlap}"


def test_safety_envelope_has_no_public_mutator_method():
    """The envelope's fields must only ever be set at construction time —
    no method on SafetyGovernor or SafetyEnvelope may reassign an envelope
    field after creation. Checked by parsing the class body for any
    `self.envelope.<field> = ...` or `self.<field> = ...` assignment
    outside `__init__`/dataclass field defaults."""
    governor_file = SRC / "safety" / "governor.py"
    tree = ast.parse(governor_file.read_text(), filename=str(governor_file))

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Attribute) and (
                            (isinstance(target.value, ast.Name) and target.value.id == "self")
                            or (isinstance(target.value, ast.Attribute) and target.attr in
                                 ("max_velocity", "min_human_distance", "forbidden_zones", "forbidden_skills"))
                        ):
                            violations.append(f"{node.name}: assigns {ast.dump(target)}")
    assert not violations, f"safety envelope mutated outside construction: {violations}"


def test_reviewed_learning_process_never_calls_into_safety():
    """Belt-and-suspenders on top of the import check: scan actual code
    identifiers (not docstrings/comments/string literals — a naive text
    search over the whole file false-positived on this exact module's own
    docstring explaining the invariant, on the first version of this test)
    for any reference to SafetyGovernor/SafetyEnvelope, in case a future
    edit passes one in as a parameter instead of importing the module
    directly."""
    learning_file = SRC / "learning" / "reviewed_learning.py"
    tree = ast.parse(learning_file.read_text(), filename=str(learning_file))
    forbidden_names = {"SafetyGovernor", "SafetyEnvelope"}
    referenced = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id in forbidden_names
    } | {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names
    }
    assert not referenced, f"reviewed_learning.py references forbidden safety identifiers: {referenced}"
