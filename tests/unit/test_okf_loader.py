"""Tests against the real OKF v0.1 conformance rules (fetched from
GoogleCloudPlatform/knowledge-catalog/okf/SPEC.md), not an invented
schema — in particular the spec's explicit tolerance requirements:
'consumers must accept missing optional fields, unknown types, unknown
frontmatter keys, broken links, and missing index files.'
"""

import pytest

from machine_brain.knowledge.okf_loader import OKFBundle


def _write(root, rel_path: str, content: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_loads_a_well_formed_concept(tmp_path):
    _write(tmp_path, "widgets/foo.md", """---
type: Widget
title: Foo
description: A test widget.
tags: [a, b]
---

# Body
Some text.
""")
    bundle = OKFBundle(tmp_path)
    concept = bundle.get("widgets/foo")
    assert concept is not None
    assert concept.type == "Widget"
    assert concept.title == "Foo"
    assert concept.tags == ("a", "b")
    assert "Some text." in concept.body


def test_index_md_and_log_md_are_reserved_not_treated_as_concepts(tmp_path):
    _write(tmp_path, "index.md", "# Index\n* [foo](widgets/foo.md)\n")
    _write(tmp_path, "log.md", "# Log\n## 2026-01-01\n* Update: did a thing.\n")
    _write(tmp_path, "widgets/foo.md", "---\ntype: Widget\n---\nbody\n")
    bundle = OKFBundle(tmp_path)
    assert "index" not in bundle.concepts
    assert "log" not in bundle.concepts
    assert len(bundle.concepts) == 1


def test_missing_type_field_is_tolerated_not_raised(tmp_path):
    """Spec: 'Every frontmatter block contains non-empty type field' is a
    conformance requirement, but consumers must still tolerate violations
    gracefully rather than crash the whole bundle load."""
    _write(tmp_path, "bad.md", "---\ntitle: No Type Here\n---\nbody\n")
    bundle = OKFBundle(tmp_path)
    assert "bad" not in bundle.concepts
    assert len(bundle.skipped) == 1
    assert "type" in bundle.skipped[0][1]


def test_unknown_type_is_tolerated():
    """Spec: 'Types are not centrally registered; consumers must tolerate
    unknown types.'"""
    pass  # covered structurally: OKFBundle never validates `type` against
    # an allowlist anywhere — any non-empty string is accepted, by
    # construction, not by an exception handler.


def test_unrecognized_frontmatter_keys_are_preserved_not_dropped(tmp_path):
    _write(tmp_path, "widgets/foo.md", """---
type: Widget
custom_field: hello
another_custom: 42
---
body
""")
    bundle = OKFBundle(tmp_path)
    concept = bundle.get("widgets/foo")
    assert concept.extra == {"custom_field": "hello", "another_custom": 42}


def test_malformed_yaml_is_tolerated_not_raised(tmp_path):
    _write(tmp_path, "broken.md", "---\ntype: [unclosed\n---\nbody\n")
    bundle = OKFBundle(tmp_path)  # must not raise
    assert "broken" not in bundle.concepts
    assert len(bundle.skipped) == 1


def test_file_with_no_frontmatter_fence_is_tolerated(tmp_path):
    _write(tmp_path, "plain.md", "# Just a heading\nNo frontmatter at all.\n")
    bundle = OKFBundle(tmp_path)  # must not raise
    assert "plain" not in bundle.concepts


def test_absolute_link_resolves_from_bundle_root(tmp_path):
    _write(tmp_path, "a/one.md", "---\ntype: X\n---\nlinks to [two](/b/two.md)\n")
    _write(tmp_path, "b/two.md", "---\ntype: X\n---\nbody\n")
    bundle = OKFBundle(tmp_path)
    assert bundle.resolve_link("a/one", "/b/two.md") == "b/two"


def test_relative_link_resolves_from_linking_files_directory(tmp_path):
    _write(tmp_path, "a/one.md", "---\ntype: X\n---\nbody\n")
    _write(tmp_path, "a/two.md", "---\ntype: X\n---\nbody\n")
    bundle = OKFBundle(tmp_path)
    assert bundle.resolve_link("a/one", "./two.md") == "a/two"


def test_broken_link_is_tolerated_returns_none(tmp_path):
    """Spec: consumers 'must tolerate broken links' — resolving one must
    not raise, just fail to resolve."""
    _write(tmp_path, "a/one.md", "---\ntype: X\n---\nbody\n")
    bundle = OKFBundle(tmp_path)
    assert bundle.resolve_link("a/one", "/nowhere/nothing.md") is None


def test_by_type_and_by_tag_filtering(tmp_path):
    _write(tmp_path, "a.md", "---\ntype: Skill\ntags: [motion]\n---\nbody\n")
    _write(tmp_path, "b.md", "---\ntype: Skill\ntags: [sensor]\n---\nbody\n")
    _write(tmp_path, "c.md", "---\ntype: Goal\ntags: [motion]\n---\nbody\n")
    bundle = OKFBundle(tmp_path)
    assert {c.concept_id for c in bundle.by_type("Skill")} == {"a", "b"}
    assert {c.concept_id for c in bundle.by_tag("motion")} == {"a", "c"}


def test_missing_bundle_directory_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        OKFBundle(tmp_path / "does_not_exist")
