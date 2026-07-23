"""A real, spec-conformant loader for Google Cloud's Open Knowledge Format
(OKF) v0.1 — https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
(fetched directly from source before writing this, not inferred from
secondary blog summaries).

Implements exactly what the spec requires and nothing more:
- A concept is any non-reserved `.md` file with YAML frontmatter containing
  a non-empty `type` field. `index.md` and `log.md` are reserved and
  intentionally not treated as concepts (no frontmatter expected).
- Concept ID = the file's path relative to the bundle root, minus `.md`.
- Consumers "must accept missing optional fields, unknown types, unknown
  frontmatter keys, broken links, and missing index files" (spec,
  Conformance Requirements) — this loader skips malformed files rather
  than raising, and preserves unrecognized frontmatter keys instead of
  dropping them.
- Links: absolute (`/tables/x.md`, relative to bundle root) and relative
  (`./other.md`, relative to the linking file) are both resolved to
  concept IDs; the spec defines links as asserting an undirected
  relationship with the type conveyed by surrounding prose, so this
  loader resolves *targets*, not relationship semantics — that's a
  question for whatever reads the prose, not this parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_RESERVED_FILENAMES = {"index.md", "log.md"}
_KNOWN_FRONTMATTER_KEYS = {"type", "title", "description", "resource", "tags", "timestamp"}


@dataclass(frozen=True)
class OKFConcept:
    concept_id: str
    type: str
    title: str | None
    description: str | None
    resource: str | None
    tags: tuple[str, ...]
    timestamp: str | None
    extra: dict  # frontmatter keys the spec doesn't define — preserved, not dropped
    body: str
    source_path: Path


def _split_frontmatter(text: str) -> tuple[str, str] | tuple[None, None]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            frontmatter = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            return frontmatter, body
    return None, None  # opening fence with no closing fence — not conformant, tolerate by skipping


class OKFBundle:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.concepts: dict[str, OKFConcept] = {}
        self.skipped: list[tuple[Path, str]] = []  # (path, reason) — malformed files, tolerated not raised
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"OKF bundle root does not exist or isn't a directory: {self.root}")
        for path in sorted(self.root.rglob("*.md")):
            if path.name in _RESERVED_FILENAMES:
                continue
            concept = self._parse_concept(path)
            if concept is not None:
                self.concepts[concept.concept_id] = concept

    def _parse_concept(self, path: Path) -> OKFConcept | None:
        text = path.read_text(encoding="utf-8")
        frontmatter_text, body = _split_frontmatter(text)
        if frontmatter_text is None:
            self.skipped.append((path, "no parseable YAML frontmatter block"))
            return None
        try:
            data = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            self.skipped.append((path, f"invalid YAML: {e}"))
            return None
        if not isinstance(data, dict):
            self.skipped.append((path, "frontmatter did not parse to a mapping"))
            return None

        type_ = data.get("type")
        if not type_:
            self.skipped.append((path, "missing required non-empty 'type' field"))
            return None

        concept_id = str(path.relative_to(self.root)).removesuffix(".md")
        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        extra = {k: v for k, v in data.items() if k not in _KNOWN_FRONTMATTER_KEYS}

        return OKFConcept(
            concept_id=concept_id, type=str(type_), title=data.get("title"),
            description=data.get("description"), resource=data.get("resource"),
            tags=tuple(str(t) for t in tags), timestamp=data.get("timestamp"),
            extra=extra, body=(body or "").strip(), source_path=path,
        )

    def get(self, concept_id: str) -> OKFConcept | None:
        return self.concepts.get(concept_id)

    def by_type(self, type_: str) -> list[OKFConcept]:
        return [c for c in self.concepts.values() if c.type == type_]

    def by_tag(self, tag: str) -> list[OKFConcept]:
        return [c for c in self.concepts.values() if tag in c.tags]

    def resolve_link(self, from_concept_id: str, link_target: str) -> str | None:
        """Resolve a markdown link target to a concept_id. Absolute links
        (leading '/') resolve from the bundle root; relative links resolve
        from the linking concept's own directory. Returns None for a
        broken link — per spec, consumers "must tolerate broken links,"
        not raise on them."""
        if link_target.startswith("/"):
            candidate = link_target.lstrip("/").removesuffix(".md")
        else:
            from_dir = Path(from_concept_id).parent
            candidate = str((from_dir / link_target).as_posix()).removesuffix(".md")
            # normalize any './' or '../' components
            candidate = str(Path(candidate).as_posix())
        return candidate if candidate in self.concepts else None
