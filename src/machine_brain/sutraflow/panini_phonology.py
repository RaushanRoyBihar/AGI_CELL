"""Real Pāṇinian phonology, built on two data files pulled verbatim from
the `vajra-v0.39-UNIFIED` donor archive
(`vajra/sutras/maheshvara_shiva_sutras.json`,
`vajra/sutras/ashtadhyayi_legacy_executable_seed.json` — copied to
`sutraflow/data/`, unmodified). See `docs/provenance/NOTES.md` for the
full accounting.

Unlike the rest of the donor tree, this data is straightforwardly
authentic: the 14 Māheśvara/Śiva Sūtras are the real phoneme-classification
sutras that open the Aṣṭādhyāyī, and the sandhi entries carry real,
verifiable canonical sutra numbers (6.1.87, 6.1.88, 6.1.89, 6.1.101). The
donor JSON is pure data (id, sutra text, canonical id, description) — no
donor *code* is used here. The pratyāhāra-construction algorithm and the
sandhi transformation logic below are written fresh against the classical
definition, not extracted from the donor's own (unseen, unused) Python
implementation.

Scope, stated plainly: this implements the standard pratyāhāra
construction (concatenate the 14 sutras' phonemes and IT-markers in
order; a pratyāhāra name is "first occurrence of the start phoneme" up to
"the next occurrence of the end IT-marker", returned as a set) and four
specific, textbook vowel-sandhi rules cited by their real canonical sutra
numbers. It does not implement the full traditional edge-case machinery
for repeated phonemes in less common pratyāhāras (only "ha", which
repeats in sutras 5 and 14, is relevant to the two pratyāhāras tested
here, and the simple algorithm happens to produce the textbook-correct
result for both — verified in tests, not assumed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class SutraRecord:
    id: str
    canonical_id: str | None
    name: str
    family: str
    description: str
    sutra_text: str | None


def load_catalog() -> list[SutraRecord]:
    """All entries from both donor JSON files, as inspectable metadata.
    Only the phonetic (Māheśvara) and vowel-sandhi entries get real
    executable logic below — everything else is catalogued honestly as
    reference data, not silently treated as active rules."""
    records = []
    for filename in ("maheshvara_shiva_sutras.json", "ashtadhyayi_legacy_executable_seed.json"):
        data = json.loads((_DATA_DIR / filename).read_text())
        for r in data["rules"]:
            records.append(SutraRecord(
                id=r["id"], canonical_id=r.get("canonical_id"), name=r["name"],
                family=r["family"], description=r["description"], sutra_text=r.get("sutra_text"),
            ))
    return records


def _maheshvara_tokens() -> list[tuple[str, bool]]:
    """Flat (phoneme_or_marker, is_it_marker) sequence across all 14
    sutras in their defined order — the raw material pratyāhāra names are
    cut from."""
    data = json.loads((_DATA_DIR / "maheshvara_shiva_sutras.json").read_text())
    rules = sorted(data["rules"], key=lambda r: r["id"])
    tokens: list[tuple[str, bool]] = []
    for r in rules:
        parts = r["sutra_text"].split()
        phonemes, it_marker = parts[:-1], parts[-1]
        for p in phonemes:
            tokens.append((p, False))
        tokens.append((it_marker, True))
    return tokens


_TOKENS = _maheshvara_tokens()


def pratyahara(start_phoneme: str, it_marker: str) -> set[str]:
    """The real Pāṇinian abbreviation mechanism: a name like "ac" (start
    "a", IT-marker "c") denotes every phoneme from the first occurrence of
    "a" up to the next occurrence of the IT consonant "c" in the 14-sutra
    sequence — here, {a,i,u,ṛ,ḷ,e,o,ai,au}, the complete vowel inventory.
    Raises ValueError if the start phoneme or IT-marker isn't found."""
    try:
        start_idx = next(i for i, (t, is_it) in enumerate(_TOKENS) if t == start_phoneme and not is_it)
    except StopIteration:
        raise ValueError(f"'{start_phoneme}' is not a phoneme in the Māheśvara Sūtras")

    collected: set[str] = set()
    for token, is_it in _TOKENS[start_idx:]:
        if is_it:
            if token == it_marker:
                return collected
            continue
        collected.add(token)
    raise ValueError(f"IT-marker '{it_marker}' never reached after '{start_phoneme}'")


# The two canonical pratyāhāras every Sanskrit grammar reference cites first.
def ac_vowels() -> set[str]:
    """अच् (ac) — every vowel."""
    return pratyahara("a", "c")


def hal_consonants() -> set[str]:
    """हल् (hal) — every consonant."""
    return pratyahara("ha", "l")


# --- vowel sandhi: four rules with real canonical sutra numbers -----------

_GUNA_I = {"i", "ī"}
_GUNA_U = {"u", "ū"}
_VRDDHI_E = {"e", "ai"}
_SAVARNA_A = {"a", "ā"}


def apply_vowel_sandhi(final_sound: str, initial_sound: str) -> tuple[str, str] | None:
    """Applies to the vowel at a word boundary. Returns (result, sutra_id)
    on a match, None if no rule applies. Only the vowel-sandhi subset with
    a directly cited canonical sutra number is implemented — this is not
    a general sandhi engine.

    - 6.1.87 (guṇa): a/ā + i/ī -> e
    - 6.1.88 (guṇa): a/ā + u/ū -> o
    - 6.1.89 (vṛddhi): a/ā + e/ai -> ai
    - 6.1.101 (savarṇa-dīrgha, a/ā case only): a/ā + a/ā -> ā
    """
    if final_sound in _SAVARNA_A:
        if initial_sound in _GUNA_I:
            return "e", "6.1.87"
        if initial_sound in _GUNA_U:
            return "o", "6.1.88"
        if initial_sound in _VRDDHI_E:
            return "ai", "6.1.89"
        if initial_sound in _SAVARNA_A:
            return "ā", "6.1.101"
    return None
