"""Tests against real, independently-verifiable classical Sanskrit
phonology facts — not just internal consistency checks. If these fail,
the pratyāhāra construction or sandhi rules are actually wrong, not just
differently implemented.
"""

from machine_brain.sutraflow.panini_phonology import (
    ac_vowels, apply_vowel_sandhi, hal_consonants, load_catalog, pratyahara,
)


def test_maheshvara_sutras_load_all_fourteen():
    catalog = load_catalog()
    maheshvara = [r for r in catalog if r.family == "phonetic_pratyahara"]
    assert len(maheshvara) == 14


def test_ashtadhyayi_seed_carries_real_canonical_sutra_ids():
    catalog = load_catalog()
    canonical_ids = {r.canonical_id for r in catalog if r.canonical_id}
    # These are real, checkable Aṣṭādhyāyī sutra numbers — 1.1.1 (vṛddhir
    # ādaic) and 1.1.2 (adeṅ guṇaḥ) are the grammar's own opening
    # definitional sutras.
    assert {"1.1.1", "1.1.2", "6.1.87", "6.1.88", "6.1.101"} <= canonical_ids


def test_ac_pratyahara_is_the_complete_vowel_inventory():
    vowels = ac_vowels()
    # The nine simple+diphthong vowels of classical Sanskrit phonology.
    assert vowels == {"a", "i", "u", "ṛ", "ḷ", "e", "o", "ai", "au"}


def test_hal_pratyahara_is_the_complete_consonant_inventory():
    consonants = hal_consonants()
    # 33 unique consonants (the classical count) — "ha" appears in both
    # sutra 5 and sutra 14, and must be deduplicated to a single entry.
    assert len(consonants) == 33
    assert "ha" in consonants and "ka" in consonants and "ta" in consonants


def test_pratyahara_unknown_start_phoneme_raises():
    import pytest
    with pytest.raises(ValueError):
        pratyahara("xyz", "c")


def test_guna_sandhi_a_plus_i_is_e():
    result, sutra_id = apply_vowel_sandhi("a", "i")
    assert result == "e"
    assert sutra_id == "6.1.87"


def test_guna_sandhi_a_plus_u_is_o():
    result, sutra_id = apply_vowel_sandhi("a", "u")
    assert result == "o"
    assert sutra_id == "6.1.88"


def test_vrddhi_sandhi_a_plus_e_is_ai():
    result, sutra_id = apply_vowel_sandhi("a", "e")
    assert result == "ai"
    assert sutra_id == "6.1.89"


def test_savarna_dirgha_a_plus_a_is_long_a():
    result, sutra_id = apply_vowel_sandhi("a", "a")
    assert result == "ā"
    assert sutra_id == "6.1.101"


def test_no_rule_applies_to_unrelated_vowels():
    assert apply_vowel_sandhi("i", "u") is None
