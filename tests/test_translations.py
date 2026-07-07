"""Test de parité des clés de traduction (spec §12, livrable 4).

Chaque clé anglaise doit avoir un équivalent français, et réciproquement —
détection automatique de la dérive bilingue.
"""

import json

from pipeline.config import UI_STRINGS_DIR


def _load(lang: str) -> dict:
    with open(UI_STRINGS_DIR / f"{lang}.json", encoding="utf-8") as f:
        return json.load(f)


def test_translation_key_parity():
    en, fr = _load("en"), _load("fr")
    assert set(en.keys()) == set(fr.keys()), (
        f"Clés seulement en EN : {set(en) - set(fr)} ; "
        f"seulement en FR : {set(fr) - set(en)}"
    )


def test_no_empty_labels():
    for lang in ("en", "fr"):
        data = _load(lang)
        empty = [k for k, v in data.items() if not str(v).strip()]
        assert not empty, f"Libellés vides dans {lang}.json : {empty}"
