"""
conftest.py — Fixtures partagées de la suite de tests.

RÈGLE (spec §8.3) : les tests tournent 100 % HORS LIGNE.
Les PR GitHub n'ont pas accès aux secrets → aucun test ne doit toucher
Earthdata, OPeNDAP ou toute source distante.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def no_earthdata_credentials(monkeypatch):
    """Retire les identifiants Earthdata de l'environnement pour CHAQUE test.

    Preuve que la suite ne peut pas joindre silencieusement un service
    distant, même si un futur changement introduit un appel réseau.
    """
    for var in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        monkeypatch.delenv(var, raising=False)


# Fixture de données : un petit instantané Parquet/CSV de gwsa_mm
# (incluant quelques mois de lacune) sera committé dans tests/fixtures/
# une fois l'ingestion réelle fonctionnelle (Tâche 15).
