# fichier : tests/conftest.py
"""
conftest.py — Fixtures partagées de la suite de tests.

Les PR GitHub n'ont pas accès aux secrets → aucun test ne doit toucher
Earthdata, OPeNDAP ou toute source distante.
"""

import pytest


@pytest.fixture(autouse=True)
def no_earthdata_credentials(monkeypatch):
    """Retire les identifiants Earthdata de l'environnement pour CHAQUE test.

    Preuve que la suite ne peut pas joindre silencieusement un service
    distant, même si un futur changement introduit un appel réseau.
    """
    for var in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        monkeypatch.delenv(var, raising=False)