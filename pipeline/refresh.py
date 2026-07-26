# fichier : pipeline/refresh.py
"""
Actualisation append-only des données GRACE + GLDAS (spec §8.3).

Stratégie :
  - GRACE : re-télécharge le mascon global (~50 Mo) via CMR (earthaccess),
    compare avec le cache existant, sauvegarde la série complète.
  - GLDAS : délègue à ingest_gldas() qui gère l'append nativement.
  - Proxy : re-calcule via proxy.run() sur les caches à jour.
  - Heartbeat : écrit last_refresh.json à CHAQUE run (même si rien de nouveau).

Idempotent : relancer sans données nouvelles ne modifie rien de matériel.
Appelé par : uv run python -m pipeline.refresh
             ou GitHub Actions (refresh-dashboard.yml, cron lundi 06:00 UTC)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline import config

logger = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────

def _last_valid_month(series: pd.Series) -> str | None:
    """Dernier mois non-NaN au format 'YYYY-MM', ou None si série vide."""
    valid = series.dropna()
    if valid.empty:
        return None
    return valid.index.max().strftime("%Y-%m")


# ── Refresh GRACE ───────────────────────────────────────────────

def refresh_grace() -> int:
    """Re-télécharge le mascon CRI et met à jour le cache twsa_cm.

    Le mascon est un fichier global unique que JPL remplace mensuellement
    (le nom change car il contient la date de fin). On supprime tout NC
    local, on re-télécharge via CMR, et on compare avec le Parquet
    existant pour compter les mois ajoutés.

    Returns
    -------
    int
        Nombre de mois GRACE nouveaux ajoutés.
    """
    from pipeline.ingest_grace import (
        download_grace_mascon,
        extract_twsa_basin_mean,
        save_twsa_parquet,
    )

    # Compter les mois en cache AVANT le re-téléchargement
    twsa_cache = config.DATA_DIR / "twsa_cm.parquet"
    n_cached = 0
    if twsa_cache.exists():
        cached = pd.read_parquet(twsa_cache)["twsa_cm"]
        n_cached = len(cached.dropna())

    # Supprimer tous les NC locaux pour forcer le re-téléchargement
    # (le nom du fichier change à chaque publication JPL)
    if config.RAW_DIR.exists():
        for nc_file in config.RAW_DIR.glob("*.nc"):
            nc_file.unlink()
            logger.info("NC local supprimé : %s", nc_file.name)

    # Télécharger le dernier mascon via CMR + extraire la série complète
    nc_path = download_grace_mascon()
    fresh_twsa = extract_twsa_basin_mean(nc_path)
    n_fresh = len(fresh_twsa.dropna())

    # Sauvegarder la série complète (écrase le Parquet — GRACE est un fichier unique)
    save_twsa_parquet(fresh_twsa)

    # Compter les mois nouveaux
    n_new = max(n_fresh - n_cached, 0)
    if n_new > 0:
        logger.info("GRACE : %d mois nouveaux (total %d).", n_new, n_fresh)
    else:
        logger.info("GRACE : aucun mois nouveau (%d en cache).", n_cached)

    return n_new


# ── Refresh GLDAS ───────────────────────────────────────────────

def refresh_gldas() -> int:
    """Télécharge les granules GLDAS manquants (append natif).

    ingest_gldas() gère déjà la détection des mois en cache et
    ne télécharge que les granules manquants — on la réutilise telle quelle.

    Returns
    -------
    int
        Nombre de mois GLDAS nouveaux ajoutés.
    """
    from pipeline.ingest_gldas import ingest_gldas, GLDAS_PARQUET

    # Compter les mois avant
    n_before = 0
    if GLDAS_PARQUET.exists():
        n_before = len(pd.read_parquet(GLDAS_PARQUET))

    # ingest_gldas() charge le cache, cherche les nouveaux granules, append
    df = ingest_gldas()
    n_after = len(df)
    n_new = max(n_after - n_before, 0)

    if n_new > 0:
        logger.info("GLDAS : %d mois nouveaux (total %d).", n_new, n_after)
    else:
        logger.info("GLDAS : aucun mois nouveau (%d en cache).", n_before)

    return n_new


# ── Refresh proxy + cache principal ─────────────────────────────

def refresh_proxy() -> None:
    """Re-calcule le proxy GWSA à partir des caches à jour.

    Délègue à proxy.run() qui :
    1. Charge twsa_cm.parquet + gldas_mm.parquet
    2. Normalise les index au 1er du mois
    3. Trouve les mois réels GRACE dans la baseline 2004-2009
    4. Anomalise GLDAS sur cette baseline
    5. Calcule gwsa_mm = twsa_cm × 10 − gldas_anom_mm
    6. Sauvegarde sass_series.parquet
    """
    from pipeline.proxy import run as proxy_run

    proxy_run()
    logger.info("Proxy GWSA recalculé → %s", config.SERIES_PARQUET)


# ── Heartbeat (last_refresh.json) ───────────────────────────────

def write_refresh_metadata(n_grace_new: int, n_gldas_new: int) -> dict:
    """Écrit last_refresh.json — TOUJOURS, même si rien de nouveau.

    Ce fichier sert de :
    - heartbeat contre l'auto-désactivation GitHub après 60 jours
    - source pour l'horodatage de fraîcheur à 3 lignes du dashboard
    """
    last_grace = None
    last_gldas = None
    last_common = None

    if config.SERIES_PARQUET.exists():
        df = pd.read_parquet(config.SERIES_PARQUET)
        if "twsa_cm" in df.columns:
            last_grace = _last_valid_month(df["twsa_cm"])
        if "gldas_anom_mm" in df.columns:
            last_gldas = _last_valid_month(df["gldas_anom_mm"])
        if "gwsa_mm" in df.columns:
            last_common = _last_valid_month(df["gwsa_mm"])

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_grace_month": last_grace,
        "last_gldas_month": last_gldas,
        "last_common_month": last_common,
        "grace_months_added": n_grace_new,
        "gldas_months_added": n_gldas_new,
    }

    meta_path = Path(config.LAST_REFRESH_JSON)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Métadonnées écrites : %s", meta_path)
    logger.info(
        "Fraîcheur — GRACE : %s | GLDAS : %s | proxy GWSA : %s",
        last_grace or "∅", last_gldas or "∅", last_common or "∅",
    )
    return meta


# ── Orchestrateur principal ─────────────────────────────────────

def refresh() -> dict:
    """Exécute l'actualisation complète.

    Ordre : GRACE → GLDAS → proxy → métadonnées.
    Chaque source est indépendante : si GRACE échoue, on garde
    le cache existant et on tente quand même GLDAS (et vice versa).
    Exit code 0 dans tous les cas.

    Returns
    -------
    dict
        Métadonnées de fraîcheur (contenu de last_refresh.json).
    """
    logger.info("=" * 60)
    logger.info("REFRESH — début (%s)", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)

    # --- GRACE (try/except : si ça échoue, on continue) ---
    try:
        n_grace = refresh_grace()
    except Exception:
        logger.exception("Échec ingestion GRACE — cache existant conservé.")
        n_grace = 0

    # --- GLDAS (try/except : si ça échoue, on continue) ---
    try:
        n_gldas = refresh_gldas()
    except Exception:
        logger.exception("Échec ingestion GLDAS — cache existant conservé.")
        n_gldas = 0

    # --- Proxy (seulement si les deux caches intermédiaires existent) ---
    twsa_cache = config.DATA_DIR / "twsa_cm.parquet"
    gldas_cache = config.DATA_DIR / "gldas_mm.parquet"

    if twsa_cache.exists() and gldas_cache.exists():
        try:
            refresh_proxy()
        except Exception:
            logger.exception("Échec calcul proxy — cache principal inchangé.")
    else:
        logger.warning(
            "Cache(s) manquant(s) — proxy non recalculé. "
            "twsa=%s, gldas=%s",
            twsa_cache.exists(), gldas_cache.exists(),
        )

    # --- Métadonnées (TOUJOURS — heartbeat) ---
    meta = write_refresh_metadata(n_grace, n_gldas)

    logger.info("=" * 60)
    logger.info("REFRESH — terminé (GRACE +%d, GLDAS +%d)", n_grace, n_gldas)
    logger.info("=" * 60)

    return meta


# ── Point d'entrée CLI ──────────────────────────────────────────

def main() -> None:
    """python -m pipeline.refresh"""
    meta = refresh()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()