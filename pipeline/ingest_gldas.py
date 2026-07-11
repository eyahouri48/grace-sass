# fichier : pipeline/ingest_gldas.py
"""Ingestion GLDAS-2.1 Noah — composantes de surface pour le proxy GWSA.

Télécharge les granules mensuels GLDAS via earthaccess (solution de repli
après retrait du serveur OPeNDAP GDS, cf. spec §4b), extrait les 6
composantes de stockage de surface sur l'AOI, calcule la moyenne de bassin
pondérée par cosinus de latitude, et cache le résultat en Parquet.

Chaque granule (~24 Mo, grille mondiale) est téléchargé par lots de 20
(parallèle), traité, puis supprimé — seul le Parquet final est conservé.
Sauvegarde progressive pour reprise en cas d'interruption.
"""

import logging
import shutil
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr

from pipeline.config import (
    BBOX_LAT,
    BBOX_LON,
    DATA_DIR,
    GLDAS_COMPONENTS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

# Sous-dossier temporaire pour les granules GLDAS (supprimés après traitement)
GLDAS_TMP_DIR = RAW_DIR / "gldas_tmp"
# Chemin du cache Parquet de sortie
GLDAS_PARQUET = DATA_DIR / "gldas_mm.parquet"


def search_gldas_granules(
    start: str = "2000-01-01",
    end: str = "2030-12-31",
) -> list:
    """Recherche les granules GLDAS_NOAH025_M v2.1 via earthaccess.

    Parameters
    ----------
    start, end : str
        Bornes temporelles de la recherche (format ISO).

    Returns
    -------
    list
        Liste de DataGranule (résultats earthaccess).
    """
    results = earthaccess.search_data(
        short_name="GLDAS_NOAH025_M",
        version="2.1",
        temporal=(start, end),
    )
    logger.info("Granules GLDAS trouvés : %d", len(results))
    return results


def process_one_granule(nc_path: Path) -> dict | None:
    """Ouvre un granule GLDAS, extrait la moyenne de bassin des 6 composantes.

    Parameters
    ----------
    nc_path : Path
        Chemin vers le fichier NetCDF du granule.

    Returns
    -------
    dict | None
        {"time": datetime, "gldas_mm": float} ou None si le traitement échoue.
    """
    try:
        ds = xr.open_dataset(nc_path)

        # --- Sous-ensemble spatial (bbox du SASS) ---
        # GLDAS est nativement en -180/180 — pas de conversion nécessaire
        lat_min, lat_max = BBOX_LAT
        lon_min, lon_max = BBOX_LON
        box = ds[GLDAS_COMPONENTS].sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )

        # --- Somme des 6 composantes (skipna=False !) ---
        # Si une composante est NaN pour un pixel → la somme est NaN
        # → ce pixel est exclu de la moyenne (pas biaisé vers 0)
        total = box.to_array(dim="component").sum(
            dim="component", skipna=False
        )

        # --- Moyenne de bassin pondérée par cosinus de latitude ---
        weights = np.cos(np.deg2rad(total.lat))
        basin_mean = float(
            total.weighted(weights).mean(dim=["lat", "lon"]).values.item()
        )

        # --- Date : extraire le mois du granule ---
        time_val = pd.Timestamp(ds.time.values[0])

        ds.close()
        return {"time": time_val, "gldas_mm": basin_mean}

    except Exception as e:
        logger.error("Erreur traitement %s : %s", nc_path.name, e)
        return None


def _save_parquet(records: list[dict]) -> None:
    """Sauvegarde la liste de records en Parquet trié par date."""
    df = pd.DataFrame(records)
    df = df.set_index("time").sort_index()
    df.index.name = "time"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GLDAS_PARQUET)


def ingest_gldas(
    start: str = "2000-01-01",
    end: str = "2030-12-31",
    batch_size: int = 20,
) -> pd.DataFrame:
    """Pipeline complet : recherche → téléchargement par lots → traitement → cache.

    Télécharge les granules par lots (parallèle via earthaccess) pour
    accélérer le transfert, traite chaque lot, supprime les fichiers bruts,
    et sauvegarde progressivement pour pouvoir reprendre si interrompu.

    Parameters
    ----------
    start, end : str
        Bornes temporelles.
    batch_size : int
        Nombre de granules téléchargés en parallèle par lot (défaut : 20).

    Returns
    -------
    pd.DataFrame
        Série mensuelle gldas_mm (mm = kg/m²), indexée par date.
    """
    # --- Authentification Earthdata ---
    earthaccess.login(strategy="environment")

    # --- Charger le cache existant (pour reprendre si interrompu) ---
    existing_dates: set[str] = set()
    records: list[dict] = []
    if GLDAS_PARQUET.exists():
        df_existing = pd.read_parquet(GLDAS_PARQUET)
        existing_dates = {d.strftime("%Y-%m") for d in df_existing.index}
        records = df_existing.reset_index().to_dict("records")
        logger.info(
            "Cache existant chargé : %d mois (reprise)", len(existing_dates)
        )

    # --- Recherche des granules ---
    granules = search_gldas_granules(start, end)
    if not granules:
        raise RuntimeError("Aucun granule GLDAS trouvé.")

    # --- Filtrer les granules déjà en cache ---
    to_process = []
    for g in granules:
        temporal = g["umm"]["TemporalExtent"]["RangeDateTime"]
        g_date = pd.Timestamp(temporal["BeginningDateTime"]).strftime("%Y-%m")
        if g_date not in existing_dates:
            to_process.append(g)

    if not to_process:
        logger.info("Tous les granules sont déjà en cache.")
        return pd.read_parquet(GLDAS_PARQUET)

    logger.info(
        "%d granules à traiter (%d déjà en cache).",
        len(to_process), len(existing_dates),
    )

    # --- Traitement par lots ---
    GLDAS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    new_count = 0

    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(to_process))
        logger.info(
            "--- Lot %d–%d / %d (téléchargement parallèle) ---",
            batch_start + 1, batch_end, len(to_process),
        )

        # Télécharger le lot en parallèle
        try:
            downloaded = earthaccess.download(batch, str(GLDAS_TMP_DIR))
        except Exception as e:
            logger.error("Échec téléchargement du lot : %s", e)
            continue

        # Traiter chaque fichier du lot
        for filepath in downloaded:
            nc_path = Path(filepath)
            result = process_one_granule(nc_path)
            if result is not None:
                records.append(result)
                new_count += 1
            # Supprimer le fichier brut immédiatement
            nc_path.unlink(missing_ok=True)

        # Sauvegarde après chaque lot
        if new_count > 0:
            _save_parquet(records)
            logger.info(
                "Sauvegarde intermédiaire : %d nouveaux mois (total %d).",
                new_count, len(records),
            )

    # Nettoyer le dossier temporaire
    if GLDAS_TMP_DIR.exists():
        shutil.rmtree(GLDAS_TMP_DIR, ignore_errors=True)

    if not records:
        raise RuntimeError("Aucun granule traité avec succès.")

    # --- Sauvegarde finale ---
    _save_parquet(records)
    logger.info(
        "Terminé : %d nouveaux mois ajoutés, total %d.", new_count, len(records)
    )

    return pd.read_parquet(GLDAS_PARQUET)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    df = ingest_gldas()
    print(f"\nSérie GLDAS : {len(df)} mois")
    print(f"Période : {df.index[0]} → {df.index[-1]}")
    print(f"Plage : {df['gldas_mm'].min():.1f} → {df['gldas_mm'].max():.1f} mm")
    print(f"\nAperçu (5 premiers) :\n{df.head()}")
    print(f"\nAperçu (5 derniers) :\n{df.tail()}")