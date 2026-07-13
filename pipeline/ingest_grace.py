# fichier : pipeline/ingest_grace.py
"""Ingestion GRACE — téléchargement HTTPS du mascon CRI, découpe AOI,
moyenne de bassin pondérée cos-lat → twsa_cm en Parquet."""

import logging
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from pathlib import Path

from pipeline.config import (
    GRACE_URL,
    AOI_GEOJSON,
    BBOX_LAT_MIN, BBOX_LAT_MAX,
    BBOX_LON_MIN, BBOX_LON_MAX,
    DATA_DIR,
    RAW_DIR,
)

logger = logging.getLogger(__name__)


# ── Téléchargement ──────────────────────────────────────────────

def download_grace_mascon(dest: Path | None = None) -> Path:
    """Télécharge le fichier mascon CRI global (~50 Mo) via HTTPS.

    Utilise earthaccess pour l'authentification Earthdata Login.
    Si le fichier existe déjà localement, on ne re-télécharge pas.

    Returns
    -------
    Path
        Chemin local du fichier .nc téléchargé.
    """
    import earthaccess

    if dest is None:
        dest = RAW_DIR / "mascon_cri.nc"

    if dest.exists():
        logger.info("Mascon CRI déjà en cache : %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # earthaccess gère l'auth via EARTHDATA_USERNAME/PASSWORD en env
    earthaccess.login(strategy="environment")

    logger.info("Téléchargement du mascon CRI depuis PO.DAAC...")

    session = earthaccess.get_requests_https_session()
    resp = session.get(GRACE_URL, timeout=300)
    resp.raise_for_status()

    dest.write_bytes(resp.content)
    logger.info("Mascon CRI sauvegardé : %s (%.1f Mo)", dest, dest.stat().st_size / 1e6)
    return dest


# ── Découpe AOI + moyenne de bassin ─────────────────────────────

def extract_twsa_basin_mean(nc_path: Path) -> pd.Series:
    """Ouvre le mascon CRI, découpe sur l'AOI, et calcule la moyenne
    de bassin pondérée par cosinus de latitude.

    Parameters
    ----------
    nc_path : Path
        Chemin du fichier mascon CRI téléchargé.

    Returns
    -------
    pd.Series
        Série mensuelle `twsa_cm` indexée par date (mois),
        en cm d'eau équivalente (anomalie vs 2004–2009).
    """
    # 1. Ouvrir le NetCDF
    ds = xr.open_dataset(nc_path)
    logger.info("Variables disponibles : %s", list(ds.data_vars))

    # 2. LONGITUDE GRACE est en 0–360°, notre AOI est en -180/180°
    #    Convertir GRACE en -180/180° avant tout découpage
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    logger.info("Longitude convertie 0–360 → -180/180")

    # 3. Découper une bounding box large (filtre rapide)
    lwe = ds["lwe_thickness"].sel(
        lat=slice(BBOX_LAT_MIN, BBOX_LAT_MAX),
        lon=slice(BBOX_LON_MIN, BBOX_LON_MAX),
    )
    logger.info("Bounding box : %d pas de temps, grille %s", lwe.sizes["time"], lwe.shape[1:])

    # 4. Découper au polygone exact avec rioxarray
    aoi = gpd.read_file(AOI_GEOJSON)
    lwe = lwe.rio.write_crs("EPSG:4326")
    lwe_clipped = lwe.rio.clip(aoi.geometry, aoi.crs, all_touched=True, drop=False)
    logger.info("Découpe AOI effectuée (all_touched=True)")

    # 5. Moyenne pondérée par cosinus de latitude
    weights = np.cos(np.deg2rad(lwe_clipped.lat))
    twsa_cm = lwe_clipped.weighted(weights).mean(dim=["lat", "lon"]).to_series()
    twsa_cm.name = "twsa_cm"

    logger.info("Série twsa_cm : %d mois, de %s à %s",
                len(twsa_cm), twsa_cm.index.min(), twsa_cm.index.max())
    return twsa_cm


# ── Sauvegarde en cache ─────────────────────────────────────────

def save_twsa_parquet(twsa_cm: pd.Series, dest: Path | None = None) -> Path:
    """Sauvegarde twsa_cm en Parquet dans data/."""
    if dest is None:
        dest = DATA_DIR / "twsa_cm.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)

    df = twsa_cm.to_frame()
    df.index.name = "time"
    df.to_parquet(dest)
    logger.info("Cache Parquet sauvegardé : %s", dest)
    return dest


# ── Point d'entrée ──────────────────────────────────────────────

def run():
    """Pipeline complet : télécharger → découper → moyenner → cacher."""
    nc_path = download_grace_mascon()
    twsa_cm = extract_twsa_basin_mean(nc_path)
    save_twsa_parquet(twsa_cm)
    return twsa_cm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    series = run()
    print(f"\nPremiers mois :\n{series.head(10)}")
    print(f"\nDerniers mois :\n{series.tail(5)}")