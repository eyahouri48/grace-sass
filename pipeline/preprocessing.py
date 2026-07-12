# fichier : pipeline/preprocessing.py

"""
Prétraitement de la série gwsa_mm :
- réindexation sur un index mensuel complet
- interpolation des lacunes + drapeau is_imputed
- détection des périodes de lacunes (lacune inter-missions)
- bande d'incertitude mascon (indicative)

NOTE (spec §5) : les facteurs de gain (scale/gain factors) du fichier
mascon sont DÉLIBÉRÉMENT ignorés. Ils restaurent le signal infra-mascon ;
pour une moyenne de bassin (~12 mascons), ils sont non matériels.
JPL note qu'ils sont dominés par le cycle annuel et discutables pour
l'analyse de tendance. Choix documenté dans le rapport technique.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — active l'accesseur .rio
import xarray as xr

from pipeline.config import (
    AOI_GEOJSON,
    BBOX_LAT_MAX,
    BBOX_LAT_MIN,
    BBOX_LON_MAX,
    BBOX_LON_MIN,
    GAP_MIN_CONSECUTIVE,
    GRACE_UNCERTAINTY_FALLBACK_MM,
    MASCON_NC_PATH,
    SERIES_PARQUET,
)

logger = logging.getLogger(__name__)


# ── Chargement ──────────────────────────────────────────────────


def load_proxy_series(path=None) -> pd.DataFrame:
    """Charge le cache Parquet et retourne le DataFrame brut.

    Coupe aux bornes réelles de gwsa_mm (premier/dernier mois non-NaN) —
    les mois hors couverture GRACE ne sont pas des lacunes, GRACE
    n'existait pas encore (avant avr 2002) ou GLDAS n'a pas encore
    rattrapé (après le dernier mois GRACE).
    """
    if path is None:
        path = SERIES_PARQUET

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    # Bornes réelles du proxy (premier et dernier mois avec une valeur)
    valid_mask = df["gwsa_mm"].notna()
    first_valid = df.index[valid_mask].min()
    last_valid = df.index[valid_mask].max()
    df = df.loc[first_valid:last_valid]

    logger.info(
        "Série brute : %d mois (%s → %s), NaN gwsa_mm = %d",
        len(df),
        first_valid.strftime("%Y-%m"),
        last_valid.strftime("%Y-%m"),
        df["gwsa_mm"].isna().sum(),
    )
    return df


# ── Réindexation ────────────────────────────────────────────────


def reindex_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Réindexe sur un DatetimeIndex mensuel complet (1er de chaque mois).

    Détecte les mois sans mesure GRACE (NaN dans gwsa_mm) et les marque
    via la colonne 'is_imputed'.
    """
    # Index mensuel complet
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS",
    )
    df_full = df.reindex(full_index)
    df_full.index.name = "date"

    # Un mois est imputé s'il n'a PAS de mesure gwsa_mm
    df_full["is_imputed"] = df_full["gwsa_mm"].isna()

    n_missing = df_full["is_imputed"].sum()
    logger.info(
        "Réindexation : %d mois, dont %d manquants (%.1f%%)",
        len(df_full),
        n_missing,
        100 * n_missing / len(df_full),
    )
    return df_full


# ── Interpolation ───────────────────────────────────────────────


def interpolate_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpole linéairement les lacunes de gwsa_mm (et twsa_cm, gldas_anom_mm).

    Les mois interpolés restent marqués is_imputed=True —
    on ne les confondra jamais avec de vraies mesures.
    """
    df = df.copy()

    for col in ["twsa_cm", "gldas_anom_mm", "gwsa_mm"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time")

    n_remaining = df["gwsa_mm"].isna().sum()
    logger.info(
        "Après interpolation : %d NaN restants dans gwsa_mm",
        n_remaining,
    )
    return df


# ── Détection des lacunes ──────────────────────────────────────


def find_gap_periods(df: pd.DataFrame) -> list[tuple]:
    """Détecte les blocs de mois imputés consécutifs (>= GAP_MIN_CONSECUTIVE).

    Retourne une liste de tuples (date_début, date_fin, nb_mois),
    triée par durée décroissante. La lacune inter-missions GRACE/GRACE-FO
    (~11 mois) sera la première.
    """
    imputed = df["is_imputed"].values
    gaps = []
    i = 0

    while i < len(imputed):
        if imputed[i]:
            start_idx = i
            while i < len(imputed) and imputed[i]:
                i += 1
            n_months = i - start_idx
            if n_months >= GAP_MIN_CONSECUTIVE:
                gaps.append((
                    df.index[start_idx],
                    df.index[i - 1],
                    n_months,
                ))
        else:
            i += 1

    gaps.sort(key=lambda x: x[2], reverse=True)
    return gaps


# ── Bande d'incertitude mascon ─────────────────────────────────


def add_uncertainty_band(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'uncertainty_mm' depuis le NetCDF mascon.

    C'est la moyenne de bassin (pondérée cos-lat) de la variable
    'uncertainty' du mascon, convertie de cm → mm.

    ⚠ Indicatif seulement — les erreurs mascon sont spatialement
    corrélées, la moyenne de bassin n'est PAS une propagation
    d'erreur formelle (spec §5, §6.1).

    Si le NetCDF est absent → fallback constante.
    """
    df = df.copy()

    try:
        ds = xr.open_dataset(MASCON_NC_PATH)
    except FileNotFoundError:
        logger.warning(
            "NetCDF mascon introuvable (%s) — fallback %.0f mm",
            MASCON_NC_PATH,
            GRACE_UNCERTAINTY_FALLBACK_MM,
        )
        df["uncertainty_mm"] = GRACE_UNCERTAINTY_FALLBACK_MM
        return df

    # Conversion longitude 0-360 → -180/180 
    ds = ds.assign_coords(
        lon=(((ds.lon + 180) % 360) - 180)
    ).sortby("lon")

    # Découpe bbox puis polygone exact
    bbox = ds["uncertainty"].sel(
        lat=slice(BBOX_LAT_MIN, BBOX_LAT_MAX),
        lon=slice(BBOX_LON_MIN, BBOX_LON_MAX),
    )
    aoi = gpd.read_file(AOI_GEOJSON)
    sub = (
        bbox.rio.write_crs("EPSG:4326").rio.clip(
            aoi.geometry, aoi.crs, all_touched=True, drop=False
        )
    )

    # Moyenne pondérée cos-lat → série mensuelle (cm → mm)
    w = np.cos(np.deg2rad(sub.lat))
    uncert_cm = sub.weighted(w).mean(dim=["lat", "lon"]).to_series()
    uncert_mm = uncert_cm * 10


    # Normaliser timestamps au 1er du mois
    uncert_mm.index = uncert_mm.index.to_period("M").to_timestamp()
    uncert_mm.index.name = "date"

    # Doublons possibles (2 solutions GRACE dans le même mois) → moyenne
    if uncert_mm.index.duplicated().any():
        uncert_mm = uncert_mm.groupby(uncert_mm.index).mean()

    # Aligner sur l'index du DataFrame et interpoler les mois imputés
    df["uncertainty_mm"] = uncert_mm.reindex(df.index)
    df["uncertainty_mm"] = df["uncertainty_mm"].interpolate(method="time")

    logger.info(
        "Incertitude mascon : moyenne %.1f mm, plage [%.1f, %.1f] mm",
        df["uncertainty_mm"].mean(),
        df["uncertainty_mm"].min(),
        df["uncertainty_mm"].max(),
    )
    return df


def save_preprocessed(df: pd.DataFrame, path=None) -> None:
    """Sauvegarde le DataFrame prétraité dans le cache Parquet principal.

    Écrase le fichier existant — le nouveau contient les colonnes
    supplémentaires (is_imputed, uncertainty_mm).
    """
    if path is None:
        path = SERIES_PARQUET

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(
        "Cache prétraité sauvegardé : %s (%d lignes, %d colonnes)",
        path, len(df), len(df.columns),
    )

# ── Point d'entrée ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # 1. Charger
    df_raw = load_proxy_series()

    # 2. Réindexer
    df_full = reindex_monthly(df_raw)

    # 3. Interpoler
    df_interp = interpolate_gaps(df_full)

    # 4. Détecter les lacunes
    gaps = find_gap_periods(df_interp)
    print(f"\nLacunes >= {GAP_MIN_CONSECUTIVE} mois consécutifs :")
    for start, end, n in gaps:
        label = " ← LACUNE INTER-MISSIONS" if n >= 8 else ""
        print(f"  {start:%Y-%m} → {end:%Y-%m} ({n} mois){label}")

    # 5. Incertitude mascon
    df_final = add_uncertainty_band(df_interp)

    # 6. Sauvegarder
    save_preprocessed(df_final)

    # Résumé
    print(f"\n{'='*50}")
    print(f"Prétraitement terminé")
    print(f"  Mois : {len(df_final)}")
    print(f"  Imputés : {df_final['is_imputed'].sum()}")
    print(f"  Incertitude moyenne : {df_final['uncertainty_mm'].mean():.1f} mm")
    print(f"  Colonnes : {list(df_final.columns)}")
    print(f"{'='*50}")