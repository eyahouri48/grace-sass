# fichier : pipeline/proxy.py
"""
Construction du proxy d'eaux souterraines (Option B-lite, spec §3).

GWSA ≈ TWSA − anomalie GLDAS (humidité du sol + neige + canopée).
Estimation prototype — pas un produit validé. Voir le disclaimer ci-dessous.
"""

import logging

import pandas as pd

from pipeline.config import BASELINE_END, BASELINE_START, DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer (spec §3) — à afficher dans le rapport ET dans le dashboard
# ---------------------------------------------------------------------------
PROXY_DISCLAIMER_FR = (
    "Estimation prototype : le stockage des eaux souterraines est approximé "
    "comme GRACE (TWSA) moins les composantes de surface GLDAS-Noah "
    "(humidité du sol, neige, canopée). Cette séparation de premier ordre "
    "hérite de l'erreur du modèle GLDAS, laisse l'eau de surface (chotts) "
    "et le stockage non modélisé dans le résidu, et n'est pas validée par "
    "la piézométrie in situ. Ne pas utiliser pour la comptabilité des "
    "ressources sans validation hydrologique."
)

PROXY_DISCLAIMER_EN = (
    "Prototype estimate: groundwater storage is approximated as GRACE (TWSA) "
    "minus GLDAS-Noah surface components (soil moisture, snow, canopy). This "
    "first-order separation inherits GLDAS model error, leaves surface water "
    "(chotts) and unmodelled storage in the residual, and is unvalidated "
    "against in-situ piezometry. Do not use for resource accounting without "
    "hydrological validation."
)


# ── Fonctions du proxy ─────────────────────────────────────────

def load_input_series() -> tuple[pd.Series, pd.Series]:
    """Charge twsa_cm et gldas_mm depuis le cache Parquet.

    Normalise les index au premier du mois (floor('MS')) —
    GRACE a des timestamps irréguliers en milieu de mois
    (ex: 2002-04-17 12:00:00), GLDAS est au 1er du mois.
    Sans cette normalisation, aucun mois ne correspond.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (twsa_cm, gldas_mm) — séries indexées par premier du mois.
    """
    twsa_path = DATA_DIR / "twsa_cm.parquet"
    gldas_path = DATA_DIR / "gldas_mm.parquet"

    twsa_cm = pd.read_parquet(twsa_path)["twsa_cm"]
    gldas_mm = pd.read_parquet(gldas_path)["gldas_mm"]

    # ── Normaliser les index au 1er du mois ──
    # GRACE : 2002-04-17 12:00:00 → 2002-04-01
    # GLDAS : 2000-01-01 00:00:00 → 2000-01-01 (déjà bon, mais on uniformise)
    twsa_cm.index = twsa_cm.index.to_period("M").to_timestamp()
    gldas_mm.index = gldas_mm.index.to_period("M").to_timestamp()

    # Si deux solutions GRACE tombent dans le même mois calendaire
    # (timestamps milieu-d'époque proches d'une fin de mois), on moyenne
    if twsa_cm.index.duplicated().any():
        n_dupes = twsa_cm.index.duplicated().sum()
        logger.warning(
            "%d mois GRACE en doublon après normalisation → moyenne",
            n_dupes,
        )
        twsa_cm = twsa_cm.groupby(twsa_cm.index).mean()
        twsa_cm.name = "twsa_cm"


    logger.info("twsa_cm  : %d mois (%s → %s)",
                len(twsa_cm), twsa_cm.index.min(), twsa_cm.index.max())
    logger.info("gldas_mm : %d mois (%s → %s)",
                len(gldas_mm), gldas_mm.index.min(), gldas_mm.index.max())
    return twsa_cm, gldas_mm


def find_baseline_months(twsa_cm: pd.Series) -> pd.DatetimeIndex:
    """Identifie les mois RÉELS de GRACE dans la fenêtre 2004-2009.

    On ne suppose PAS que les 72 mois calendaires existent tous —
    GRACE a quelques solutions manquantes même dans la baseline.
    C'est sur ces mois réels qu'on alignera la référence GLDAS.

    Parameters
    ----------
    twsa_cm : pd.Series
        Série GRACE (peut contenir des NaN pour les mois manquants,
        ou simplement ne pas avoir ces mois dans l'index).

    Returns
    -------
    pd.DatetimeIndex
        Les mois où GRACE a une valeur dans [2004-01, 2009-12].
    """
    # Filtrer la fenêtre 2004-2009
    mask = (twsa_cm.index >= BASELINE_START) & (twsa_cm.index <= BASELINE_END)
    baseline_window = twsa_cm.loc[mask]

    # Ne garder que les mois où GRACE a une vraie observation (pas NaN)
    real_months = baseline_window.dropna().index

    logger.info(
        "Baseline 2004–2009 : %d mois réels GRACE sur 72 calendaires",
        len(real_months),
    )
    return real_months


def anomalize_gldas(
    gldas_mm: pd.Series,
    baseline_months: pd.DatetimeIndex,
) -> pd.Series:
    """Convertit le stock absolu GLDAS en anomalie vs baseline GRACE.

    On calcule la moyenne GLDAS uniquement sur les mois où GRACE
    a des données dans 2004–2009, puis on soustrait cette moyenne
    de toute la série.

    Parameters
    ----------
    gldas_mm : pd.Series
        Stock absolu GLDAS (mm).
    baseline_months : pd.DatetimeIndex
        Mois réels de GRACE dans la fenêtre 2004–2009.

    Returns
    -------
    pd.Series
        gldas_anom_mm — anomalie GLDAS (mm) sur la même baseline que GRACE.
    """
    # Moyenne GLDAS sur les mois réels de la baseline GRACE
    gldas_baseline_mean = gldas_mm.reindex(baseline_months).mean()

    logger.info("Moyenne GLDAS sur la baseline : %.1f mm", gldas_baseline_mean)

    # Soustraire → anomalie
    gldas_anom_mm = gldas_mm - gldas_baseline_mean
    gldas_anom_mm.name = "gldas_anom_mm"

    logger.info(
        "gldas_anom_mm : plage [%.1f, %.1f] mm",
        gldas_anom_mm.min(), gldas_anom_mm.max(),
    )
    return gldas_anom_mm


def compute_gwsa(
    twsa_cm: pd.Series,
    gldas_anom_mm: pd.Series,
) -> pd.Series:
    """Calcule le proxy GWSA = TWSA (mm) − GLDAS anomalie (mm).

    Étapes :
    1. Convertir TWSA de cm → mm (× 10)
    2. Aligner les deux séries sur leur index commun
    3. Soustraire

    Parameters
    ----------
    twsa_cm : pd.Series
        TWSA GRACE en cm (anomalie vs 2004–2009).
    gldas_anom_mm : pd.Series
        Anomalie GLDAS en mm (même baseline).

    Returns
    -------
    pd.Series
        gwsa_mm — proxy eaux souterraines en mm.
    """
    # Conversion cm → mm
    twsa_mm = twsa_cm * 10
    twsa_mm.name = "twsa_mm"

    # Alignement sur l'index commun (inner join)
    # → on ne garde que les mois où les DEUX séries existent
    twsa_aligned, gldas_aligned = twsa_mm.align(gldas_anom_mm, join="inner")

    logger.info(
        "Mois communs après alignement : %d (GRACE %d, GLDAS %d)",
        len(twsa_aligned), len(twsa_mm), len(gldas_anom_mm),
    )

    # Soustraction — UNE ligne, comme promis par la spec
    gwsa_mm = twsa_aligned - gldas_aligned
    gwsa_mm.name = "gwsa_mm"

    logger.info(
        "gwsa_mm : %d mois, plage [%.1f, %.1f] mm",
        len(gwsa_mm), gwsa_mm.min(), gwsa_mm.max(),
    )
    return gwsa_mm


def save_proxy_parquet(
    twsa_cm: pd.Series,
    gldas_anom_mm: pd.Series,
    gwsa_mm: pd.Series,
    dest=None,
) -> None:
    """Sauvegarde les trois séries dans le cache principal.

    C'est ce fichier (sass_series.parquet) que le dashboard lit.
    """
    if dest is None:
        dest = DATA_DIR / "sass_series.parquet"

    # Aligner tout sur le même index (outer → NaN si une série est plus courte)
    df = pd.DataFrame({
        "twsa_cm": twsa_cm,
        "gldas_anom_mm": gldas_anom_mm,
        "gwsa_mm": gwsa_mm,
    })
    df.index.name = "time"

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest)
    logger.info("Cache principal sauvegardé : %s (%d lignes)", dest, len(df))


# ── Point d'entrée ──────────────────────────────────────────────

def run() -> pd.Series:
    """Pipeline complet du proxy : charger → anomaliser → soustraire → cacher."""
    twsa_cm, gldas_mm = load_input_series()

    # 1. Identifier les mois réels de GRACE dans la baseline
    baseline_months = find_baseline_months(twsa_cm)

    # 2. Anomaliser GLDAS sur cette baseline
    gldas_anom_mm = anomalize_gldas(gldas_mm, baseline_months)

    # 3. Calculer le proxy
    gwsa_mm = compute_gwsa(twsa_cm, gldas_anom_mm)

    # 4. Sauvegarder le cache principal
    save_proxy_parquet(twsa_cm, gldas_anom_mm, gwsa_mm)

    return gwsa_mm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    gwsa = run()
    print(f"\nPremiers mois :\n{gwsa.head(10)}")
    print(f"\nDerniers mois :\n{gwsa.tail(5)}")
    print(f"\n{PROXY_DISCLAIMER_FR}")