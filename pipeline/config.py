"""
config.py — Configuration centrale du pipeline GRACE/SASS.

RÈGLE : toute constante (chemin, seuil, période, URL, couleur) vit ICI et
uniquement ici. Aucun module ne doit contenir de valeur « en dur ».
Remplacer le SASS par un autre aquifère (phase 2) doit se faire en changeant
ce fichier (ou un registre), jamais le code des modules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins du dépôt
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" 
DOCS_DIR = REPO_ROOT / "docs"          # cible du rendu statique (GitHub Pages)
UI_STRINGS_DIR = REPO_ROOT / "ui_strings"

# ---------------------------------------------------------------------------
# AOI — paramètre, PAS une constante métier codée en dur dans les modules
# ---------------------------------------------------------------------------
AQUIFER_ID = "sass"                     # clé du cache Parquet (extensible phase 2)
AOI_GEOJSON = REPO_ROOT / "sass.geojson"


# Fenêtre englobante généreuse pour le premier découpage (spec §4a).
# Le découpage final se fait TOUJOURS sur le polygone exact, pas sur la bbox.
BBOX_LAT = (24.0, 35.0)
BBOX_LON = (-3.0, 19.0)                 # en convention -180/180
BBOX_LAT_MIN, BBOX_LAT_MAX = BBOX_LAT   # pour les imports directs dans les modules
BBOX_LON_MIN, BBOX_LON_MAX = BBOX_LON


# ---------------------------------------------------------------------------
# GRACE — mascon JPL RL06.3 V04, filtré CRI (spec §4a)
# ---------------------------------------------------------------------------
GRACE_COLLECTION = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"
GRACE_VAR = "lwe_thickness"             # cm, anomalie vs moyenne 2004–2009
GRACE_URL = (
    "https://archive.podaac.earthdata.nasa.gov/"
    "podaac-ops-cumulus-protected/"
    "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4/"
    "GRCTellus.JPL.200204_202605.GLO.RL06.3M.MSCNv04CRI.nc"
)

# ---------------------------------------------------------------------------
# GLDAS-2.1 Noah — flux de production principal (spec §4b)
# ---------------------------------------------------------------------------
GLDAS_SHORT_NAME = "GLDAS_NOAH025_M"
GLDAS_VERSION = "2.1"
#  Vérifier en semaine 1 : l'agrégation /dods peut renommer/minusculiser
#    les variables par rapport aux granules natifs.
GLDAS_COMPONENTS = [
    "SoilMoi0_10cm_inst",
    "SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst",
    "SWE_inst",        # ≈ 0 sur le SASS ; gardé pour la généralité (phase 2)
    "CanopInt_inst",
]                                       # toutes en kg/m² ≡ mm

# ---------------------------------------------------------------------------
# Période de référence (baseline) — spec §3
# ---------------------------------------------------------------------------
BASELINE_START = "2004-01"
BASELINE_END = "2009-12"
#    Ne jamais supposer que les 72 mois existent : on aligne GLDAS sur les
#    mois RÉELLEMENT présents dans la série GRACE à l'intérieur de la fenêtre.

# ---------------------------------------------------------------------------
# Cache local (spec §4) — le dashboard ne lit QUE ce fichier
# ---------------------------------------------------------------------------
SERIES_PARQUET = DATA_DIR / f"{AQUIFER_ID}_series.parquet"
LAST_REFRESH_JSON = DATA_DIR / "last_refresh.json"

# ---------------------------------------------------------------------------
# Prétraitement 
# ---------------------------------------------------------------------------
MASCON_NC_PATH = RAW_DIR / "mascon_cri.nc"   # fichier NetCDF local 
GRACE_UNCERTAINTY_FALLBACK_MM = 15.0         # fallback si NetCDF absent (indicatif, ~1-2 cm typique)
GAP_MIN_CONSECUTIVE = 3                      # seuil pour détecter les blocs de lacunes


# ---------------------------------------------------------------------------
# Statistiques / prévision
# ---------------------------------------------------------------------------
STL_SEASONAL = 13        # fenêtre Loess saisonnière 
STL_ROBUST = True         # résistant aux outliers (mois interpolés, événements ponctuels)
ACF_NLAGS = 36            # 3 ans de lags — suffisant pour voir les motifs annuels
HAC_MAXLAGS = 12                        # erreurs Newey–West (spec §6.1)
STL_PERIOD = 12                         # saisonnalité annuelle (spec §6.2)
CV_INITIAL = "2920 days"                # ~8 ans — validation à origine glissante
CV_PERIOD = "365 days"
CV_HORIZON = "730 days"                 # ~24 mois = horizon VALIDÉ (spec §7.3)
SCENARIO_HORIZON_MONTHS = 60            # extrapolation = scénario, PAS prévision
MIN_MASCONS_PER_COUNTRY = 3             # règle §6.1 pour la subdivision pays

# ---------------------------------------------------------------------------
# Palette du dashboard (Bloc E) — limitée et intentionnelle
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1a4e8a",     # bleu foncé — série GWSA (donnée principale)
    "accent": "#c0392b",      # rouge — tendance négative / alertes
    "context": "#8a8f98",     # gris — GLDAS, mois imputés, éléments secondaires
    "band": "#d6e4f0",        # bleu très pâle — bandes d'incertitude
    "background": "#ffffff",
}
