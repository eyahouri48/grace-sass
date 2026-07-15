# fichier : notebooks/07_trend_analysis.py
"""Tâche 6 — Validation de l'estimation de tendance."""

import pandas as pd
from pipeline.trend import compute_full_trend

# --- Charger les données prétraitées ---
df = pd.read_parquet("data/sass_series.parquet")

# --- Calcul complet ---
results = compute_full_trend(
    series=df["gwsa_mm"],
    is_imputed=df["is_imputed"],
)

# --- Affichage structuré ---
print("=" * 60)
print("ESTIMATION DE TENDANCE — GWSA proxy (SASS)")
print("=" * 60)

print(f"\n Aire AOI : {results['area_km2']:,.0f} km²")
print(f"   (1 mm ≈ {results['area_m2'] / 1e12:.2f} km³)")

print("\n OLS + HAC (Newey-West, maxlags=12)")
print(f"   Pente      : {results['ols_slope_mm_yr']:.2f} mm/an")
print(f"   IC 95% HAC : [{results['ols_ci_lower_mm_yr']:.2f}, {results['ols_ci_upper_mm_yr']:.2f}] mm/an")
print(f"   En volume  : {results['ols_slope_km3_yr']:.2f} km³/an")
print(f"   IC volume  : [{results['ols_ci_lower_km3_yr']:.2f}, {results['ols_ci_upper_km3_yr']:.2f}] km³/an")
print(f"   p-value    : {results['ols_pvalue']:.2e}")
print(f"   R²         : {results['ols_r_squared']:.3f}")
print(f"   N obs      : {results['ols_n_obs']}")
print(f"   Erreur-type naïve : {results['ols_slope_naive_se']:.3f} mm/an")
print(f"   Erreur-type HAC   : {results['ols_slope_hac_se']:.3f} mm/an")
print(f"   → Ratio HAC/naïf  : {results['ols_slope_hac_se'] / results['ols_slope_naive_se']:.1f}x")

print("\n Mann-Kendall saisonnier + Sen")
print(f"   Tendance   : {results['mk_mk_trend']}")
print(f"   p-value MK : {results['mk_mk_pvalue']:.2e}")
print(f"   z-stat     : {results['mk_mk_z']:.2f}")
print(f"   Pente Sen  : {results['mk_sen_slope_mm_yr']:.2f} mm/an")
print(f"   En volume  : {results['sen_slope_km3_yr']:.2f} km³/an")

print("\n Plausibilité OSS")
lo, hi = results["oss_expected_deficit_km3_yr"]
obs_rate = abs(results["ols_slope_km3_yr"])
print(f"   Déficit attendu (OSS) : {lo}-{hi} km³/an")
print(f"   Taux observé (OLS)    : {obs_rate:.2f} km³/an")
if lo <= obs_rate <= hi:
    print("    Cohérent avec la fourchette OSS.")
else:
    print("     Hors fourchette — à documenter dans le rapport (spec §6.1).")
    print("   Pistes : GRACE mesure le stockage total (pas les prélèvements),")
    print("   erreur résiduelle GLDAS, convention d'anomalie, réponse retardée,")
    print("   décalage polygone vs. figures administratives.")