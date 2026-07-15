# fichier : notebooks/09_stl_acf_pacf.py
"""Exploration Tâche 8 — Décomposition STL + diagnostics ACF/PACF.

Produit 3 graphiques :
1. Panneau STL (4 sous-graphiques : observed, trend, seasonal, resid)
2. ACF du résidu
3. PACF du résidu
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pipeline.config import SERIES_PARQUET, ACF_NLAGS
from pipeline.decomposition import run_stl, compute_acf_pacf


# ── 1. Charger la série interpolée ──────────────────────────────────
df = pd.read_parquet(SERIES_PARQUET)
gwsa = df["gwsa_mm"].copy()

# STL a besoin d'une série continue — on utilise la version interpolée
# (les NaN ont été comblés en Tâche 5, signalés par is_imputed)
if gwsa.isna().any():
    print(f"⚠ {gwsa.isna().sum()} NaN détectés — interpolation linéaire")
    gwsa = gwsa.interpolate(method="time")


# ── 2. Décomposition STL ────────────────────────────────────────────
stl_df = run_stl(gwsa)

# Vérification : observed ≈ trend + seasonal + resid ?
reconstruction = stl_df["trend"] + stl_df["seasonal"] + stl_df["resid"]
max_err = (stl_df["observed"] - reconstruction).abs().max()
print(f"Erreur max de reconstruction : {max_err:.2e} mm (doit être ≈ 0)")

# Amplitude saisonnière
amp = stl_df["seasonal"].max() - stl_df["seasonal"].min()
print(f"Amplitude saisonnière : {amp:.1f} mm")
print(f"  → {'Faible (attendu sur le SASS)' if amp < 10 else 'Significative'}")


# ── 3. Graphique STL — 4 panneaux superposés ────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Panneau 1 : série observée
axes[0].plot(stl_df.index, stl_df["observed"], color="steelblue", lw=1)
axes[0].set_ylabel("gwsa_mm")
axes[0].set_title("Décomposition STL de la GWSA — SASS", fontsize=13, fontweight="bold")

# Marquer la lacune 2017-2018
axes[0].axvspan("2017-06", "2018-06", color="grey", alpha=0.15, label="Lacune GRACE")
axes[0].legend(loc="upper right", fontsize=9)

# Panneau 2 : tendance
axes[1].plot(stl_df.index, stl_df["trend"], color="darkred", lw=1.5)
axes[1].set_ylabel("Tendance (mm)")

# Panneau 3 : saisonnalité
axes[2].plot(stl_df.index, stl_df["seasonal"], color="seagreen", lw=1)
axes[2].set_ylabel("Saisonnalité (mm)")
axes[2].axhline(0, color="grey", ls="--", lw=0.5)

# Panneau 4 : résidu
axes[3].plot(stl_df.index, stl_df["resid"], color="grey", lw=0.8)
axes[3].set_ylabel("Résidu (mm)")
axes[3].axhline(0, color="grey", ls="--", lw=0.5)
axes[3].set_xlabel("Date")

plt.tight_layout()
plt.savefig("notebooks/stl_decomposition.png", dpi=150)
plt.show()


# ── 4. ACF et PACF du résidu ────────────────────────────────────────
resid = stl_df["resid"]

fig2, (ax_acf, ax_pacf) = plt.subplots(2, 1, figsize=(12, 7))

plot_acf(resid, lags=ACF_NLAGS, ax=ax_acf, title="ACF du résidu STL")
ax_acf.set_xlabel("Lag (mois)")

plot_pacf(resid, lags=ACF_NLAGS, ax=ax_pacf, title="PACF du résidu STL",
          method="ywm")
ax_pacf.set_xlabel("Lag (mois)")

plt.tight_layout()
plt.savefig("notebooks/acf_pacf_residu.png", dpi=150)
plt.show()


# ── 5. Lecture rapide des résultats ─────────────────────────────────
acf_pacf = compute_acf_pacf(resid)

print("\n── Résumé ACF (5 premiers lags, hors lag 0) ──")
for i in range(1, 6):
    val = acf_pacf["acf_values"][i]
    lo, hi = acf_pacf["acf_confint"][i]
    sig = "✓ significatif" if lo > 0 or hi < 0 else "  non significatif"
    print(f"  Lag {i:2d} : ACF = {val:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")

print("\n── Résumé PACF (5 premiers lags, hors lag 0) ──")
for i in range(1, 6):
    val = acf_pacf["pacf_values"][i]
    lo, hi = acf_pacf["pacf_confint"][i]
    sig = "✓ significatif" if val < lo or val > hi else "  non significatif"
    print(f"  Lag {i:2d} : PACF = {val:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")

# Vérifier le lag 12 (saisonnalité résiduelle ?)
acf_12 = acf_pacf["acf_values"][12]
lo_12, hi_12 = acf_pacf["acf_confint"][12]
sig_12 = "✓ OUI" if lo_12 > 0 or hi_12 < 0 else "NON"
print("\n── Saisonnalité résiduelle au lag 12 ? ")
print(f"  ACF lag 12 = {acf_12:+.3f}  [{lo_12:+.3f}, {hi_12:+.3f}]  → {sig_12}")
print(f"  → {'Ajouter des termes saisonniers (P,D,Q)₁₂' if sig_12 == '✓ OUI' else 'Pas de signal saisonnier résiduel — ARIMA simple peut suffire'}")