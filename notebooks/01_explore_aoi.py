# fichier : notebooks/01_explore_aoi.py
# Script d'exploration de l'AOI — Tâche 1


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from pipeline.config import AOI_GEOJSON

# --- Étape 1 : Charger le polygone AOI ---
print("=== Chargement du sass.geojson ===")
aoi = gpd.read_file(AOI_GEOJSON)

print(f"Nombre de features : {len(aoi)}")
print(f"CRS : {aoi.crs}")
print(f"\nPropriétés :")
print(aoi.drop(columns="geometry"))
print(f"\nBounding box :")
print(aoi.bounds)

# --- Étape 2 : Vérifier le CRS ---
print(f"\n=== CRS ===")
print(f"CRS : {aoi.crs}")
# Si None → le GeoJSON n'a pas déclaré de CRS, mais par convention c'est WGS84

# --- Étape 3 : Calculer la surface (CORRECTEMENT) ---
print(f"\n=== Surface ===")


# Projeter en équi-surface d'abord
aoi_proj = aoi.to_crs("ESRI:54009")  # Mollweide (projection équi-surface)
area_m2 = aoi_proj.geometry.area.iloc[0]
area_km2 = area_m2 / 1e6
print(f"Surface CORRECTE (Mollweide) : {area_km2:,.0f} km²")
print(f"  → Attendu : ~1 000 000 km² (le placeholder est approximatif)")
print(f"  → 1 mm sur cette surface ≈ {area_km2 / 1e6:.2f} km³")

# --- Étape 4 : Visualiser le polygone ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
aoi.plot(ax=ax, edgecolor="black", facecolor="lightblue", alpha=0.5)
ax.set_title("SASS — Emprise de l'AOI (placeholder)")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.grid(True, alpha=0.3)

# Annoter la bounding box
bounds = aoi.bounds.iloc[0]
ax.axhline(y=bounds.miny, color="red", linestyle="--", alpha=0.3, label="Bounding box")
ax.axhline(y=bounds.maxy, color="red", linestyle="--", alpha=0.3)
ax.axvline(x=bounds.minx, color="red", linestyle="--", alpha=0.3)
ax.axvline(x=bounds.maxx, color="red", linestyle="--", alpha=0.3)

# Marquer le méridien 0° — notre zone le CROISE
ax.axvline(x=0, color="green", linestyle="-", alpha=0.5, label="Méridien 0° (Greenwich)")
ax.legend()

plt.tight_layout()
plt.savefig("notebooks/aoi_preview.png", dpi=150)
print("\n→ Carte sauvegardée dans notebooks/aoi_preview.png")
plt.show()
