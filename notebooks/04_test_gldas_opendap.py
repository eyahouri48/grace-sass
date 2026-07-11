# fichier : notebooks/04_test_gldas_opendap.py
"""Test d'accès GLDAS via earthaccess — Tâche 3.

L'ancien serveur OPeNDAP (hydro1.gesdisc...) est retiré (410 Gone).
On utilise la solution de repli : earthaccess search + download par granule.

"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import earthaccess
import xarray as xr
from pipeline.config import GLDAS_COMPONENTS

# --- 1. Authentification ---
earthaccess.login(strategy="environment")

# --- 2. Rechercher les granules GLDAS_NOAH025_M v2.1 ---
print("Recherche des granules GLDAS_NOAH025_M v2.1...")
results = earthaccess.search_data(
    short_name="GLDAS_NOAH025_M",
    version="2.1",
    temporal=("2000-01-01", "2024-12-31"),  # toute la période
)
print(f"Nombre de granules trouvés : {len(results)}")

if len(results) > 0:
    # Afficher le premier et le dernier pour voir la couverture
    print(f"\nPremier granule : {results[0]}")
    print(f"\nDernier granule : {results[-1]}")

    # --- 3. Télécharger UN SEUL granule pour inspection ---
    print("\n--- Téléchargement d'un seul granule pour inspection ---")
    test_dir = Path("data/raw/gldas_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Prendre le premier granule (janvier 2000)
    downloaded = earthaccess.download([results[0]], str(test_dir))
    print(f"Fichier téléchargé : {downloaded}")

    # --- 4. Ouvrir et inspecter ---
    if downloaded:
        nc_file = Path(downloaded[0])
        print(f"\nTaille du fichier : {nc_file.stat().st_size / 1e6:.1f} Mo")

        ds = xr.open_dataset(nc_file)
        print(f"\nVariables disponibles ({len(ds.data_vars)}) :")
        for v in sorted(ds.data_vars):
            print(f"  {v} : {ds[v].dims} | {ds[v].attrs.get('units', '?')}")

        print(f"\nDimensions : {dict(ds.dims)}")
        print(f"Latitude : {float(ds.lat.min()):.2f} → {float(ds.lat.max()):.2f}")
        print(f"Longitude : {float(ds.lon.min()):.2f} → {float(ds.lon.max()):.2f}")

        # Vérifier nos 6 composantes
        print("\n--- Vérification des 6 composantes attendues ---")
        available = set(ds.data_vars)
        for comp in GLDAS_COMPONENTS:
            status = "✅" if comp in available else "❌ MANQUANTE"
            print(f"  {comp} : {status}")

        # Chercher les noms en minuscules si manquantes
        missing = [c for c in GLDAS_COMPONENTS if c not in available]
        if missing:
            lower_map = {v.lower(): v for v in available}
            for m in missing:
                if m.lower() in lower_map:
                    print(f"  → '{m}' existe sous le nom : '{lower_map[m.lower()]}'")

        ds.close()
else:
    print("❌ Aucun granule trouvé. Vérifie le short_name et la version.")