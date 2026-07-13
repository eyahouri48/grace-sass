# fichier : tests/test_ingest.py
"""
Tests unitaires — pipeline/ingest_grace.py + pipeline/ingest_gldas.py

On crée des NetCDF synthétiques (xarray → fichier temporaire) pour tester
les fonctions de traitement HORS LIGNE. Pas de téléchargement, pas d'Earthdata.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pipeline.ingest_gldas import process_one_granule
from pipeline.ingest_grace import extract_twsa_basin_mean, save_twsa_parquet


# ── Helpers ─────────────────────────────────────────────────────


def _make_gldas_granule(
    tmp_path,
    lat_range=(24.0, 35.0),
    lon_range=(-3.0, 19.0),
    step=1.0,
    time="2005-06-01",
    soil_value=100.0,
    swe_value=0.0,
    canopy_value=2.0,
    inject_nan_pixel=False,
):
    """Crée un faux granule GLDAS NetCDF avec des valeurs contrôlées.

    Parameters
    ----------
    tmp_path : Path
        Dossier temporaire pytest.
    soil_value : float
        Valeur identique pour les 4 couches d'humidité du sol (mm).
    swe_value : float
        Valeur SWE (mm).
    canopy_value : float
        Valeur canopy (mm).
    inject_nan_pixel : bool
        Si True, met un pixel à NaN dans une composante
        → la somme doit donner NaN pour ce pixel (skipna=False).

    Returns
    -------
    Path
        Chemin vers le fichier NetCDF créé.
    """
    lats = np.arange(lat_range[0], lat_range[1] + step, step)
    lons = np.arange(lon_range[0], lon_range[1] + step, step)

    shape = (1, len(lats), len(lons))

    def _full(val):
        arr = np.full(shape, val, dtype=np.float32)
        if inject_nan_pixel:
            arr[0, 0, 0] = np.nan
        return arr

    ds = xr.Dataset(
        {
            "SoilMoi0_10cm_inst": (["time", "lat", "lon"], _full(soil_value)),
            "SoilMoi10_40cm_inst": (["time", "lat", "lon"], _full(soil_value)),
            "SoilMoi40_100cm_inst": (["time", "lat", "lon"], _full(soil_value)),
            "SoilMoi100_200cm_inst": (["time", "lat", "lon"], _full(soil_value)),
            "SWE_inst": (["time", "lat", "lon"], _full(swe_value)),
            "CanopInt_inst": (["time", "lat", "lon"], _full(canopy_value)),
        },
        coords={
            "time": [np.datetime64(time)],
            "lat": lats,
            "lon": lons,
        },
    )

    nc_path = tmp_path / f"GLDAS_{time}.nc"
    ds.to_netcdf(nc_path)
    return nc_path


# ── Tests process_one_granule ───────────────────────────────────


class TestProcessOneGranule:
    """Vérifier le traitement d'un granule GLDAS sur une grille synthétique."""

    def test_somme_composantes_valeur_connue(self, tmp_path):
        """4 couches sol à 100 + SWE 0 + canopée 2 = 402 mm par pixel.

        Sur une grille uniforme, la moyenne de bassin doit ≈ 402 mm
        (pas exactement 402 car la pondération cos-lat varie légèrement).
        """
        nc = _make_gldas_granule(tmp_path, soil_value=100.0, swe_value=0.0, canopy_value=2.0)
        result = process_one_granule(nc)

        assert result is not None
        # 4×100 + 0 + 2 = 402 — la moyenne pondérée cos-lat sera très proche
        assert abs(result["gldas_mm"] - 402.0) < 1.0  # tolérance pour cos-lat

    def test_date_extraite_correctement(self, tmp_path):
        """La date du granule est correctement extraite."""
        nc = _make_gldas_granule(tmp_path, time="2008-03-01")
        result = process_one_granule(nc)

        assert result is not None
        assert result["time"].year == 2008
        assert result["time"].month == 3

    def test_skipna_false_propage_nan(self, tmp_path):
        """Si un pixel a un NaN dans une composante, il est exclu de la moyenne.

        Avec skipna=False dans la somme, un pixel NaN → somme NaN pour ce pixel.
        La moyenne pondérée exclut alors ce pixel (xarray weighted.mean
        ignore les NaN par défaut), donc le résultat change légèrement.
        L'important : ça ne plante PAS, et le résultat reste plausible.
        """
        # Sans NaN
        nc_clean = _make_gldas_granule(
            tmp_path, inject_nan_pixel=False, soil_value=100.0, time="2005-06-01",
        )
        result_clean = process_one_granule(nc_clean)

        # Avec un pixel NaN — utiliser un mois différent pour un nom de fichier distinct
        nc_nan = _make_gldas_granule(
            tmp_path, inject_nan_pixel=True, soil_value=100.0, time="2005-07-01",
        )
        result_nan = process_one_granule(nc_nan)

        assert result_nan is not None
        # Le résultat doit être un nombre valide (pas NaN global)
        assert not np.isnan(result_nan["gldas_mm"])
        # Les deux résultats sont proches (un seul pixel exclu sur ~250)
        assert abs(result_nan["gldas_mm"] - result_clean["gldas_mm"]) < 5.0

    def test_fichier_inexistant_retourne_none(self, tmp_path):
        """Un chemin invalide retourne None (pas de crash)."""
        fake_path = tmp_path / "inexistant.nc"
        result = process_one_granule(fake_path)
        assert result is None


# =====================================================================
# Tests GRACE — extract_twsa_basin_mean + save_twsa_parquet
# =====================================================================


def _make_grace_mascon_nc(tmp_path, lon_convention="0_360"):
    """Crée un faux mascon CRI NetCDF avec des valeurs connues.

    Grille 4×4 très grossière, 3 pas de temps, variable lwe_thickness (cm).

    Parameters
    ----------
    tmp_path : Path
        Dossier temporaire pytest.
    lon_convention : str
        "0_360"   → longitudes en 0–360° (comme le vrai GRACE)
        "-180_180" → longitudes déjà en -180/180° (pour contrôle)

    La grille couvre la zone [25–34°N, -2–18°E] soit 4 lats × 4 lons.
    En convention 0–360°, les longitudes négatives deviennent 358° et 359°.
    """
    lats = np.array([25.0, 28.0, 31.0, 34.0])

    if lon_convention == "0_360":
        # -2° → 358°, 4° → 4°, 11° → 11°, 18° → 18°
        # ⚠ L'ordre naturel en 0-360 place 4, 11, 18 AVANT 358
        lons = np.array([4.0, 11.0, 18.0, 358.0])
    else:
        lons = np.array([-2.0, 4.0, 11.0, 18.0])

    times = pd.date_range("2005-01-15", periods=3, freq="ME")

    # Valeurs connues : chaque pas de temps a une constante spatiale
    # t0 = 1.0 cm, t1 = 2.0 cm, t2 = 3.0 cm partout
    data = np.array([
        np.full((4, 4), 1.0),
        np.full((4, 4), 2.0),
        np.full((4, 4), 3.0),
    ])

    ds = xr.Dataset(
        {"lwe_thickness": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    # Attributs CF pour que rioxarray identifie les dimensions spatiales
    ds["lat"].attrs["axis"] = "Y"
    ds["lon"].attrs["axis"] = "X"

    nc_path = tmp_path / "mascon_test.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _make_simple_aoi_geojson(tmp_path):
    """Crée un GeoJSON rectangulaire couvrant [26–33°N, -1–17°E].

    Plus petit que la grille pour vérifier que le clip fonctionne.
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"aquifer_id": "test"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-1.0, 26.0], [17.0, 26.0], [17.0, 33.0],
                    [-1.0, 33.0], [-1.0, 26.0],
                ]],
            },
        }],
    }
    path = tmp_path / "test_aoi.geojson"
    path.write_text(json.dumps(geojson))
    return path


class TestExtractTwsaBasinMean:
    """Vérifier la chaîne complète : ouverture → conversion lon → clip → moyenne."""

    def test_conversion_longitude_0_360(self, tmp_path, monkeypatch):
        """Un NetCDF en 0–360° doit produire le même résultat qu'en -180/180.

        C'est LE piège principal de l'ingestion GRACE (spec §4a) :
        la grille mascon utilise 0–360°, le SASS croise le méridien 0°
        (longitudes ~357–360° + 0–19°). Sans conversion, un slice(-3, 19)
        sur une grille 0–360 retourne un sous-ensemble vide ou faux.
        """
        aoi_path = _make_simple_aoi_geojson(tmp_path)
        monkeypatch.setattr("pipeline.ingest_grace.AOI_GEOJSON", aoi_path)

        # Créer le même NetCDF dans les deux conventions de longitude
        nc_0_360 = _make_grace_mascon_nc(tmp_path, lon_convention="0_360")

        sub = tmp_path / "sub"
        sub.mkdir()
        nc_180 = _make_grace_mascon_nc(sub, lon_convention="-180_180")

        result_0_360 = extract_twsa_basin_mean(nc_0_360)
        result_180 = extract_twsa_basin_mean(nc_180)

        # Les deux conventions doivent donner le même résultat
        np.testing.assert_array_almost_equal(
            result_0_360.values, result_180.values, decimal=5,
        )

    def test_valeurs_connues(self, tmp_path, monkeypatch):
        """Sur une grille spatiale constante (1, 2, 3 cm), la moyenne = la constante.

        Si tous les pixels valent X cm à un pas de temps donné, la moyenne
        pondérée cos-lat doit aussi valoir X cm (le poids varie mais la
        valeur est uniforme → la moyenne reste X).
        """
        aoi_path = _make_simple_aoi_geojson(tmp_path)
        monkeypatch.setattr("pipeline.ingest_grace.AOI_GEOJSON", aoi_path)

        nc_path = _make_grace_mascon_nc(tmp_path, lon_convention="0_360")
        result = extract_twsa_basin_mean(nc_path)

        # t0 = 1.0 cm partout → moyenne = 1.0
        # t1 = 2.0 cm partout → moyenne = 2.0
        # t2 = 3.0 cm partout → moyenne = 3.0
        np.testing.assert_array_almost_equal(
            result.values, [1.0, 2.0, 3.0], decimal=4,
        )

    def test_nom_et_type_serie(self, tmp_path, monkeypatch):
        """La sortie est une pd.Series nommée 'twsa_cm'."""
        aoi_path = _make_simple_aoi_geojson(tmp_path)
        monkeypatch.setattr("pipeline.ingest_grace.AOI_GEOJSON", aoi_path)

        nc_path = _make_grace_mascon_nc(tmp_path)
        result = extract_twsa_basin_mean(nc_path)

        assert isinstance(result, pd.Series)
        assert result.name == "twsa_cm"
        assert len(result) == 3


class TestSaveTwsaParquet:
    """Vérifier le format du cache Parquet (lu par proxy.py)."""

    def test_roundtrip_parquet(self, tmp_path):
        """Écrire puis relire le Parquet — les données sont préservées."""
        index = pd.date_range("2005-01-01", periods=4, freq="MS")
        original = pd.Series([1.0, 2.0, 3.0, 4.0], index=index, name="twsa_cm")
        original.index.name = "time"

        dest = tmp_path / "test_twsa.parquet"
        save_twsa_parquet(original, dest=dest)

        # Relire exactement comme proxy.py le fait
        df = pd.read_parquet(dest)
        assert "twsa_cm" in df.columns
        assert df.index.name == "time"
        np.testing.assert_array_equal(df["twsa_cm"].values, [1.0, 2.0, 3.0, 4.0])