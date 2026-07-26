# %% [markdown]
# # Test carte SASS améliorée — Contour interpolé + masquage AOI
# Objectif : remplacer le heatmap blocky par un contour lissé professionnel

# %%
import numpy as np
import xarray as xr
import geopandas as gpd
import plotly.graph_objects as go
from scipy.ndimage import zoom
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from pipeline import config

# %%  Charger données
ds = xr.open_dataset(config.MASCON_NC_PATH)
ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
aoi = gpd.read_file(config.AOI_GEOJSON)
poly = aoi.geometry.iloc[0]

# Clip avec rioxarray AVANT interpolation (supprime les valeurs hors AOI)
lwe = ds["lwe_thickness"].sel(
    lat=slice(config.BBOX_LAT_MIN, config.BBOX_LAT_MAX),
    lon=slice(config.BBOX_LON_MIN, config.BBOX_LON_MAX),
)
lwe = lwe.rio.write_crs("EPSG:4326")
lwe_clipped = lwe.rio.clip(aoi.geometry, aoi.crs, all_touched=True, drop=False)

last_field = lwe_clipped.isel(time=-1)
last_date = str(last_field.time.values)[:7]
vals_mm = last_field.values * 10.0  # cm → mm

lons = last_field.lon.values
lats = last_field.lat.values

print(f"Original grid: {vals_mm.shape}, date: {last_date}")
valid_vals = vals_mm[~np.isnan(vals_mm)]
print(f"Value range (clipped): {valid_vals.min():.1f} to {valid_vals.max():.1f} mm")

# %% Remplacer NaN par la moyenne locale pour une interpolation propre,
#    puis masquer après interpolation
fill_val = np.nanmean(vals_mm)
vals_filled = np.where(np.isnan(vals_mm), fill_val, vals_mm)

# Interpolation cubique vers grille fine
zoom_factor = 10  # 0.5° → 0.05°
vals_fine = zoom(vals_filled, zoom_factor, order=3)
lons_fine = np.linspace(lons[0], lons[-1], vals_fine.shape[1])
lats_fine = np.linspace(lats[0], lats[-1], vals_fine.shape[0])

print(f"Interpolated grid: {vals_fine.shape}")

# %% Masque rapide avec rasterio.features.rasterize
transform = from_bounds(
    lons_fine[0], lats_fine[0], lons_fine[-1], lats_fine[-1],
    vals_fine.shape[1], vals_fine.shape[0],
)
# rasterize : 1 inside polygon, 0 outside
mask_arr = rasterize(
    [(poly, 1)],
    out_shape=vals_fine.shape,
    transform=transform,
    fill=0,
    dtype=np.uint8,
)
# Flip : rasterize uses top-down, our array is bottom-up (lat ascending)
mask_arr = mask_arr[::-1]

vals_masked = vals_fine.copy()
vals_masked[mask_arr == 0] = np.nan
valid_masked = vals_masked[~np.isnan(vals_masked)]
print(f"Masked cells: {(mask_arr == 0).sum()} / {mask_arr.size}")
print(f"Final range: {valid_masked.min():.1f} to {valid_masked.max():.1f} mm")

# %% Contour plot
fig = go.Figure()

vmin, vmax = valid_masked.min(), valid_masked.max()
v_abs = max(abs(vmin), abs(vmax))

fig.add_trace(go.Contour(
    z=vals_masked,
    x=lons_fine,
    y=lats_fine,
    colorscale="RdBu",
    reversescale=True,
    zmin=-v_abs,
    zmax=v_abs,
    contours=dict(
        start=-v_abs,
        end=v_abs,
        size=v_abs * 2 / 20,
        showlines=True,
        showlabels=False,
    ),
    line=dict(width=0.3, color="rgba(0,0,0,0.15)"),
    colorbar=dict(
        title=dict(text="GWSA (mm)", font=dict(size=11)),
        thickness=14,
        len=0.85,
        tickfont=dict(size=10),
        tickformat=".0f",
    ),
    hovertemplate="lon: %{x:.2f}°  lat: %{y:.2f}°<br>GWSA: %{z:.1f} mm<extra></extra>",
))

# Contour SASS boundary
coords = list(poly.exterior.coords)
lons_aoi = [c[0] for c in coords]
lats_aoi = [c[1] for c in coords]
fig.add_trace(go.Scatter(
    x=lons_aoi, y=lats_aoi,
    mode="lines",
    line=dict(width=2.5, color="#1B3A4B"),
    name="SASS boundary",
    hoverinfo="skip",
))

# Cities
cities = [
    {"name": "Ouargla", "lon": 5.33, "lat": 31.95},
    {"name": "Ghardaïa", "lon": 3.67, "lat": 32.49},
    {"name": "Tozeur", "lon": 8.13, "lat": 33.92},
    {"name": "Ghadames", "lon": 9.50, "lat": 30.13},
    {"name": "In Salah", "lon": 2.47, "lat": 27.19},
]
fig.add_trace(go.Scatter(
    x=[c["lon"] for c in cities],
    y=[c["lat"] for c in cities],
    mode="markers+text",
    marker=dict(size=7, color="#0F172A", symbol="circle",
                line=dict(width=1.5, color="white")),
    text=[c["name"] for c in cities],
    textposition="top center",
    textfont=dict(size=9, color="#0F172A", family="Inter, sans-serif"),
    name="Cities",
    hovertemplate="%{text}<extra></extra>",
))

fig.update_layout(
    xaxis=dict(
        title="Longitude (°)",
        range=[config.BBOX_LON_MIN - 0.5, config.BBOX_LON_MAX + 0.5],
        scaleanchor="y", scaleratio=1,
        showgrid=False,
    ),
    yaxis=dict(
        title="Latitude (°)",
        range=[config.BBOX_LAT_MIN - 0.5, config.BBOX_LAT_MAX + 0.5],
        showgrid=False,
    ),
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12,
              color=config.COLORS["text"]),
    margin=dict(l=50, r=20, t=35, b=40),
    height=420,
    showlegend=False,
    title=dict(
        text=f"GWSA Spatial Anomaly — {last_date}",
        font=dict(size=12, color=config.COLORS["text_mid"]),
        x=0.01,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
)

fig.write_html("test_map_v2.html", auto_open=True)
print("Done — test_map_v2.html")
ds.close()
