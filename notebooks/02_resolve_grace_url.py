# fichier : notebooks/02_resolve_grace_url.py
# Résoudre l'URL du fichier mascon CRI via l'API CMR de la NASA


import requests

# Rechercher le mascon CRI dans le catalogue CMR
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
params = {
    "short_name": "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
    "page_size": 5,
}

resp = requests.get(CMR_URL, params=params, timeout=30)
resp.raise_for_status()
granules = resp.json()["feed"]["entry"]

print(f"Nombre de granules trouvés : {len(granules)}")
for g in granules:
    title = g.get("title", "?")
    # Chercher le lien de téléchargement direct
    links = [l["href"] for l in g.get("links", [])
             if l.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"
             and l["href"].endswith(".nc")]
    print(f"\nTitre : {title}")
    for link in links:
        print(f"  URL : {link}")