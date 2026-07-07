# GRACE/SASS — Suivi & prévision du stockage des eaux souterraines

Pipeline Python reproductible + tableau de bord pour le suivi (« veille ») et
la prévision (« perspectives ») du stockage des eaux souterraines du
**Système Aquifère du Sahara Septentrional (SASS/NWSAS)**, à partir des
mascons GRACE/GRACE-FO (JPL RL06.3 V04, filtrés CRI) et d'un proxy résiduel
GLDAS-2.1 Noah. Projet OSS, programme « Veille et Perspectives ».

> ⚠️ **Estimation prototype** : `gwsa_mm ≈ TWSA − anomalie GLDAS` est une
> séparation de premier ordre, non validée par la piézométrie in situ.
> Visualisation de tendance et démonstration méthodologique uniquement.

## Arborescence

```
pipeline/            modules Python (une responsabilité par module)
├── config.py        toutes les constantes — ZÉRO valeur en dur ailleurs
├── ingest_grace.py  GRACE : HTTPS direct → découpe AOI → twsa_cm (Parquet)
├── ingest_gldas.py  GLDAS : OPeNDAP → 6 composantes → gldas_mm (Parquet)
├── proxy.py         alignement référence/unités → gwsa_mm
├── preprocessing.py réindexation, lacunes, is_imputed
├── trend.py         OLS+HAC, MK saisonnier, Sen, volume km³
├── indicators.py    z-score, percentile
├── decomposition.py STL + ACF/PACF
├── forecast.py      Prophet, SARIMA, CV à origine glissante
├── scenarios.py     horizon validé (~24 mois) vs extrapolation (60 mois)
├── refresh.py       actualisation append-only par source (CI)
├── build_dashboard.py  rendu statique → docs/index.html (Path A)
└── dashboard.py     Dash (dev local / Path B)
ui_strings/          libellés bilingues EN/FR (parité testée)
tests/               suite 100 % hors ligne (fixture Parquet committée)
data/                cache Parquet + last_refresh.json (générés)
docs/                dashboard statique publié sur GitHub Pages
sass.geojson         AOI — ⚠️ PLACEHOLDER, à remplacer par le polygone OSS
```

## Démarrage

```bash
uv sync                                # installe l'environnement
uv run pytest                          # suite de tests (hors ligne)
uv run python -m pipeline.refresh      # ingestion + proxy (Earthdata requis)
uv run python -m pipeline.build_dashboard  # → docs/index.html
```

Identifiants : un compte **NASA Earthdata Login** unique (gratuit), avec les
archives « PO.DAAC » et « NASA GESDISC DATA ARCHIVE » autorisées dans le
profil. En CI : secrets `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` (principaux,
→ `.netrc`), `EARTHDATA_TOKEN` optionnel (expire après 60 jours).

## Références

- Cahier des charges : `Spécification_Technique_Stage_GRACE.docx`
- OSS (2008), *Système aquifère du Sahara septentrional — gestion commune
  d'un bassin transfrontalier* (lecture semaine 1)
