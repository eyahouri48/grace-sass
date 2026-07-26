# GRACE/SASS : Groundwater-Storage Monitoring & Forecasting

Pipeline Python reproductible + tableau de bord pour le suivi (« veille »)
et la prévision (« perspectives ») du stockage des eaux souterraines du
**Système Aquifère du Sahara Septentrional (SASS/NWSAS)**, à partir des
mascons GRACE/GRACE-FO (JPL RL06.3 V04, filtrés CRI) et d'un proxy
résiduel GLDAS-2.1 Noah.

Projet [OSS](https://www.oss-online.org/) : programme « Veille et Perspectives ».

> ⚠️ **Estimation prototype** : `gwsa_mm ≈ TWSA − anomalie GLDAS` est une
> séparation de premier ordre, non validée par la piézométrie in situ.
> Visualisation de tendance et démonstration méthodologique uniquement.

---

## Prérequis

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — gestionnaire de packages
  (installer : `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Compte NASA Earthdata** (gratuit) — nécessaire uniquement pour
  l'actualisation des données, pas pour la consultation du dashboard.
  Créer le compte sur [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/),
  puis autoriser **PO.DAAC** et **NASA GESDISC DATA ARCHIVE** dans le profil.

## Installation

```bash
git clone https://github.com/eyahouri48/grace-sass.git
cd grace-sass
uv sync              # installe toutes les dépendances
uv run pytest -v     # vérifie que la suite de tests passe (100 % hors ligne)
```

## Utilisation

### Consulter le dashboard (sans credentials Earthdata)

Le dashboard statique est pré-généré dans `docs/index.html`.
Ouvrir directement dans un navigateur, ou accéder à la version en ligne :
**[https://eyahouri48.github.io/grace-sass/]()**

### Actualiser les données (credentials Earthdata requis)

```bash
export EARTHDATA_USERNAME="..."
export EARTHDATA_PASSWORD="..."
make refresh         # ingestion GRACE + GLDAS → recalcul du proxy
make dashboard       # régénère le dashboard statique
```

### Commandes disponibles

```bash
make help            # liste toutes les commandes
make install         # installe l'environnement
make test            # lance les tests (hors ligne)
make lint            # vérifie le style (ruff)
make refresh         # actualise les données (Earthdata requis)
make dashboard       # génère docs/index.html
make clean           # supprime les fichiers temporaires
```

## Architecture du pipeline

```
┌───────────────────────────────────────────────────┐
│  PRÉSENTATION  Plotly (HTML statique, Path A)     │
│   • série temporelle GWSA + prévision + IC        │
│   • KPI : tendance (mm/an, km³/an), p-value MK   │
│   • carte AOI / grille mascons                    │
│   • horodatage de fraîcheur à 3 lignes            │
├───────────────────────────────────────────────────┤
│  ANALYSE  modules Python testables                │
│   • proxy.py  (TWSA − GLDAS → gwsa_mm)           │
│   • trend.py  (OLS+HAC, Sen, MK saisonnier)      │
│   • decomposition.py  (STL)                       │
│   • forecast.py  (Prophet, SARIMA, CV glissante)  │
├───────────────────────────────────────────────────┤
│  INGESTION/CACHE                                  │
│   • GRACE mascon CRI via HTTPS → Parquet          │
│   • GLDAS-Noah via OPeNDAP/HTTPS → Parquet       │
│   • Actualisation automatique (GitHub Actions)    │
└───────────────────────────────────────────────────┘
```


## Structure du dépôt

```
grace-sass/
├── pipeline/                # code source du pipeline
│   ├── config.py            # toutes les constantes (chemins, seuils, palette)
│   ├── ingest_grace.py      # ingestion GRACE (HTTPS + clip AOI)
│   ├── ingest_gldas.py      # ingestion GLDAS (OPeNDAP / HTTPS)
│   ├── proxy.py             # proxy GWSA = TWSA − anomalie GLDAS
│   ├── preprocessing.py     # réindexation, lacunes, is_imputed
│   ├── trend.py             # OLS+HAC, Mann-Kendall, Sen, volume
│   ├── indicators.py        # z-score, percentile
│   ├── decomposition.py     # STL (period=12)
│   ├── forecast.py          # Prophet, SARIMA, validation glissante
│   ├── scenarios.py         # cadrage scénarios (validé vs extrapolation)
│   ├── refresh.py           # actualisation append-only (CI)
│   └── build_dashboard.py   # rendu HTML statique (Path A)
├── ui_strings/              # libellés bilingues EN/FR
├── tests/                   # suite de tests (100 % hors ligne)
├── data/                    # cache Parquet (committé) + last_refresh.json
├── docs/                    # dashboard statique (cible GitHub Pages)
├── sass.geojson             # emprise AOI du SASS
├── pyproject.toml           # dépendances (uv)
└── Makefile                 # commandes documentées
```


## Données

| Source | Rôle | Accès |
|--------|------|-------|
| JPL Mascon RL06.3 V04 (CRI) | TWSA mensuelle (cm) | PO.DAAC / Earthdata |
| GLDAS-2.1 Noah 0.25° mensuel | Composantes de surface (mm) | GES DISC / Earthdata |
| sass.geojson | Emprise AOI du SASS | Fourni (OSS/IGRAC) |

## Limites documentées

- **Résolution spatiale** : ~300 km (mascons) — résultats à l'échelle du bassin uniquement
- **Proxy prototype** : gwsa_mm hérite de l'erreur GLDAS, non validé in situ
- **Stationnarité** : les prévisions extrapolent le régime historique
- **Non-séparabilité verticale** : GRACE ne distingue pas le CI du CT
- **Lacune 2017–2018** : interpolée et signalée, pas observée

## Automatisation (GitHub Actions)

- **`refresh-dashboard.yml`** : actualisation hebdomadaire (lundi 06:00 UTC),
  append-only par source, rendu statique → GitHub Pages
- **`ci.yml`** : pytest + ruff à chaque push/PR (100 % hors ligne, sans secrets)

## Licence

*À compléter selon la politique OSS.*
