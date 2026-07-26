# fichier : CHANGELOG.md

# Changelog

Tous les changements notables du projet sont documentés ici.

## [0.6.0] — YYYY-MM-DD (date du jour)
### Ajouté
- `pipeline/refresh.py` : orchestrateur d'actualisation append-only
- CI/CD : `refresh-dashboard.yml` (cron hebdomadaire) + `ci.yml` (pytest + ruff)
- GitHub Pages : dashboard statique accessible en ligne
- CMR dynamique pour GRACE (résolution d'URL à l'exécution)

## [0.5.0]
### Ajouté
- Dashboard MVP : série temporelle, KPI cards, carte AOI, prévision, horodatage de fraîcheur
- `pipeline/build_dashboard.py` : rendu HTML statique (Jinja2 + Plotly)
- Interface bilingue EN/FR avec bascule de langue
- Avertissement prototype visible dans l'interface

## [0.4.0]
### Ajouté
- `pipeline/forecast.py` : Prophet + SARIMA + validation glissante
- `pipeline/scenarios.py` : cadrage scénarios (horizon validé vs extrapolation)
- Métriques MAE/RMSE reportées en mm

## [0.3.0]
### Ajouté
- `pipeline/trend.py` : OLS+HAC, Mann-Kendall saisonnier, pente de Sen, conversion volume
- `pipeline/decomposition.py` : STL (period=12)
- `pipeline/indicators.py` : z-score, percentile
- Vérification de plausibilité vs chiffres OSS

## [0.2.0]
### Ajouté
- `pipeline/proxy.py` : proxy GWSA = TWSA − anomalie GLDAS (Option B-lite)
- `pipeline/preprocessing.py` : réindexation, interpolation, is_imputed
- Alignement de référence GLDAS sur mois réels GRACE 2004–2009
- Disclaimer prototype rédigé

## [0.1.0]
### Ajouté
- `pipeline/ingest_grace.py` : ingestion GRACE HTTPS + clip AOI
- `pipeline/ingest_gldas.py` : ingestion GLDAS OPeNDAP / HTTPS
- Cache Parquet local

## [0.0.1] — Squelette initial
### Ajouté
- Arborescence complète du dépôt
- `config.py` : constantes de départ
- `sass.geojson` (placeholder)
- Libellés bilingues `ui_strings/{en,fr}.json` + test de parité
- Workflows CI/CD (squelette)
- Suite de tests hors ligne (fixture Earthdata retirée)
