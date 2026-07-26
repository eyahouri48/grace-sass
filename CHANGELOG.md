# Changelog

Tous les changements notables du projet sont documentés ici.
Format : [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] — 2026-07-26
### Modifié
- `pyproject.toml` : ajout shapely, pyproj comme dépendances explicites
- `Makefile` : ajout cibles help, lint-fix, clean
- `README.md` : documentation complète (installation, usage, architecture, limites)
- `load_data()` : applique le prétraitement + tronque les lignes pré-GRACE

## [0.6.0] — 2026-07-26
### Ajouté
- `pipeline/refresh.py` : orchestrateur d'actualisation append-only
- CI/CD : `refresh-dashboard.yml` (cron hebdomadaire) + `ci.yml` (pytest + ruff)
- GitHub Pages : dashboard statique accessible en ligne
- CMR dynamique pour GRACE (résolution d'URL à l'exécution)
- Tests d'intégration : pipeline bout en bout + isolation réseau

## [0.5.0] — 2026-07-19
### Ajouté
- Dashboard MVP + stretch : série temporelle, KPI cards, carte AOI, prévision,
  décomposition STL, scénarios, vues Décision et Expert
- `pipeline/build_dashboard.py` : rendu HTML statique (Jinja2 + Plotly)
- Interface bilingue EN/FR avec bascule de langue
- Avertissement prototype visible dans l'interface
- Horodatage de fraîcheur à 3 lignes

## [0.4.0] — 2026-07-17
### Ajouté
- `pipeline/forecast.py` : Prophet + SARIMA + ETS + validation glissante
- `pipeline/scenarios.py` : cadrage scénarios (horizon validé vs extrapolation)
- Métriques MAE/RMSE reportées en mm
- Tests : forecast (prepare, scoring, horizon) + scenarios (zones, IC, jalons)

## [0.3.0] — 2026-07-15
### Ajouté
- `pipeline/trend.py` : OLS+HAC, Mann-Kendall saisonnier, pente de Sen, conversion volume
- `pipeline/decomposition.py` : STL (period=12)
- `pipeline/indicators.py` : z-score, percentile vs 2004–2009
- Vérification de plausibilité vs chiffres OSS
- Tests : trend (OLS, Sen, MK, aire, volume) + indicators (z-score, percentile)

### Corrigé
- Pente de Sen : suppression de la multiplication x12 erronée (déjà en mm/an)
- Test de significativité : vérifie si zéro est exclu de l'IC

## [0.2.0] — 2026-07-12
### Ajouté
- `pipeline/proxy.py` : proxy GWSA = TWSA − anomalie GLDAS (Option B-lite)
- `pipeline/preprocessing.py` : réindexation mensuelle, interpolation, is_imputed
- Alignement de référence GLDAS sur mois réels GRACE 2004–2009
- Disclaimer prototype rédigé
- Tests : proxy (baseline, unités, soustraction) + preprocessing + ingest (29 tests)

## [0.1.0] — 2026-07-11
### Ajouté
- `pipeline/ingest_grace.py` : ingestion GRACE HTTPS + clip AOI
- `pipeline/ingest_gldas.py` : ingestion GLDAS-2.1 Noah + moyenne de bassin
- Cache Parquet local
- Polygone SASS officiel intégré

## [0.0.1] — 2026-07-07
### Ajouté
- Arborescence complète du dépôt (squelette initial)
- `config.py` : constantes de départ
- `sass.geojson` (placeholder)
- Libellés bilingues `ui_strings/{en,fr}.json` + test de parité
- Workflows CI/CD (squelette)
- Suite de tests hors ligne (fixture Earthdata retirée)