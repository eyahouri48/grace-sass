# Commandes du projet (équivalents uv run documentés)
.PHONY: install test lint refresh dashboard

install:        ## installe l'environnement
	uv sync

test:           ## suite de tests (100 % hors ligne)
	uv run pytest -v

lint:           ## vérification de style
	uv run ruff check .

refresh:        ## ingestion GRACE + GLDAS + recalcul du proxy (Earthdata requis)
	uv run python -m pipeline.refresh

dashboard:      ## rendu statique -> docs/index.html
	uv run python -m pipeline.build_dashboard
