# Commandes du projet grace-sass (équivalents uv run documentés)
.PHONY: install test lint lint-fix refresh dashboard clean help

help:           ## affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:        ## installe l'environnement (uv sync)
	uv sync

test:           ## lance la suite de tests (100 % hors ligne)
	uv run pytest -v

lint:           ## vérification de style (ruff)
	uv run ruff check .

lint-fix:       ## corrige automatiquement les problèmes de style
	uv run ruff check --fix .

refresh:        ## ingestion GRACE + GLDAS + recalcul du proxy (Earthdata requis)
	uv run python -m pipeline.refresh

dashboard:      ## rendu statique → docs/index.html
	uv run python -m pipeline.build_dashboard

clean:          ## supprime les fichiers temporaires (pas le cache Parquet)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -f *.nc