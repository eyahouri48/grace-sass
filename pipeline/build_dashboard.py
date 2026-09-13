# fichier : pipeline/build_dashboard.py
"""
Orchestrateur du dashboard statique (Path A — GitHub Pages).

Charge les données, appelle les fonctions de dashboard.py,
remplit le template HTML, et écrit docs/index.html.
"""

from datetime import datetime, timezone
from pathlib import Path

import plotly.io as pio
from jinja2 import Template

from pipeline import config
from pipeline.dashboard import (
    compute_forecast_milestones,
    compute_kpis,
    compute_scenario_comparison,
    compute_trend_table,
    load_data,
    load_freshness,
    load_strings,
    make_annual_bar_figure,
    make_aoi_map,
    make_decision_mini_bar,
    make_expert_scenario_figure,
    make_gldas_contribution_figure,
    make_multi_scenario_figure,
    make_seasonal_bar_figure,
    make_sparkline_data,
    make_stl_figure,
    make_timeseries_figure,
)


def _compute_sync_status(freshness: dict) -> dict:
    """Détermine le statut SYNC à partir de la fraîcheur et des erreurs."""
    now = datetime.now(timezone.utc)

    if freshness.get("grace_status") == "error" or \
       freshness.get("gldas_status") == "error":
        return {"label": "error"}

    thresholds = [
        ("last_grace_month", config.GRACE_EXPECTED_LAG_MONTHS),
        ("last_gldas_month", config.GLDAS_EXPECTED_LAG_MONTHS),
    ]
    for key, max_lag in thresholds:
        month_str = freshness.get(key)
        if not month_str or month_str == "N/A":
            return {"label": "stale"}
        last = datetime.strptime(month_str, "%Y-%m").replace(tzinfo=timezone.utc)
        age_months = (now.year - last.year) * 12 + (now.month - last.month)
        if age_months > max_lag:
            return {"label": "stale"}

    return {"label": "ok"}


def build():
    """Point d'entrée : génère docs/index.html."""
    print("[1/5] Chargement des données...")
    df = load_data()
    strings_en, strings_fr = load_strings("en"), load_strings("fr")
    freshness = load_freshness(df)

    print("[2/5] Calcul des KPI et analyses...")
    kpis = compute_kpis(df)
    sparklines = make_sparkline_data(df)
    trend_table = compute_trend_table(df)
    forecast_milestones = compute_forecast_milestones(df)
    scenario_comparison = compute_scenario_comparison(df)

    print("[3/5] Création des figures...")
    figs = {
        "fig_timeseries": make_timeseries_figure(df, strings_en),
        "fig_stl": make_stl_figure(df, strings_en),
        "fig_seasonal_bar": make_seasonal_bar_figure(df, strings_en),
        "fig_annual_bar": make_annual_bar_figure(df, strings_en),
        "fig_gldas": make_gldas_contribution_figure(df, strings_en),
        "fig_map": make_aoi_map(strings_en),
        "fig_multi_scenario": make_multi_scenario_figure(df, strings_en),
        "fig_decision_bar": make_decision_mini_bar(df, strings_en),
        "fig_expert_scenario": make_expert_scenario_figure(df, strings_en),
    }

    # Responsive config for all Plotly charts
    plotly_cfg = {"responsive": True}
    divs = {k: pio.to_html(v, full_html=False, include_plotlyjs=False,
                            config=plotly_cfg)
            for k, v in figs.items()}

    # Date de dernière mise à jour (dernier mois observé)
    last_update_date = freshness["last_grace_month"]
    sync_status = _compute_sync_status(freshness)

    print("[4/5] Assemblage du HTML...")
    tpl_path = Path(__file__).parent / "templates" / "dashboard.html"
    html = Template(tpl_path.read_text(encoding="utf-8")).render(
        strings_en=strings_en, strings_fr=strings_fr,
        freshness=freshness, sparklines=sparklines,
        last_update_date=last_update_date,
        sync_status=sync_status,
        colors=config.COLORS,
        trend_table=trend_table,
        forecast_milestones=forecast_milestones,
        scenario_comparison=scenario_comparison,
        **divs, **kpis,
    )

    print("[5/5] Écriture du fichier...")
    out = config.DOCS_DIR / "index.html"
    config.DOCS_DIR.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard généré : {out}")


if __name__ == "__main__":
    build()
