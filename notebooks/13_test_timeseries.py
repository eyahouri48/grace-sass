# fichier : notebooks/13_test_timeseries.py 

from pipeline.build_dashboard import (
    load_data, load_strings,
    make_timeseries_figure, add_forecast_to_figure,
)

df = load_data()
strings = load_strings("en")

# Créer la figure de base (historique)
fig = make_timeseries_figure(df, strings)

# Ajouter la prévision par-dessus
fig = add_forecast_to_figure(fig, df, strings)

fig.show()
fig.write_html("test_ts_forecast.html")

print("Vérifie dans le navigateur :")
print("  ✓ La prévision prolonge la série après le dernier mois observé ?")
print("  ✓ Zone bleue (trait plein) = horizon validé ≤24 mois ?")
print("  ✓ Zone ambre (tirets) = extrapolation >24 mois ?")
print("  ✓ Bande d'IC qui s'élargit ?")
print("  ✓ Ligne verticale de coupure avec annotation ?")
print("  ✓ Légende : 4-5 entrées (Observed, Interpolated, Forecast, Scenario, CI) ?")