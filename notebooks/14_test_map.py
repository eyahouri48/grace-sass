# fichier : notebooks/13_test_map.py
# Pour vérifier la carte AOI 

from pipeline.build_dashboard import load_strings, make_aoi_map

strings = load_strings("en")
fig = make_aoi_map(strings)

fig.show()
fig.write_html("test_map.html")

print("Vérifie dans le navigateur :")
print("  ✓ Le contour du SASS est visible (polygone bleu) ?")
print("  ✓ La grille de mascons 3° est en pointillés gris ?")
print("  ✓ On voit les frontières des pays (Algérie, Tunisie, Libye) ?")
print("  ✓ Le fond de carte est sobre (beige/blanc) ?")
print("  ✓ Pas de logo Mapbox ni de chargement externe ?")