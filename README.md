# MH3SFCA-λ — Accessibilité spatiale aux soins

Application interactive d'analyse de l'accessibilité spatiale aux établissements
de santé dans la région **Béni Mellal-Khénifra** (Maroc), basée sur la méthode
**MH3SFCA-λ** (Modified Huff 3-Step Floating Catchment Area avec coefficient de
préférence sectorielle λ).

Développée dans le cadre du Projet de Fin d'Études (PFE) de
**EL JAZOULI Brahim**, Faculté des Sciences Économiques et de Gestion,
Béni Mellal.

## Aperçu

L'application permet :
- 🗺️ Carte interactive des 135 communes avec choix du fond (polygones / centroïdes)
- 📊 Tableaux des communes, des 38 établissements de santé et des résultats SPAI
- 📈 Tableau de bord : SPAI moyen, déserts médicaux, coefficient de Gini, courbe de Lorenz
- 🧪 Simulation : ajout dynamique de 1 à 3 cliniques privées (sites Pareto)
- ⚙️ Modification interactive des paramètres du modèle (dmax, α, β, λ)

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre dans le navigateur à `http://localhost:8501`.

## Architecture

```
.
├── data/                  # Données brutes (CSV, XLSX, shapefile)
├── src/
│   ├── config.py          # Paramètres du modèle (dmax, β, α, λ)
│   ├── data_loader.py     # Chargement des fichiers
│   ├── mh3sfca.py         # Modèle (3 étapes vectorisées)
│   ├── simulation.py      # Scénario avant/après
│   ├── pareto.py          # Optimisation multicritère
│   ├── analytics.py       # Indicateurs (Gini, Lorenz, classes)
│   └── visualisation.py   # Cartes et graphiques (export PNG)
├── app.py                 # Application Streamlit (5 onglets)
├── main.py                # Pipeline CLI (alternative en ligne de commande)
└── requirements.txt
```

## Méthode MH3SFCA-λ

Pour chaque commune $i$ :

1. **Poids de Huff** :
   $\text{Huff}_{ij} = \dfrac{\lambda_j \cdot S_j^{\alpha} \cdot e^{-d_{ij}^2/\beta}}
   {\sum_k \lambda_k \cdot S_k^{\alpha} \cdot e^{-d_{ik}^2/\beta}}$

2. **Ratio offre/demande** :
   $R_j = \dfrac{\lambda_j \cdot S_j}{\sum_i \text{Huff}_{ij} \cdot P_i}$

3. **Indice SPAI** :
   $\text{SPAI}_i = \sum_j \text{Huff}_{ij} \cdot R_j \cdot e^{-d_{ij}^2/\beta}$

## Sources des données

- **Démographie** : RGPH 2024 (Haut-Commissariat au Plan)
- **Établissements** : Carte Sanitaire 2025 (Ministère de la Santé et de la Protection Sociale)
- **Matrice OD** : OpenStreetMap + plugin OD QGIS

## Références

- Jörg et al. (2019), *A modified two-step floating catchment area method…*
- Subal et al. (2021), *MH3SFCA: accounting for sectoral preferences…*
- Huff (1963), *A probabilistic analysis of shopping center trade areas*
