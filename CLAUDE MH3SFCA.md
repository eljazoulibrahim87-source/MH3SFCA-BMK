# Projet MH3SFCA-λ — Accessibilité spatiale aux soins — Béni Mellal-Khénifra

## Contexte du projet
Ce projet est le code applicatif du mémoire de fin d'études (PFE) de EL JAZOULI Brahim,
étudiant à la Faculté des Sciences Économiques et de Gestion de Béni Mellal.

**Objectif** : Modéliser l'accessibilité spatiale aux soins de santé dans la région de
Béni Mellal-Khénifra en utilisant la méthode MH3SFCA-λ (Modified Huff Three-Step
Floating Catchment Area avec coefficient de préférence sectorielle λ).

---

## La méthode MH3SFCA-λ

### Principe général
La méthode MH3SFCA-λ mesure pour chaque commune un indice SPAI (Spatial Accessibility
Index, en lits disponibles pour 1 000 habitants) en trois étapes séquentielles.

### Étape 1 — Calcul du poids de Huff (Huffᵢⱼ)
Pour chaque paire (commune i, établissement j) :

    Huffᵢⱼ = (λⱼ · Sⱼ^α · exp(-dᵢⱼ² / β))
              / Σₖ (λₖ · Sₖ^α · exp(-dᵢₖ² / β))

    où la somme porte sur tous les établissements k accessibles depuis i (dᵢₖ ≤ dₘₐₓ)

Propriété clé : Σⱼ Huffᵢⱼ = 1  (conservation de la demande totale)

### Étape 2 — Calcul du ratio offre/demande pondéré (Rⱼ)
Pour chaque établissement j :

    Rⱼ = Sⱼ / Σᵢ (Huffᵢⱼ · Pᵢ)

    où la somme porte sur toutes les communes i accessibles depuis j (dᵢⱼ ≤ dₘₐₓ)

### Étape 3 — Calcul de l'indice SPAI (SPAIᵢ)
Pour chaque commune i :

    SPAIᵢ = Σⱼ Huffᵢⱼ · Rⱼ · exp(-dᵢⱼ² / β)

    où la somme porte sur tous les établissements j accessibles depuis i (dᵢⱼ ≤ dₘₐₓ)

---

## Paramètres du modèle (NE PAS MODIFIER sans justification)

| Paramètre         | Valeur   | Justification                                        |
|-------------------|----------|------------------------------------------------------|
| dₘₐₓ             | 90 min   | Médiane des temps de trajet région BMK               |
| β                 | 1758.89  | Calibré depuis dₘₐₓ selon Jörg et al. (2019, p.29)  |
| α                 | 1        | Recommandation Huff (1963) — attractivité linéaire   |
| λ_privé (clinique)| 0.9      | 90 % des dépenses AMO vers le secteur privé          |
| λ_public (hôpital)| 0.1      | 10 % des dépenses AMO vers le secteur public         |
| Capacité Sⱼ       | Nb lits  | Standard FCA (Wang, 2012)                            |

---

## Données disponibles dans ce dossier

| Fichier                                    | Contenu                                                         |
|--------------------------------------------|-----------------------------------------------------------------|
| `centroïdes_des_communes.csv`              | 135 communes : Code_Commu, nom, longitude, latitude, population |
| `EMPLACEMENT_ETAB_REEL_ET_SIMULE.csv`      | 35 établissements réels + 3 simulés : coordonnées, lits, type  |
| `OD_site_candidat_communes.xlsx`           | Matrice OD communes → sites candidats (temps en minutes)        |
| `données_PFE.xlsx`                         | Données principales : communes, établissements, matrice OD      |
| `clacul_des_étapes_de_la_MH3SFCA.xlsx`     | Calcul détaillé des 3 étapes (référence pour validation)        |
| `MH3SFCA_simulation_3cliniques.xlsx`       | Résultats de la simulation avec les 3 nouvelles cliniques       |
| `comparatif_SPAI_simulé_initial.xlsx`      | Comparaison SPAI initial vs SPAI simulé par commune             |

**Encodage** : certains CSV sont en cp1252 (Windows) — toujours spécifier l'encoding à la lecture.

---

## Résultats de référence (état initial — à reproduire exactement)

- SPAI moyen régional : **0,137** lits / 1 000 hab
- Écart-type : **0,171**
- SPAI minimum : **0,000**  |  SPAI maximum : **0,707** (Khouribga)
- Communes déserts médicaux absolus (SPAI = 0) : **21** communes (15,5 %)
- Communes très faible accessibilité (0 < SPAI ≤ 0,076) : **48** communes

## Résultats de la simulation (3 cliniques de 40 lits ajoutées)

- Sites retenus (front de Pareto) : **Khénifra, Souk Sebt, Demnate**
- Amélioration du SPAI moyen régional : **+8,03 %**
- Réduction des déserts médicaux absolus : **−38 %** (21 → ~13 communes)

---

## Architecture du code à produire

```
mh3sfca_bmk/
│
├── data/                        # Dossier des données brutes (ne pas modifier)
│
├── src/
│   ├── data_loader.py           # Lecture et nettoyage des fichiers CSV/Excel
│   ├── mh3sfca.py               # Implémentation des 3 étapes du modèle
│   ├── pareto.py                # Optimisation multicritère par front de Pareto
│   ├── simulation.py            # Scénario avant/après ajout de cliniques
│   └── visualisation.py         # Cartes choroplèthes et graphiques
│
├── outputs/                     # Résultats générés (cartes, tableaux, CSV)
│
├── main.py                      # Point d'entrée principal
├── requirements.txt
└── CLAUDE.md                    # Ce fichier
```

---

## Stack technique préférée

- **Python 3.11+**
- `pandas`, `numpy` — manipulation des données
- `geopandas`, `shapely` — données spatiales
- `matplotlib`, `seaborn` — graphiques statistiques
- `folium` ou `plotly` — cartes interactives (optionnel)
- `openpyxl` — lecture des fichiers Excel

---

## Conventions de code

- Commentaires et docstrings **en français**
- Variables nommées en minuscules avec underscores (`spai_initial`, `matrice_od`)
- Les paramètres du modèle sont des **constantes en majuscules** dans un fichier `config.py`
- Chaque étape du calcul doit afficher sa progression et un résumé des résultats
- Toujours valider les résultats contre les valeurs de référence ci-dessus

---

## Commandes utiles

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le calcul complet
python main.py

# Lancer uniquement la simulation
python main.py --mode simulation

# Lancer uniquement la visualisation
python main.py --mode visualisation
```

---

## Contexte académique

- **Méthode** : MH3SFCA-λ (Jörg et al., 2019 ; Subal et al., 2021)
- **Terrain** : Région Béni Mellal-Khénifra, Maroc (135 communes, 6 provinces)
- **Sources** : RGPH 2024 (HCP), Carte Sanitaire 2025 (MSPS), OSM + QGIS (matrice OD)
- **Problématique** : Décalage entre couverture AMO (86 %) et accessibilité géographique réelle
