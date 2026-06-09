"""
Chargement des données brutes du projet MH3SFCA-λ — Béni Mellal-Khénifra.

Ce module centralise la lecture de tous les fichiers du dossier ``data/`` et
fournit un résumé descriptif de chaque jeu de données (dimensions, colonnes,
aperçu des premières lignes).

Conventions :
- Encodage CSV : ``utf-8`` pour les centroïdes, ``cp1252`` pour le fichier
  des établissements (export Windows historique).
- Séparateur CSV : virgule pour les centroïdes, point-virgule pour les
  établissements.
- Les classeurs Excel comportant plusieurs feuilles sont chargés sous forme
  d'un dictionnaire ``{nom_feuille: DataFrame}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Racine du projet : src/ se trouve à côté de data/
RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE_PROJET / "data"

# Noms canoniques (clé logique) → nom de fichier réel dans data/
FICHIERS = {
    "communes": "centroïdes_des_communes.csv",
    "etablissements": "EMPLACEMENT_ETAB_REEL_ET_SIMULE.csv",
    "od_sites_candidats": "OD_site_candidat_communes.xlsx",
    "donnees_pfe": "données_PFE.xlsx",
    "calcul_mh3sfca": "clacul_des_étapes_de_la_MH3SFCA.xlsx",
    "simulation_3cliniques": "MH3SFCA_simulation_3cliniques.xlsx",
    "comparatif_spai": "comparatif_SPAI_simulé_initial.xlsx",
}

# Dossier des géométries (shapefile communes Maroc)
DOSSIER_SHAPEFILE = DOSSIER_DATA / "communes_shapefile"
FICHIER_SHAPEFILE = DOSSIER_SHAPEFILE / "populaion_commune.shp"

# Type renvoyé par les chargeurs Excel multi-feuilles
DictDeDataFrames = Dict[str, pd.DataFrame]
JeuDeDonnees = Union[pd.DataFrame, DictDeDataFrames]


# ---------------------------------------------------------------------------
# Chargeurs individuels
# ---------------------------------------------------------------------------

def charger_communes() -> pd.DataFrame:
    """Charge les 135 centroïdes communaux avec population.

    Colonnes attendues : ``Code_Commu``, ``nom``, ``longitude``, ``latitude``,
    ``Populati_1`` (population totale RGPH 2024), etc.
    """
    chemin = DOSSIER_DATA / FICHIERS["communes"]
    return pd.read_csv(chemin, sep=",", encoding="utf-8")


def charger_etablissements() -> pd.DataFrame:
    """Charge les 35 établissements réels + 3 simulés.

    Fichier exporté depuis Excel Windows : encodage ``cp1252`` et séparateur
    point-virgule. Une colonne vide (artefact d'export) est éliminée.
    """
    chemin = DOSSIER_DATA / FICHIERS["etablissements"]
    df = pd.read_csv(chemin, sep=";", encoding="cp1252")
    # Suppression des colonnes vides éventuelles (export Excel)
    df = df.dropna(axis=1, how="all")
    return df


def charger_od_sites_candidats() -> pd.DataFrame:
    """Charge la matrice origine-destination communes → sites candidats.

    Les temps de trajet sont exprimés en minutes (réseau routier OSM via
    QGIS). Une seule feuille ``Feuil1``.
    """
    chemin = DOSSIER_DATA / FICHIERS["od_sites_candidats"]
    return pd.read_excel(chemin, sheet_name="Feuil1", engine="openpyxl")


def charger_donnees_pfe() -> pd.DataFrame:
    """Charge le classeur synthétique ``données_PFE.xlsx``.

    Contient une unique feuille ``données pour PFE`` rassemblant les données
    principales utilisées par le mémoire.
    """
    chemin = DOSSIER_DATA / FICHIERS["donnees_pfe"]
    return pd.read_excel(chemin, sheet_name=0, engine="openpyxl")


def charger_calcul_mh3sfca() -> DictDeDataFrames:
    """Charge le classeur de référence des 3 étapes de la méthode.

    Renvoie un dictionnaire ``{nom_feuille: DataFrame}`` couvrant :
    ``GUIDE_UTILISATION``, ``Param``, ``Demande``, ``Offre``, ``Matrice_OD``,
    ``1_Friction``, ``2_Utilité``, ``3_Huff``, ``4_SPAI``.
    """
    chemin = DOSSIER_DATA / FICHIERS["calcul_mh3sfca"]
    return pd.read_excel(chemin, sheet_name=None, engine="openpyxl")


def charger_simulation_3cliniques() -> DictDeDataFrames:
    """Charge les résultats de simulation avec 3 cliniques ajoutées.

    Même structure que le classeur de calcul, mais avec l'offre augmentée
    des sites Khénifra, Souk Sebt et Demnate (front de Pareto retenu).
    """
    chemin = DOSSIER_DATA / FICHIERS["simulation_3cliniques"]
    return pd.read_excel(chemin, sheet_name=None, engine="openpyxl")


def charger_comparatif_spai() -> DictDeDataFrames:
    """Charge la comparaison SPAI initial vs SPAI simulé par commune."""
    chemin = DOSSIER_DATA / FICHIERS["comparatif_spai"]
    return pd.read_excel(chemin, sheet_name=None, engine="openpyxl")


def charger_geometries_communes():
    """Charge les polygones des 135 communes de Béni Mellal-Khénifra.

    Le shapefile contient toutes les communes du Maroc ; on filtre sur
    la région BMK via les Code_Commu présents dans
    ``centroïdes_des_communes.csv``.

    Returns
    -------
    geopandas.GeoDataFrame
        Geo-dataframe en EPSG:4326, avec colonnes ``Code_Commu``, ``nom``,
        ``Populati_1``, ``geometry``.
    """
    # Import différé : geopandas n'est requis que pour la visualisation
    import geopandas as gpd

    codes_bmk = set(charger_communes()["Code_Commu"].astype(str))
    gdf = gpd.read_file(FICHIER_SHAPEFILE)
    gdf = gdf[gdf["Code_Commu"].astype(str).isin(codes_bmk)].reset_index(drop=True)
    return gdf


# ---------------------------------------------------------------------------
# Chargement global
# ---------------------------------------------------------------------------

def charger_tout() -> Dict[str, JeuDeDonnees]:
    """Charge l'ensemble des jeux de données et les renvoie dans un dict.

    Les clés correspondent à celles définies dans :data:`FICHIERS`.
    """
    return {
        "communes": charger_communes(),
        "etablissements": charger_etablissements(),
        "od_sites_candidats": charger_od_sites_candidats(),
        "donnees_pfe": charger_donnees_pfe(),
        "calcul_mh3sfca": charger_calcul_mh3sfca(),
        "simulation_3cliniques": charger_simulation_3cliniques(),
        "comparatif_spai": charger_comparatif_spai(),
    }


# ---------------------------------------------------------------------------
# Affichage des résumés
# ---------------------------------------------------------------------------

LARGEUR_SEPARATEUR = 78


def _afficher_resume_dataframe(nom: str, df: pd.DataFrame, n_apercu: int = 3) -> None:
    """Affiche un résumé compact d'un DataFrame unique."""
    print(f"  • {nom}")
    print(f"      dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    colonnes = list(df.columns)
    apercu_colonnes = ", ".join(map(str, colonnes[:8]))
    if len(colonnes) > 8:
        apercu_colonnes += f", … (+{len(colonnes) - 8})"
    print(f"      colonnes    : {apercu_colonnes}")
    with pd.option_context(
        "display.max_columns", 8,
        "display.width", LARGEUR_SEPARATEUR,
        "display.max_colwidth", 20,
    ):
        apercu = df.head(n_apercu).to_string(index=False)
    apercu_indente = "\n".join("        " + ligne for ligne in apercu.splitlines())
    print("      aperçu :")
    print(apercu_indente)


def afficher_resume(nom_logique: str, jeu: JeuDeDonnees) -> None:
    """Affiche un résumé descriptif d'un jeu de données.

    Le jeu peut être un :class:`DataFrame` ou un dictionnaire de DataFrames
    (cas des classeurs Excel multi-feuilles).
    """
    print("─" * LARGEUR_SEPARATEUR)
    fichier_source = FICHIERS.get(nom_logique, "?")
    print(f"[{nom_logique}]  source : {fichier_source}")
    print("─" * LARGEUR_SEPARATEUR)

    if isinstance(jeu, pd.DataFrame):
        _afficher_resume_dataframe(nom_logique, jeu)
    else:
        print(f"  Classeur multi-feuilles ({len(jeu)} feuille(s))")
        for nom_feuille, df in jeu.items():
            print()
            _afficher_resume_dataframe(nom_feuille, df)
    print()


def afficher_resume_global(donnees: Dict[str, JeuDeDonnees]) -> None:
    """Affiche le résumé de tous les jeux de données chargés."""
    print()
    print("=" * LARGEUR_SEPARATEUR)
    print(" RÉSUMÉ DES DONNÉES — Projet MH3SFCA-λ Béni Mellal-Khénifra ".center(
        LARGEUR_SEPARATEUR, "="))
    print("=" * LARGEUR_SEPARATEUR)
    print(f"Dossier source : {DOSSIER_DATA}")
    print(f"Jeux chargés   : {len(donnees)}")
    print()

    for nom_logique, jeu in donnees.items():
        afficher_resume(nom_logique, jeu)

    print("=" * LARGEUR_SEPARATEUR)
    print(" Chargement terminé avec succès. ".center(LARGEUR_SEPARATEUR, "="))
    print("=" * LARGEUR_SEPARATEUR)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    """Charge tous les jeux de données et affiche un résumé descriptif."""
    print("Chargement des jeux de données depuis :", DOSSIER_DATA)
    donnees = {}
    for nom_logique, nom_fichier in FICHIERS.items():
        chemin = DOSSIER_DATA / nom_fichier
        if not chemin.exists():
            print(f"  ⚠  Fichier manquant : {nom_fichier}")
            continue
        print(f"  → {nom_logique:24s}  ({nom_fichier})")

    donnees = charger_tout()
    afficher_resume_global(donnees)


if __name__ == "__main__":
    main()
