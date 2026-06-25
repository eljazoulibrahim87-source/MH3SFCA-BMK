"""
Simulation : ajout de 3 cliniques privées dans la région BMK.

Le scénario consiste à injecter trois cliniques simulées de 40 lits aux
sites du front de Pareto (Khénifra, Souk Sebt, Demnate) et à recalculer
l'indice SPAI selon la méthode MH3SFCA-λ.

Les temps de trajet vers les sites candidats sont fournis dans
``OD_site_candidat_communes.xlsx`` (matrice 135 communes × 3 sites,
indexée par le nom court de commune ; le fichier ``centroïdes_des_communes
.csv`` sert de pont vers les codes géographiques utilisés ailleurs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import LAMBDA_PRIVE
from data_loader import (
    charger_communes,
    charger_donnees_pfe,
    charger_od_sites_candidats,
)
from mh3sfca import (
    DonneesModele,
    ResultatsMH3SFCA,
    construire_table_spai,
    executer_modele,
    preparer_donnees,
)

# ---------------------------------------------------------------------------
# Définition des sites candidats
# ---------------------------------------------------------------------------

# Capacité standard utilisée par toutes les cliniques simulées (mémoire).
LITS_CLINIQUE_SIMULEE = 40

# Mapping nom de colonne dans OD_site_candidat_communes.xlsx → nom logique
# (utilisé comme nom_etablissement dans le DataFrame long).
SITES_PARETO = {
    "Khenifra": "clinique simule 1",
    "Souk Sebt Ouled Nemma": "clinique simule 2",
    "Demnate": "clinique simule 3",
}


@dataclass
class ComparaisonScenarios:
    """Comparaison entre l'état initial et le scénario simulé."""

    table_initial: pd.DataFrame
    table_simule: pd.DataFrame
    resume: pd.DataFrame   # 1 ligne par indicateur


# ---------------------------------------------------------------------------
# Construction du dataset enrichi
# ---------------------------------------------------------------------------

def construire_lignes_pour_site(
    nom_etab: str,
    capacite_lits: int,
    od_par_commune: pd.Series,
    table_communes: pd.DataFrame,
    coord_lat: float = np.nan,
    coord_lon: float = np.nan,
    type_etab: str = "clinique simulée",
    circ_sanitaire: str = "",
) -> pd.DataFrame:
    """Construit les lignes (commune × site) au format long pour UN site donné.

    Fonction générique réutilisable pour les sites Pareto (avec leur OD dans
    ``OD_site_candidat_communes.xlsx``) ou pour un site personnalisé fourni
    par l'utilisateur (avec sa propre matrice OD uploadée).

    Parameters
    ----------
    nom_etab : str
        Nom de l'établissement injecté dans la colonne ``nom_etablissement``.
    capacite_lits : int
        Nombre de lits du site simulé (capacité variable par site).
    od_par_commune : pd.Series
        Index = nom de commune (format HCP, ex. "C d'Afourar"), valeurs =
        temps de trajet en minutes.
    table_communes : pd.DataFrame
        Centroïdes : sert de pont nom → (code, population).
    coord_lat, coord_lon : float
        Coordonnées du site (pour affichage carte ; pas utilisé dans le calcul).
    type_etab : str
        Type d'établissement (détermine λ via :func:`_determiner_lambda`).
    """
    pont = table_communes.set_index("nom")[["Code_Commu", "Populati_1"]]

    lignes = []
    for nom_commune, temps in od_par_commune.items():
        if nom_commune not in pont.index or pd.isna(temps):
            continue
        info = pont.loc[nom_commune]
        lignes.append({
            "Code géographique": info["Code_Commu"],
            "Collectivités territoriales": nom_commune,
            "Population": info["Populati_1"],
            "type_etablissement": type_etab,
            "nom_etablissement": nom_etab,
            "TEMPS_TRAJET_MINUTES": float(temps),
            "DIST_KM": np.nan,
            "capacite_lits": int(capacite_lits),
            "Latitude_Etablissement": coord_lat,
            "Longitude_Etablissement": coord_lon,
            "Province_Etablissement": np.nan,
            "Region_Etablissement": "Beni Mellal-Khenifra",
            "Milieu_Etablissement": np.nan,
            "Circ_Sanitaire": circ_sanitaire or nom_etab,
        })
    return pd.DataFrame(lignes)


def _construire_lignes_simulees(
    od_candidats: pd.DataFrame,
    table_communes: pd.DataFrame,
    sites: dict[str, str],
) -> pd.DataFrame:
    """Construit les lignes (commune × site simulé) au format long de
    ``données_PFE.xlsx``.

    Parameters
    ----------
    od_candidats : pd.DataFrame
        Format wide : colonnes = ['Commune', site_1, site_2, ...].
    table_communes : pd.DataFrame
        Centroïdes : sert de pont nom → code_commune et population.
    sites : dict
        ``{nom_colonne_OD: nom_etablissement_simulé}``.
    """
    # Pont nom_commune → (code, population)
    pont = table_communes.set_index("nom")[["Code_Commu", "Populati_1"]]

    lignes = []
    for nom_colonne, nom_etab in sites.items():
        if nom_colonne not in od_candidats.columns:
            raise KeyError(f"Site candidat absent du fichier OD : {nom_colonne!r}")
        for _, ligne in od_candidats.iterrows():
            info_commune = pont.loc[ligne["Commune"]]
            lignes.append({
                "Code géographique": info_commune["Code_Commu"],
                "Collectivités territoriales": ligne["Commune"],
                "Population": info_commune["Populati_1"],
                "type_etablissement": "clinique simulée",
                "nom_etablissement": nom_etab,
                "TEMPS_TRAJET_MINUTES": ligne[nom_colonne],
                "DIST_KM": np.nan,
                "capacite_lits": LITS_CLINIQUE_SIMULEE,
                "Latitude_Etablissement": np.nan,
                "Longitude_Etablissement": np.nan,
                "Province_Etablissement": np.nan,
                "Region_Etablissement": "Beni Mellal-Khenifra",
                "Milieu_Etablissement": np.nan,
                "Circ_Sanitaire": nom_colonne,
            })
    return pd.DataFrame(lignes)


def construire_dataset_simule(
    sites: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble le dataset enrichi du scénario.

    Returns
    -------
    df_initial : pd.DataFrame
        Le format long original (référence avant simulation).
    df_simule : pd.DataFrame
        Le format long avec les sites simulés ajoutés.
    """
    if sites is None:
        sites = SITES_PARETO

    df_initial = charger_donnees_pfe()
    od_candidats = charger_od_sites_candidats()
    communes = charger_communes()

    lignes_simulees = _construire_lignes_simulees(od_candidats, communes, sites)
    df_simule = pd.concat([df_initial, lignes_simulees], ignore_index=True)
    return df_initial, df_simule


# ---------------------------------------------------------------------------
# Statistiques de comparaison
# ---------------------------------------------------------------------------

def _statistiques_spai(table: pd.DataFrame) -> dict[str, float]:
    """Statistiques agrégées d'une table SPAI (sortie de
    :func:`construire_table_spai`).
    """
    spai_pm = table["spai_pour_mille"]
    pop = table["population"]
    return {
        "spai_moyen": float(spai_pm.mean()),
        "spai_ecart_type": float(spai_pm.std(ddof=1)),
        "spai_max": float(spai_pm.max()),
        "spai_min": float(spai_pm.min()),
        "spai_median": float(spai_pm.median()),
        "spai_moyen_pondere": float((spai_pm * pop).sum() / pop.sum()),
        "deserts_medicaux": int((spai_pm == 0).sum()),
        "population_dans_deserts": int(pop[spai_pm == 0].sum()),
    }


def comparer_scenarios(
    table_initial: pd.DataFrame,
    table_simule: pd.DataFrame,
) -> pd.DataFrame:
    """Construit la table de comparaison initial vs simulé."""
    stats_i = _statistiques_spai(table_initial)
    stats_s = _statistiques_spai(table_simule)

    libelles = {
        "spai_moyen": "SPAI moyen (lits/1000 hab.)",
        "spai_ecart_type": "SPAI écart-type",
        "spai_max": "SPAI max",
        "spai_min": "SPAI min",
        "spai_median": "SPAI médian",
        "spai_moyen_pondere": "SPAI moyen pondéré par population",
        "deserts_medicaux": "Déserts médicaux absolus (SPAI = 0)",
        "population_dans_deserts": "Population dans les déserts",
    }

    lignes = []
    for cle, libelle in libelles.items():
        v_i = stats_i[cle]
        v_s = stats_s[cle]
        if v_i != 0:
            variation = (v_s - v_i) / v_i * 100
        else:
            variation = float("nan")
        lignes.append({
            "indicateur": libelle,
            "initial": v_i,
            "simulé": v_s,
            "variation_pct": variation,
        })
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def executer_simulation(
    sites: dict[str, str] | None = None,
    verbeux: bool = True,
) -> ComparaisonScenarios:
    """Pipeline complet : initial + simulé, puis comparaison."""
    df_initial, df_simule = construire_dataset_simule(sites)

    if verbeux:
        print("Scénario initial (état actuel)")
        print("-" * 60)
    donnees_i = preparer_donnees(df_initial)
    resultats_i = executer_modele(donnees_i, verbeux=verbeux)
    table_i = construire_table_spai(donnees_i, resultats_i)

    if verbeux:
        print("Scénario simulé (3 cliniques privées de 40 lits ajoutées)")
        print("-" * 60)
    donnees_s = preparer_donnees(df_simule)
    resultats_s = executer_modele(donnees_s, verbeux=verbeux)
    table_s = construire_table_spai(donnees_s, resultats_s)

    resume = comparer_scenarios(table_i, table_s)
    return ComparaisonScenarios(
        table_initial=table_i,
        table_simule=table_s,
        resume=resume,
    )


# ---------------------------------------------------------------------------
# Analyse complémentaire — communes qui sortent du désert médical
# ---------------------------------------------------------------------------

def identifier_communes_sorties_du_desert(
    table_initial: pd.DataFrame,
    table_simule: pd.DataFrame,
) -> pd.DataFrame:
    """Renvoie les communes dont SPAI passe de 0 à > 0 après simulation."""
    fusion = table_initial[["code", "nom", "population", "spai_pour_mille"]].merge(
        table_simule[["code", "spai_pour_mille"]],
        on="code",
        suffixes=("_initial", "_simule"),
    )
    sorties = fusion[
        (fusion["spai_pour_mille_initial"] == 0)
        & (fusion["spai_pour_mille_simule"] > 0)
    ].sort_values("spai_pour_mille_simule", ascending=False)
    return sorties.reset_index(drop=True)


def identifier_plus_grandes_progressions(
    table_initial: pd.DataFrame,
    table_simule: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Renvoie les n communes ayant le plus gros gain absolu de SPAI."""
    fusion = table_initial[["code", "nom", "population", "spai_pour_mille"]].merge(
        table_simule[["code", "spai_pour_mille"]],
        on="code",
        suffixes=("_initial", "_simule"),
    )
    fusion["gain_absolu"] = fusion["spai_pour_mille_simule"] - fusion["spai_pour_mille_initial"]
    return fusion.sort_values("gain_absolu", ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print(" Simulation MH3SFCA-λ : ajout de 3 cliniques (Pareto) ".center(78, "="))
    print("=" * 78)
    print(f"Sites retenus : {', '.join(SITES_PARETO.keys())}")
    print(f"Capacité      : {LITS_CLINIQUE_SIMULEE} lits / clinique, λ = {LAMBDA_PRIVE}")
    print()

    comparaison = executer_simulation(verbeux=True)

    print()
    print("=" * 78)
    print(" Tableau comparatif — État initial vs Scénario simulé ".center(78, "="))
    print("=" * 78)
    with pd.option_context("display.max_columns", None, "display.width", 110,
                            "display.float_format", lambda x: f"{x:>10.4f}"):
        print(comparaison.resume.to_string(index=False))

    print()
    print("Communes sorties du désert médical (SPAI 0 → > 0) :")
    sorties = identifier_communes_sorties_du_desert(
        comparaison.table_initial, comparaison.table_simule
    )
    if sorties.empty:
        print("  Aucune commune sortie du désert.")
    else:
        print(sorties[["code", "nom", "population", "spai_pour_mille_simule"]]
              .rename(columns={"spai_pour_mille_simule": "SPAI×1000 simulé"})
              .to_string(index=False))

    print()
    print("Top 5 progressions absolues de SPAI :")
    print(
        identifier_plus_grandes_progressions(
            comparaison.table_initial, comparaison.table_simule, n=5
        )[["code", "nom", "spai_pour_mille_initial",
           "spai_pour_mille_simule", "gain_absolu"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
