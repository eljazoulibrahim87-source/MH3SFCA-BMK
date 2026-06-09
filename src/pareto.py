"""
Analyse de Pareto bi-objectif — sélection des meilleurs sites candidats.

Pour chaque site candidat (commune où l'on simule l'ajout d'une nouvelle
clinique privée de 40 lits), deux objectifs sont évalués simultanément :

    X = Demande solvable drainée (habitants AMO-couverts captés par le
        nouveau site)
          X = Σᵢ Huffᵢ,new · Pᵢ · TAUX_AMO

    Y = Gain d'accessibilité moyen pondéré par la population
          Y = (SPAI_moyen_simulé − SPAI_moyen_initial) × 10⁶
              avec moyennes pondérées par Pᵢ.

Un candidat est dit *Pareto-optimal* si aucun autre candidat ne le domine
sur les deux dimensions simultanément. Le front de Pareto regroupe les
candidats non dominés ; il constitue l'ensemble des choix défendables
pour l'aménagement.

Méthodologie cohérente avec ``Pareto_MH3SFCA.xlsx`` (mémoire).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config import LAMBDA_PRIVE, TAUX_AMO
from data_loader import (
    charger_communes,
    charger_donnees_pfe,
    charger_od_sites_candidats,
)
from mh3sfca import (
    DonneesModele,
    calculer_friction,
    calculer_huff,
    calculer_ratio_offre_demande,
    calculer_spai,
    calculer_utilite,
    preparer_donnees,
)
from simulation import LITS_CLINIQUE_SIMULEE, _construire_lignes_simulees


# ---------------------------------------------------------------------------
# Évaluation d'un candidat individuel
# ---------------------------------------------------------------------------

@dataclass
class ScoreCandidat:
    """Coordonnées (X, Y) d'un candidat dans le plan de Pareto."""

    nom_site: str                    # nom de la colonne dans OD candidats
    nom_etablissement: str           # nom logique injecté dans les données
    population_commune: int          # population de la commune hôte
    demande_drainee_amo: float       # X
    delta_spai_pondere_x1e6: float   # Y
    spai_moyen_simule: float         # info complémentaire (×1000)
    deserts_simules: int             # info complémentaire
    pareto_optimal: bool = False     # rempli par identifier_front_pareto


def _spai_moyen_pondere_par_population(
    spai: np.ndarray, population: np.ndarray
) -> float:
    """Moyenne du SPAI pondérée par la population (lits/habitant)."""
    return float((spai * population).sum() / population.sum())


def _resultats_complets(donnees: DonneesModele):
    """Recalcule toutes les sorties intermédiaires (helper interne)."""
    lits = donnees.etablissements["lits"].to_numpy(dtype=float)
    lambdas = donnees.etablissements["lambda"].to_numpy(dtype=float)
    population = donnees.communes["population"].to_numpy(dtype=float)

    friction = calculer_friction(donnees.matrice_temps)
    utilite = calculer_utilite(friction, lits, lambdas)
    huff = calculer_huff(utilite)
    ratio = calculer_ratio_offre_demande(huff, population, lits, lambdas)
    spai = calculer_spai(huff, ratio, friction)
    return friction, huff, spai, population


def evaluer_candidat(
    df_initial: pd.DataFrame,
    table_communes: pd.DataFrame,
    nom_colonne_od: str,
    od_candidats: pd.DataFrame,
    nom_etablissement: str | None = None,
) -> ScoreCandidat:
    """Évalue un site candidat seul (40 lits) sur les deux axes (X, Y).

    Parameters
    ----------
    df_initial : pd.DataFrame
        Données long de l'état actuel (sortie de ``charger_donnees_pfe``).
    table_communes : pd.DataFrame
        Centroïdes des communes (pont nom → code, population).
    nom_colonne_od : str
        Nom de la colonne dans ``od_candidats`` représentant le site.
    od_candidats : pd.DataFrame
        Matrice OD des sites candidats.
    nom_etablissement : str, optional
        Nom logique pour le nouvel établissement. Par défaut, dérivé de
        ``nom_colonne_od``.
    """
    if nom_etablissement is None:
        nom_etablissement = f"candidat — {nom_colonne_od}"

    # 1) État initial — uniquement les communes/population/SPAI moyen
    donnees_i = preparer_donnees(df_initial)
    _, _, spai_i, pop_i = _resultats_complets(donnees_i)
    spai_moyen_i = _spai_moyen_pondere_par_population(spai_i, pop_i)

    # 2) État simulé — on injecte UN seul candidat
    lignes = _construire_lignes_simulees(
        od_candidats=od_candidats,
        table_communes=table_communes,
        sites={nom_colonne_od: nom_etablissement},
    )
    df_simule = pd.concat([df_initial, lignes], ignore_index=True)
    donnees_s = preparer_donnees(df_simule)
    _, huff_s, spai_s, pop_s = _resultats_complets(donnees_s)
    spai_moyen_s = _spai_moyen_pondere_par_population(spai_s, pop_s)

    # 3) X — demande drainée AMO par le nouveau site
    idx_new = donnees_s.etablissements.index[
        donnees_s.etablissements["nom"] == nom_etablissement
    ][0]
    huff_new = huff_s[:, idx_new]
    demande_drainee = float((huff_new * pop_s).sum() * TAUX_AMO)

    # 4) Y — ΔSPAI moyen pondéré par la population × 10⁶
    delta_y = (spai_moyen_s - spai_moyen_i) * 1e6

    # 5) Métadonnées complémentaires
    info_commune = (
        table_communes.set_index("nom")
        .loc[nom_colonne_od, "Populati_1"]
        if nom_colonne_od in table_communes["nom"].values
        else np.nan
    )

    return ScoreCandidat(
        nom_site=nom_colonne_od,
        nom_etablissement=nom_etablissement,
        population_commune=int(info_commune) if not pd.isna(info_commune) else 0,
        demande_drainee_amo=demande_drainee,
        delta_spai_pondere_x1e6=delta_y,
        spai_moyen_simule=spai_moyen_s * 1000.0,
        deserts_simules=int((spai_s == 0).sum()),
    )


# ---------------------------------------------------------------------------
# Identification du front de Pareto (deux objectifs à maximiser)
# ---------------------------------------------------------------------------

def identifier_front_pareto(scores: list[ScoreCandidat]) -> list[ScoreCandidat]:
    """Marque les candidats Pareto-optimaux (maximisation de X et Y).

    Un candidat (Xᵢ, Yᵢ) est dominé s'il existe un autre candidat (Xⱼ, Yⱼ)
    avec Xⱼ ≥ Xᵢ ET Yⱼ ≥ Yᵢ, avec au moins une inégalité stricte.
    """
    xs = np.array([s.demande_drainee_amo for s in scores])
    ys = np.array([s.delta_spai_pondere_x1e6 for s in scores])

    for i, score in enumerate(scores):
        domine = (
            (xs >= xs[i]) & (ys >= ys[i])
            & ((xs > xs[i]) | (ys > ys[i]))
        )
        score.pareto_optimal = not domine.any()
    return scores


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def executer_analyse_pareto(
    sites_a_tester: Iterable[str] | None = None,
    verbeux: bool = True,
) -> pd.DataFrame:
    """Évalue tous les candidats et identifie le front de Pareto.

    Parameters
    ----------
    sites_a_tester : iterable de str, optional
        Liste des noms de colonnes du fichier OD candidats à évaluer.
        Par défaut, tous les sites pour lesquels une OD est disponible.

    Returns
    -------
    pd.DataFrame
        Table avec, pour chaque candidat : nom, X, Y, statut Pareto, etc.
    """
    df_initial = charger_donnees_pfe()
    od_candidats = charger_od_sites_candidats()
    communes = charger_communes()

    if sites_a_tester is None:
        sites_a_tester = [c for c in od_candidats.columns if c != "Commune"]

    if verbeux:
        print(f"Candidats évalués ({len(list(sites_a_tester))}) : "
              f"{', '.join(sites_a_tester)}")
        print(f"Capacité injectée : {LITS_CLINIQUE_SIMULEE} lits "
              f"(λ = {LAMBDA_PRIVE}, type = clinique privée)")
        print(f"Taux AMO utilisé  : {TAUX_AMO}")
        print()

    scores: list[ScoreCandidat] = []
    for site in sites_a_tester:
        if verbeux:
            print(f"  • Évaluation de {site}…", end=" ", flush=True)
        score = evaluer_candidat(
            df_initial=df_initial,
            table_communes=communes,
            nom_colonne_od=site,
            od_candidats=od_candidats,
        )
        scores.append(score)
        if verbeux:
            print(f"X={score.demande_drainee_amo:>10.1f}, "
                  f"Y={score.delta_spai_pondere_x1e6:>7.3f}")

    identifier_front_pareto(scores)

    table = pd.DataFrame([
        {
            "site": s.nom_site,
            "population_commune": s.population_commune,
            "X_demande_drainee_amo": s.demande_drainee_amo,
            "Y_delta_spai_x1e6": s.delta_spai_pondere_x1e6,
            "spai_moyen_simulé": s.spai_moyen_simule,
            "déserts_simulés": s.deserts_simules,
            "statut": "Pareto" if s.pareto_optimal else "Dominé",
        }
        for s in scores
    ])
    return table.sort_values("Y_delta_spai_x1e6", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print(" Analyse de Pareto — sélection de sites candidats ".center(78, "="))
    print("=" * 78)
    print()

    table = executer_analyse_pareto(verbeux=True)

    print()
    print("=" * 78)
    print(" Résultats — coordonnées (X, Y) et statut Pareto ".center(78, "="))
    print("=" * 78)
    with pd.option_context("display.width", 120, "display.max_columns", None,
                            "display.float_format", lambda x: f"{x:>10.3f}"):
        print(table.to_string(index=False))

    print()
    front = table[table["statut"] == "Pareto"]
    print(f"Front de Pareto : {len(front)} site(s) non dominé(s)")
    for _, ligne in front.iterrows():
        print(f"  ✓ {ligne['site']:30s}  X={ligne['X_demande_drainee_amo']:>10.1f}  "
              f"Y={ligne['Y_delta_spai_x1e6']:>6.3f}")


if __name__ == "__main__":
    main()
