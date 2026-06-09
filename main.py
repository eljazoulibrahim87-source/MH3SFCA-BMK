"""
Point d'entrée principal du projet MH3SFCA-λ — Béni Mellal-Khénifra.

Usage en ligne de commande :

    python main.py                              # pipeline complet
    python main.py --mode initial               # SPAI état actuel
    python main.py --mode simulation            # initial + scénario simulé
    python main.py --mode pareto                # analyse des candidats
    python main.py --mode visualisation         # toutes les figures
    python main.py --mode tout                  # alias par défaut

Les sorties CSV (tables SPAI, comparatif, Pareto) sont écrites dans
``outputs/``. Les figures PNG aussi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permettre les imports "from src.x" même quand on lance depuis la racine
RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "src"))

DOSSIER_OUTPUTS = RACINE / "outputs"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_initial() -> None:
    """Calcule le SPAI sur l'état actuel uniquement et exporte le CSV."""
    from data_loader import charger_donnees_pfe
    from mh3sfca import construire_table_spai, executer_modele, preparer_donnees

    print("\n[1/1] Calcul SPAI — état initial\n")
    donnees = preparer_donnees(charger_donnees_pfe())
    resultats = executer_modele(donnees, verbeux=True)
    table = construire_table_spai(donnees, resultats)

    chemin = DOSSIER_OUTPUTS / "spai_initial.csv"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(chemin, index=False, encoding="utf-8")
    print(f"\n→ Table exportée : {chemin}")


def mode_simulation() -> None:
    """Lance la simulation 3 cliniques et exporte initial + simulé + comparatif."""
    from simulation import executer_simulation

    print("\n[1/1] Simulation — ajout de 3 cliniques\n")
    comparaison = executer_simulation(verbeux=True)

    DOSSIER_OUTPUTS.mkdir(parents=True, exist_ok=True)
    chemin_i = DOSSIER_OUTPUTS / "spai_initial.csv"
    chemin_s = DOSSIER_OUTPUTS / "spai_simule.csv"
    chemin_c = DOSSIER_OUTPUTS / "comparatif_initial_vs_simule.csv"
    comparaison.table_initial.to_csv(chemin_i, index=False, encoding="utf-8")
    comparaison.table_simule.to_csv(chemin_s, index=False, encoding="utf-8")
    comparaison.resume.to_csv(chemin_c, index=False, encoding="utf-8")

    print(f"\n→ Tables exportées :")
    print(f"  {chemin_i}")
    print(f"  {chemin_s}")
    print(f"  {chemin_c}")


def mode_pareto() -> None:
    """Lance l'analyse de Pareto et exporte la table des candidats."""
    from pareto import executer_analyse_pareto

    print("\n[1/1] Analyse de Pareto — sites candidats\n")
    table = executer_analyse_pareto(verbeux=True)

    DOSSIER_OUTPUTS.mkdir(parents=True, exist_ok=True)
    chemin = DOSSIER_OUTPUTS / "pareto_candidats.csv"
    table.to_csv(chemin, index=False, encoding="utf-8")
    print(f"\n→ Table exportée : {chemin}")


def mode_visualisation() -> None:
    """Génère toutes les figures (cartes, distribution, Pareto)."""
    from pareto import executer_analyse_pareto
    from simulation import executer_simulation
    from visualisation import generer_toutes_visualisations

    print("\n[1/3] Calcul des scénarios initial et simulé…")
    comparaison = executer_simulation(verbeux=False)

    print("[2/3] Analyse de Pareto…")
    table_pareto = executer_analyse_pareto(verbeux=False)

    print("[3/3] Génération des figures…\n")
    fichiers = generer_toutes_visualisations(
        table_initial=comparaison.table_initial,
        table_simule=comparaison.table_simule,
        resume_comparaison=comparaison.resume,
        table_pareto=table_pareto,
        dossier_sortie=DOSSIER_OUTPUTS,
    )

    for nom, chemin in fichiers.items():
        print(f"  ✓ {nom:30s} → {chemin.name}")


def mode_tout() -> None:
    """Pipeline complet : initial → simulation → Pareto → visualisations."""
    from pareto import executer_analyse_pareto
    from simulation import executer_simulation
    from visualisation import generer_toutes_visualisations

    DOSSIER_OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Calcul des scénarios initial et simulé\n")
    comparaison = executer_simulation(verbeux=True)
    comparaison.table_initial.to_csv(DOSSIER_OUTPUTS / "spai_initial.csv",
                                      index=False, encoding="utf-8")
    comparaison.table_simule.to_csv(DOSSIER_OUTPUTS / "spai_simule.csv",
                                     index=False, encoding="utf-8")
    comparaison.resume.to_csv(DOSSIER_OUTPUTS / "comparatif_initial_vs_simule.csv",
                               index=False, encoding="utf-8")

    print("\n[2/4] Analyse de Pareto\n")
    table_pareto = executer_analyse_pareto(verbeux=True)
    table_pareto.to_csv(DOSSIER_OUTPUTS / "pareto_candidats.csv",
                        index=False, encoding="utf-8")

    print("\n[3/4] Génération des figures\n")
    fichiers = generer_toutes_visualisations(
        table_initial=comparaison.table_initial,
        table_simule=comparaison.table_simule,
        resume_comparaison=comparaison.resume,
        table_pareto=table_pareto,
        dossier_sortie=DOSSIER_OUTPUTS,
    )
    for nom, chemin in fichiers.items():
        print(f"  ✓ {nom:30s} → {chemin.name}")

    print("\n[4/4] Récapitulatif\n")
    print(f"Sorties écrites dans : {DOSSIER_OUTPUTS}")
    print("  • CSV : spai_initial, spai_simule, comparatif_initial_vs_simule, "
          "pareto_candidats")
    print("  • PNG : 3 cartes, 1 distribution, 1 tableau de bord, 1 Pareto")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODES = {
    "initial": mode_initial,
    "simulation": mode_simulation,
    "pareto": mode_pareto,
    "visualisation": mode_visualisation,
    "tout": mode_tout,
}


def parser_arguments() -> argparse.Namespace:
    parseur = argparse.ArgumentParser(
        description="Modèle MH3SFCA-λ — accessibilité spatiale aux soins, "
                    "région Béni Mellal-Khénifra.",
    )
    parseur.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="tout",
        help="Étape à exécuter (défaut : tout).",
    )
    return parseur.parse_args()


def main() -> None:
    args = parser_arguments()
    print("=" * 78)
    print(" MH3SFCA-λ — Béni Mellal-Khénifra ".center(78, "="))
    print(f" Mode : {args.mode} ".center(78, "="))
    print("=" * 78)
    MODES[args.mode]()
    print("\nTerminé.")


if __name__ == "__main__":
    main()
