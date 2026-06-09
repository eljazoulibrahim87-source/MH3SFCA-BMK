"""
Visualisation des résultats MH3SFCA-λ — cartes et graphiques.

Quatre familles de sorties produites dans ``outputs/`` :

1. **Cartes choroplèthes** : SPAI initial, SPAI simulé, et carte des
   gains absolus de SPAI (commune par commune).
2. **Diagramme de Pareto** : positionnement (X, Y) des sites candidats
   et front de Pareto en surbrillance.
3. **Distribution** : histogrammes superposés SPAI initial vs simulé.
4. **Tableau comparatif** : barres pour les indicateurs synthétiques.

Toutes les figures utilisent matplotlib et sont sauvegardées en PNG
haute résolution (150 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm


# ---------------------------------------------------------------------------
# Constantes esthétiques
# ---------------------------------------------------------------------------

# Coordonnées (lon, lat) des 3 cliniques simulées du scénario Pareto.
SITES_SIMULES_COORDS = {
    "Khénifra":  (-5.666568, 32.938293),
    "Souk Sebt": (-6.702381, 32.293866),
    "Demnate":   (-7.003886, 31.733306),
}

# Bornes pour la classification choroplèthe du SPAI (en lits/1000 hab),
# alignées sur les seuils du mémoire (déserts, très faible, faible, etc.).
BORNES_SPAI = [0.0, 1e-9, 0.076, 0.137, 0.250, 0.500, 1.000]
ETIQUETTES_SPAI = [
    "Désert (SPAI = 0)",
    "Très faible (≤ 0.076)",
    "Faible (0.076 – 0.137)",
    "Moyen (0.137 – 0.250)",
    "Bon (0.250 – 0.500)",
    "Très bon (> 0.500)",
]
COULEURS_SPAI = ["#404040", "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]

DPI = 150


# ---------------------------------------------------------------------------
# Jointure géométries × résultats
# ---------------------------------------------------------------------------

def _joindre_spai_et_geometries(table_spai: pd.DataFrame):
    """Joint la table SPAI (codes sans point final) au GeoDataFrame des
    communes (codes avec point final). Renvoie un GeoDataFrame.
    """
    from data_loader import charger_geometries_communes

    gdf = charger_geometries_communes().copy()
    gdf["code_norm"] = gdf["Code_Commu"].astype(str).str.rstrip(".")
    table = table_spai.copy()
    table["code_norm"] = table["code"].astype(str).str.rstrip(".")
    return gdf.merge(table, on="code_norm", how="left")


# ---------------------------------------------------------------------------
# Cartes choroplèthes SPAI
# ---------------------------------------------------------------------------

def tracer_carte_spai(
    table_spai: pd.DataFrame,
    titre: str,
    chemin_png: Path,
    montrer_sites_simules: bool = False,
) -> None:
    """Carte choroplèthe du SPAI par commune (en lits/1000 hab.)."""
    gdf = _joindre_spai_et_geometries(table_spai)

    cmap = ListedColormap(COULEURS_SPAI)
    norm = BoundaryNorm(BORNES_SPAI, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    gdf.plot(
        column="spai_pour_mille",
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
    )

    # Légende avec les classes
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COULEURS_SPAI[i], ec="white")
        for i in range(len(ETIQUETTES_SPAI))
    ]
    ax.legend(
        handles, ETIQUETTES_SPAI,
        title="SPAI (lits / 1 000 hab.)",
        loc="lower left",
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )

    if montrer_sites_simules:
        for nom, (lon, lat) in SITES_SIMULES_COORDS.items():
            ax.plot(lon, lat, marker="*", markersize=22,
                    markerfacecolor="#ffcc00", markeredgecolor="black",
                    linestyle="none")
            ax.annotate(nom, xy=(lon, lat), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=10, fontweight="bold")

    ax.set_title(titre, fontsize=14, fontweight="bold", pad=14)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def tracer_carte_gain_spai(
    table_initial: pd.DataFrame,
    table_simule: pd.DataFrame,
    chemin_png: Path,
) -> None:
    """Carte choroplèthe du gain absolu SPAI (simulé − initial)."""
    fusion = table_initial[["code", "nom", "spai_pour_mille"]].merge(
        table_simule[["code", "spai_pour_mille"]],
        on="code",
        suffixes=("_initial", "_simule"),
    )
    fusion["gain"] = fusion["spai_pour_mille_simule"] - fusion["spai_pour_mille_initial"]
    gdf = _joindre_spai_et_geometries(
        fusion.rename(columns={"gain": "spai_pour_mille"})
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    vmax = max(abs(gdf["spai_pour_mille"].min()),
               abs(gdf["spai_pour_mille"].max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    gdf.plot(
        column="spai_pour_mille",
        cmap="RdYlGn",
        norm=norm,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
        legend=True,
        legend_kwds={
            "label": "Gain SPAI (lits / 1 000 hab.)",
            "shrink": 0.65,
        },
    )

    for nom, (lon, lat) in SITES_SIMULES_COORDS.items():
        ax.plot(lon, lat, marker="*", markersize=22,
                markerfacecolor="#ffcc00", markeredgecolor="black",
                linestyle="none")
        ax.annotate(nom, xy=(lon, lat), xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=10, fontweight="bold")

    ax.set_title(
        "Gain d'accessibilité après ajout des 3 cliniques (Khénifra, Souk Sebt, Demnate)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagramme de Pareto
# ---------------------------------------------------------------------------

def tracer_diagramme_pareto(
    table_pareto: pd.DataFrame,
    chemin_png: Path,
) -> None:
    """Scatter (X, Y) des candidats avec front de Pareto en surbrillance."""
    fig, ax = plt.subplots(figsize=(10, 7))

    pareto = table_pareto[table_pareto["statut"] == "Pareto"]
    domines = table_pareto[table_pareto["statut"] == "Dominé"]

    ax.scatter(
        domines["X_demande_drainee_amo"], domines["Y_delta_spai_x1e6"],
        s=120, c="lightgray", edgecolor="black", label="Dominé", zorder=2,
    )
    ax.scatter(
        pareto["X_demande_drainee_amo"], pareto["Y_delta_spai_x1e6"],
        s=200, c="#d62728", edgecolor="black", label="Pareto-optimal",
        zorder=3, marker="*",
    )

    # Étiquettes des sites
    for _, ligne in table_pareto.iterrows():
        ax.annotate(
            ligne["site"],
            xy=(ligne["X_demande_drainee_amo"], ligne["Y_delta_spai_x1e6"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10,
            fontweight="bold" if ligne["statut"] == "Pareto" else "normal",
        )

    # Tracé du front (ligne reliant les points Pareto, triés par X croissant)
    if len(pareto) > 1:
        front = pareto.sort_values("X_demande_drainee_amo")
        ax.plot(front["X_demande_drainee_amo"], front["Y_delta_spai_x1e6"],
                linestyle="--", color="#d62728", alpha=0.6,
                label="Front de Pareto")

    ax.set_xlabel("X — Demande solvable drainée AMO (habitants)", fontsize=11)
    ax.set_ylabel("Y — Gain d'accessibilité moyen ΔSPAI × 10⁶", fontsize=11)
    ax.set_title(
        "Analyse de Pareto bi-objectif — sites candidats (clinique 40 lits)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distribution SPAI
# ---------------------------------------------------------------------------

def tracer_distribution_spai(
    table_initial: pd.DataFrame,
    table_simule: pd.DataFrame,
    chemin_png: Path,
) -> None:
    """Histogramme superposé des distributions SPAI initial vs simulé."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(table_initial["spai_pour_mille"].max(),
                              table_simule["spai_pour_mille"].max()) + 0.05,
                       25)
    ax.hist(table_initial["spai_pour_mille"], bins=bins,
            alpha=0.55, color="#d73027", label="Initial",
            edgecolor="black", linewidth=0.4)
    ax.hist(table_simule["spai_pour_mille"], bins=bins,
            alpha=0.55, color="#1a9850", label="Simulé (+3 cliniques)",
            edgecolor="black", linewidth=0.4)

    ax.axvline(table_initial["spai_pour_mille"].mean(),
               color="#d73027", linestyle="--",
               label=f"Moyenne initiale ({table_initial['spai_pour_mille'].mean():.3f})")
    ax.axvline(table_simule["spai_pour_mille"].mean(),
               color="#1a9850", linestyle="--",
               label=f"Moyenne simulée ({table_simule['spai_pour_mille'].mean():.3f})")

    ax.set_xlabel("SPAI (lits / 1 000 hab.)", fontsize=11)
    ax.set_ylabel("Nombre de communes", fontsize=11)
    ax.set_title("Distribution du SPAI — État initial vs scénario simulé",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tableau de bord — indicateurs synthétiques
# ---------------------------------------------------------------------------

def tracer_indicateurs_synthetiques(
    resume_comparaison: pd.DataFrame,
    chemin_png: Path,
) -> None:
    """Barres horizontales comparant les principaux indicateurs."""
    indicateurs_a_tracer = [
        "SPAI moyen (lits/1000 hab.)",
        "SPAI médian",
        "SPAI moyen pondéré par population",
        "Déserts médicaux absolus (SPAI = 0)",
    ]
    df = resume_comparaison[resume_comparaison["indicateur"].isin(indicateurs_a_tracer)]

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(df))
    hauteur = 0.4
    ax.barh(y - hauteur / 2, df["initial"], hauteur,
            label="Initial", color="#d73027", edgecolor="black")
    ax.barh(y + hauteur / 2, df["simulé"], hauteur,
            label="Simulé", color="#1a9850", edgecolor="black")

    for i, (vi, vs, var) in enumerate(zip(df["initial"], df["simulé"],
                                          df["variation_pct"])):
        annot = f"{var:+.1f}%" if pd.notna(var) else ""
        ax.text(max(vi, vs) * 1.02, i, annot, va="center", fontsize=9,
                color="#1a9850" if (var or 0) > 0 else "#d73027")

    ax.set_yticks(y)
    ax.set_yticklabels(df["indicateur"], fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Indicateurs synthétiques — Initial vs Simulé",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------

def generer_toutes_visualisations(
    table_initial: pd.DataFrame,
    table_simule: pd.DataFrame,
    resume_comparaison: pd.DataFrame,
    table_pareto: pd.DataFrame | None = None,
    dossier_sortie: Path | str | None = None,
) -> dict[str, Path]:
    """Génère l'ensemble des figures dans ``outputs/`` (ou autre dossier)."""
    if dossier_sortie is None:
        dossier_sortie = Path(__file__).resolve().parent.parent / "outputs"
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    fichiers: dict[str, Path] = {}

    fichiers["carte_spai_initial"] = dossier_sortie / "carte_spai_initial.png"
    tracer_carte_spai(table_initial, "SPAI initial — région Béni Mellal-Khénifra",
                      fichiers["carte_spai_initial"])

    fichiers["carte_spai_simule"] = dossier_sortie / "carte_spai_simule.png"
    tracer_carte_spai(table_simule,
                      "SPAI simulé — ajout de 3 cliniques (Pareto)",
                      fichiers["carte_spai_simule"],
                      montrer_sites_simules=True)

    fichiers["carte_gain_spai"] = dossier_sortie / "carte_gain_spai.png"
    tracer_carte_gain_spai(table_initial, table_simule, fichiers["carte_gain_spai"])

    fichiers["distribution_spai"] = dossier_sortie / "distribution_spai.png"
    tracer_distribution_spai(table_initial, table_simule, fichiers["distribution_spai"])

    fichiers["indicateurs_synthetiques"] = dossier_sortie / "indicateurs_synthetiques.png"
    tracer_indicateurs_synthetiques(resume_comparaison,
                                    fichiers["indicateurs_synthetiques"])

    if table_pareto is not None:
        fichiers["diagramme_pareto"] = dossier_sortie / "diagramme_pareto.png"
        tracer_diagramme_pareto(table_pareto, fichiers["diagramme_pareto"])

    return fichiers


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    from pareto import executer_analyse_pareto
    from simulation import executer_simulation

    print("=" * 78)
    print(" Génération des visualisations MH3SFCA-λ ".center(78, "="))
    print("=" * 78)
    print()

    print("→ Calcul des scénarios initial et simulé…")
    comparaison = executer_simulation(verbeux=False)

    print("→ Analyse de Pareto…")
    table_pareto = executer_analyse_pareto(verbeux=False)

    print("→ Génération des figures…")
    fichiers = generer_toutes_visualisations(
        table_initial=comparaison.table_initial,
        table_simule=comparaison.table_simule,
        resume_comparaison=comparaison.resume,
        table_pareto=table_pareto,
    )

    print()
    print(f"Figures sauvegardées dans : {fichiers['carte_spai_initial'].parent}")
    for nom, chemin in fichiers.items():
        print(f"  ✓ {nom:30s} → {chemin.name}")


if __name__ == "__main__":
    main()
