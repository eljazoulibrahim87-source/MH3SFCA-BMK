"""
Indicateurs statistiques additionnels — équité, classes, agrégations.

Utilisé par l'application interactive Streamlit (``app.py``) pour
calculer les KPI affichés dans l'onglet "Résultats" : coefficient de
Gini d'accessibilité, courbe de Lorenz, classes choroplèthes,
agrégations par province.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Bornes pour la classification choroplèthe du SPAI (lits / 1 000 hab.).
BORNES_CLASSES = [0.0, 1e-9, 0.076, 0.137, 0.250, 0.500, np.inf]
LIBELLES_CLASSES = [
    "Désert",
    "Très faible",
    "Faible",
    "Moyen",
    "Bon",
    "Très bon",
]
COULEURS_CLASSES = ["#404040", "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]


# ---------------------------------------------------------------------------
# Coefficient de Gini et courbe de Lorenz
# ---------------------------------------------------------------------------

def coefficient_gini(
    valeurs: Sequence[float],
    poids: Sequence[float] | None = None,
) -> float:
    """Coefficient de Gini (pondéré par défaut par la population).

    0 = égalité parfaite, 1 = inégalité maximale. Les valeurs négatives
    et NaN sont écartées.
    """
    x = np.asarray(valeurs, dtype=float)
    w = (np.asarray(poids, dtype=float) if poids is not None
         else np.ones_like(x))
    masque = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[masque], w[masque]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    ordre = np.argsort(x)
    x, w = x[ordre], w[ordre]
    cumw = np.cumsum(w)
    cumxw = np.cumsum(x * w)
    # Aire sous Lorenz / aire d'égalité parfaite
    return 1.0 - 2.0 * np.trapezoid(cumxw / cumxw[-1], cumw / cumw[-1])


def courbe_lorenz(
    valeurs: Sequence[float],
    poids: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Coordonnées (% population cumulée, % SPAI cumulé) de la Lorenz."""
    x = np.asarray(valeurs, dtype=float)
    w = (np.asarray(poids, dtype=float) if poids is not None
         else np.ones_like(x))
    masque = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[masque], w[masque]
    if len(x) == 0:
        return pd.DataFrame({"pop_cum": [0.0, 1.0], "spai_cum": [0.0, 1.0]})
    ordre = np.argsort(x)
    x, w = x[ordre], w[ordre]
    cumw = np.cumsum(w) / w.sum()
    sxw = (x * w).sum()
    cumxw = np.cumsum(x * w) / sxw if sxw > 0 else np.zeros_like(cumw)
    return pd.DataFrame({
        "pop_cum": np.concatenate(([0.0], cumw)),
        "spai_cum": np.concatenate(([0.0], cumxw)),
    })


# ---------------------------------------------------------------------------
# Classification SPAI en 6 classes
# ---------------------------------------------------------------------------

def classer_spai(spai_pour_mille: Sequence[float]) -> pd.Series:
    """Renvoie l'étiquette de classe (cf. LIBELLES_CLASSES)."""
    return pd.cut(
        spai_pour_mille,
        bins=BORNES_CLASSES,
        labels=LIBELLES_CLASSES,
        include_lowest=True,
        right=False,
    )


def repartition_classes(
    spai_pour_mille: Sequence[float],
) -> pd.DataFrame:
    """Compte les communes par classe d'accessibilité."""
    classes = classer_spai(spai_pour_mille)
    table = (
        pd.Series(classes, name="classe")
          .value_counts()
          .reindex(LIBELLES_CLASSES, fill_value=0)
          .rename_axis("classe")
          .reset_index(name="nb_communes")
    )
    table["couleur"] = COULEURS_CLASSES
    return table


# ---------------------------------------------------------------------------
# Agrégations géographiques
# ---------------------------------------------------------------------------

# Mapping Code_Provi → libellé province (région BMK = 6 provinces).
PROVINCES_BMK = {
    "05.091.": "Béni Mellal",
    "05.081.": "Azilal",
    "05.255.": "Fkih Ben Saleh",
    "05.311.": "Khouribga",
    "05.301.": "Khénifra",
    "10.011.": "Béni Mellal",   # fallback si code différent
}


def _code_province(code_commune: str) -> str:
    """Extrait les 6 premiers caractères significatifs (code province)."""
    code = str(code_commune)
    # Cherche le préfixe province (premiers 2 niveaux séparés par '.')
    morceaux = code.split(".")
    if len(morceaux) >= 2:
        return f"{morceaux[0]}.{morceaux[1]}."
    return code


def spai_par_province(
    table_spai: pd.DataFrame,
    communes_avec_provinces: pd.DataFrame,
) -> pd.DataFrame:
    """SPAI moyen pondéré par population pour chaque province.

    Parameters
    ----------
    table_spai : pd.DataFrame
        Sortie de :func:`mh3sfca.construire_table_spai` (colonnes ``code``,
        ``population``, ``spai_pour_mille``).
    communes_avec_provinces : pd.DataFrame
        Centroïdes des communes (colonne ``Code_Commu`` et ``Code_Provi``).
    """
    pont = communes_avec_provinces[["Code_Commu", "Code_Provi"]].copy()
    pont["code"] = pont["Code_Commu"].astype(str).str.rstrip(".")

    fusion = table_spai.merge(pont[["code", "Code_Provi"]], on="code", how="left")
    fusion["province"] = fusion["Code_Provi"].map(PROVINCES_BMK).fillna("Autres")

    def _moyenne_ponderee(groupe):
        pop = groupe["population"]
        spai = groupe["spai_pour_mille"]
        if pop.sum() == 0:
            return 0.0
        return (spai * pop).sum() / pop.sum()

    res = (
        fusion.groupby("province")
              .apply(_moyenne_ponderee, include_groups=False)
              .reset_index(name="spai_moyen_pondere")
    )
    pop_par_province = fusion.groupby("province")["population"].sum().reset_index()
    res = res.merge(pop_par_province, on="province")
    return res.sort_values("spai_moyen_pondere", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI synthétiques (pour les cartes de l'onglet "Résultats")
# ---------------------------------------------------------------------------

def kpi_synthetiques(table_spai: pd.DataFrame) -> dict:
    """Renvoie les indicateurs affichés dans le tableau de bord."""
    spai = table_spai["spai_pour_mille"]
    pop = table_spai["population"]

    idx_max = spai.idxmax()
    nom_top = table_spai.loc[idx_max, "nom"] if idx_max is not None else "—"
    nom_top = str(nom_top).replace("Commune de ", "").replace("Commune d'", "")

    return {
        "spai_moyen": float(spai.mean()),
        "spai_median": float(spai.median()),
        "spai_max": float(spai.max()),
        "spai_max_commune": nom_top,
        "spai_pondere": float((spai * pop).sum() / pop.sum()) if pop.sum() else 0.0,
        "deserts": int((spai == 0).sum()),
        "pop_dans_deserts": int(pop[spai == 0].sum()),
        "gini": float(coefficient_gini(spai.values, pop.values)),
    }
