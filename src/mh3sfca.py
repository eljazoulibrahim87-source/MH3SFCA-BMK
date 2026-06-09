"""
Implémentation de la méthode MH3SFCA-λ — accessibilité spatiale aux soins.

La méthode procède en trois étapes (cf. ``CLAUDE MH3SFCA.md``) :

    Étape 1 — Poids de Huff
        Huffᵢⱼ = (λⱼ·Sⱼ^α·exp(-dᵢⱼ²/β)) / Σₖ(λₖ·Sₖ^α·exp(-dᵢₖ²/β))
        avec dᵢₖ ≤ dₘₐₓ.

    Étape 2 — Ratio offre / demande pondéré
        Rⱼ = Sⱼ / Σᵢ(Huffᵢⱼ·Pᵢ)

    Étape 3 — Indice SPAI
        SPAIᵢ = Σⱼ Huffᵢⱼ·Rⱼ·exp(-dᵢⱼ²/β)

L'indice est ensuite multiplié par 1000 pour s'exprimer en "lits disponibles
pour 1000 habitants" (convention du mémoire).

L'implémentation est entièrement vectorisée avec NumPy : les matrices ont
pour dimensions (n_communes × n_etablissements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    ALPHA,
    BETA,
    D_MAX,
    LAMBDA_PRIVE,
    LAMBDA_PUBLIC,
    MOTS_CLES_PRIVE,
    NB_DESERTS_MEDICAUX_REF,
    PARAMS_DEFAUT,
    Parametres,
    SPAI_ECART_TYPE_REF,
    SPAI_MAX_REF,
    SPAI_MOYEN_REF,
)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class DonneesModele:
    """Données mises en forme pour le calcul MH3SFCA-λ.

    Attributes
    ----------
    communes : pd.DataFrame
        Une ligne par commune (n=135), colonnes ``code``, ``nom``,
        ``population``.
    etablissements : pd.DataFrame
        Une ligne par établissement, colonnes ``nom``, ``lits``, ``type``,
        ``lambda``.
    matrice_temps : np.ndarray
        Matrice de temps de trajet (n_communes × n_etablissements), en
        minutes. ``np.nan`` lorsque la paire est inaccessible (dᵢⱼ > dₘₐₓ).
    """

    communes: pd.DataFrame
    etablissements: pd.DataFrame
    matrice_temps: np.ndarray


@dataclass
class ResultatsMH3SFCA:
    """Résultats du calcul MH3SFCA-λ."""

    friction: np.ndarray         # f(dᵢⱼ) = exp(-d²/β), 0 si inaccessible
    utilite: np.ndarray          # λⱼ·Sⱼ^α·f(dᵢⱼ)
    huff: np.ndarray             # Huffᵢⱼ
    ratio_offre_demande: np.ndarray  # Rⱼ (vecteur)
    spai: np.ndarray             # SPAIᵢ brut (lits/habitant)
    spai_pour_mille: np.ndarray  # SPAIᵢ × 1000 (lits / 1000 hab)
    n_etablissements_accessibles: np.ndarray  # par commune


# ---------------------------------------------------------------------------
# Préparation des données depuis le format long de données_PFE.xlsx
# ---------------------------------------------------------------------------

def _normaliser_code_commune(code: str) -> str:
    """Supprime un éventuel point final pour homogénéiser les Code_Commu."""
    return str(code).rstrip(".")


def _determiner_lambda(
    type_etablissement: str,
    params: Parametres = PARAMS_DEFAUT,
) -> float:
    """Renvoie le coefficient λ selon le type d'établissement.

    Tout établissement contenant un mot-clé du secteur privé reçoit
    :attr:`Parametres.lambda_prive` ; les autres :attr:`lambda_public`.
    """
    libelle = str(type_etablissement).lower()
    if any(motcle in libelle for motcle in MOTS_CLES_PRIVE):
        return params.lambda_prive
    return params.lambda_public


def preparer_donnees(
    df_long: pd.DataFrame,
    params: Parametres = PARAMS_DEFAUT,
) -> DonneesModele:
    """Met en forme la matrice OD longue en structures vectorisées.

    Parameters
    ----------
    df_long : pd.DataFrame
        Format long avec au minimum les colonnes ``Code géographique``,
        ``Collectivités territoriales``, ``Population``,
        ``nom_etablissement``, ``type_etablissement``,
        ``TEMPS_TRAJET_MINUTES``, ``capacite_lits``.

    Returns
    -------
    DonneesModele
        Structure prête à passer aux fonctions des étapes 1 à 3.
    """
    df = df_long.copy()
    df["code_commune"] = df["Code géographique"].map(_normaliser_code_commune)

    # Communes : une ligne par code, population stable au sein d'un groupe
    communes = (
        df.groupby("code_commune", as_index=False)
          .agg(nom=("Collectivités territoriales", "first"),
               population=("Population", "first"))
          .rename(columns={"code_commune": "code"})
          .sort_values("code")
          .reset_index(drop=True)
    )

    # Établissements : une ligne par nom unique
    etablissements = (
        df.groupby("nom_etablissement", as_index=False)
          .agg(lits=("capacite_lits", "first"),
               type=("type_etablissement", "first"))
          .rename(columns={"nom_etablissement": "nom"})
          .sort_values("nom")
          .reset_index(drop=True)
    )
    etablissements["lambda"] = etablissements["type"].map(
        lambda t: _determiner_lambda(t, params)
    )

    # Matrice OD : reshape long → wide
    matrice = (
        df.pivot_table(
            index="code_commune",
            columns="nom_etablissement",
            values="TEMPS_TRAJET_MINUTES",
            aggfunc="first",
        )
        .reindex(index=communes["code"], columns=etablissements["nom"])
    )
    matrice_temps = matrice.to_numpy(dtype=float)

    # Au-delà de dmax → inaccessible (NaN)
    matrice_temps = np.where(matrice_temps > params.d_max, np.nan, matrice_temps)

    return DonneesModele(
        communes=communes,
        etablissements=etablissements,
        matrice_temps=matrice_temps,
    )


# ---------------------------------------------------------------------------
# Étape 1 — Poids de Huff
# ---------------------------------------------------------------------------

def calculer_friction(
    matrice_temps: np.ndarray,
    params: Parametres = PARAMS_DEFAUT,
) -> np.ndarray:
    """f(dᵢⱼ) = exp(-dᵢⱼ²/β), 0 quand dᵢⱼ > dₘₐₓ (NaN).

    Vérification rapide : f(dₘₐₓ) ≈ seuil_friction (seuil du modèle).
    """
    friction = np.exp(-np.square(matrice_temps) / params.beta)
    return np.nan_to_num(friction, nan=0.0)


def calculer_utilite(
    friction: np.ndarray,
    lits: np.ndarray,
    lambdas: np.ndarray,
    params: Parametres = PARAMS_DEFAUT,
) -> np.ndarray:
    """uᵢⱼ = λⱼ · Sⱼ^α · f(dᵢⱼ).

    ``lits`` et ``lambdas`` sont des vecteurs de taille n_etablissements.
    """
    attractivite = lambdas * np.power(lits, params.alpha)  # vecteur (n_etab,)
    return friction * attractivite[np.newaxis, :]


def calculer_huff(utilite: np.ndarray) -> np.ndarray:
    """Huffᵢⱼ = uᵢⱼ / Σₖ uᵢₖ — somme par commune normalisée à 1.

    Si une commune n'a aucun établissement accessible (somme nulle),
    la ligne reste à zéro.
    """
    somme_par_commune = utilite.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        huff = np.where(somme_par_commune > 0,
                        utilite / somme_par_commune,
                        0.0)
    return huff


# ---------------------------------------------------------------------------
# Étape 2 — Ratio offre / demande pondéré
# ---------------------------------------------------------------------------

def calculer_ratio_offre_demande(
    huff: np.ndarray,
    population: np.ndarray,
    lits: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Rⱼ = (λⱼ · Sⱼ) / Σᵢ (Huffᵢⱼ · Pᵢ).

    L'offre effective est pondérée par le coefficient de préférence λⱼ :
    seule la fraction de la capacité sectoriellement préférée alimente
    le ratio offre/demande (cohérent avec la colonne ``λ×Sj`` de la
    feuille ``Offre`` du classeur de référence).

    Renvoie un vecteur de taille n_etablissements. Si la demande
    pondérée est nulle (établissement non capté par aucune commune),
    Rⱼ = 0 par convention.
    """
    offre_effective = lambdas * lits
    demande_ponderee = (huff * population[:, np.newaxis]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(demande_ponderee > 0,
                         offre_effective / demande_ponderee,
                         0.0)
    return ratio


# ---------------------------------------------------------------------------
# Étape 3 — Indice SPAI
# ---------------------------------------------------------------------------

def calculer_spai(
    huff: np.ndarray,
    ratio: np.ndarray,
    friction: np.ndarray,
) -> np.ndarray:
    """SPAIᵢ = Σⱼ Huffᵢⱼ · Rⱼ · f(dᵢⱼ).

    Renvoie un vecteur de taille n_communes (lits/habitant).
    """
    contributions = huff * ratio[np.newaxis, :] * friction
    return contributions.sum(axis=1)


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------

def executer_modele(
    donnees: DonneesModele,
    params: Parametres = PARAMS_DEFAUT,
    verbeux: bool = True,
) -> ResultatsMH3SFCA:
    """Enchaîne les 3 étapes et renvoie l'ensemble des intermédiaires.

    Parameters
    ----------
    donnees : DonneesModele
        Issu de :func:`preparer_donnees`.
    params : Parametres
        Paramètres du modèle (dmax, β, α, λ). Par défaut les valeurs
        du mémoire.
    verbeux : bool
        Si True, affiche un résumé après chaque étape.
    """
    lits = donnees.etablissements["lits"].to_numpy(dtype=float)
    lambdas = donnees.etablissements["lambda"].to_numpy(dtype=float)
    population = donnees.communes["population"].to_numpy(dtype=float)

    if verbeux:
        print(f"  Communes        : {len(donnees.communes)}")
        print(f"  Établissements  : {len(donnees.etablissements)} "
              f"(privé={int((lambdas == params.lambda_prive).sum())}, "
              f"public={int((lambdas == params.lambda_public).sum())})")
        print(f"  Lits totaux     : {int(lits.sum())}")
        print()

    # Étape 1
    friction = calculer_friction(donnees.matrice_temps, params)
    utilite = calculer_utilite(friction, lits, lambdas, params)
    huff = calculer_huff(utilite)
    n_acc = (donnees.matrice_temps <= params.d_max).sum(axis=1)
    if verbeux:
        print("Étape 1 — Poids de Huff")
        print(f"  Friction max    : {friction.max():.4f}  "
              f"(seuil théorique f(dmax) ≈ 0.01)")
        sommes_huff = huff.sum(axis=1)
        sommes_non_nulles = sommes_huff[sommes_huff > 0]
        print(f"  Σ Huff par i    : moyenne = {sommes_non_nulles.mean():.6f} "
              f"(doit valoir 1)")
        print(f"  Étab. accessibles : min={n_acc.min()}, "
              f"médiane={int(np.median(n_acc))}, max={n_acc.max()}")
        print()

    # Étape 2
    ratio = calculer_ratio_offre_demande(huff, population, lits, lambdas)
    if verbeux:
        print("Étape 2 — Ratio offre/demande pondéré (Rⱼ)")
        print(f"  Rⱼ min / méd / max : {ratio.min():.6f} / "
              f"{np.median(ratio):.6f} / {ratio.max():.6f}")
        print()

    # Étape 3
    spai = calculer_spai(huff, ratio, friction)
    spai_pm = spai * 1000.0
    if verbeux:
        print("Étape 3 — Indice SPAI")
        print(f"  SPAI moyen        : {spai_pm.mean():.4f}  "
              f"(référence {SPAI_MOYEN_REF})")
        print(f"  SPAI écart-type   : {spai_pm.std(ddof=1):.4f}  "
              f"(référence {SPAI_ECART_TYPE_REF})")
        print(f"  SPAI max          : {spai_pm.max():.4f}  "
              f"(référence {SPAI_MAX_REF})")
        print(f"  Déserts (SPAI=0)  : {(spai == 0).sum()}  "
              f"(référence {NB_DESERTS_MEDICAUX_REF})")
        print()

    return ResultatsMH3SFCA(
        friction=friction,
        utilite=utilite,
        huff=huff,
        ratio_offre_demande=ratio,
        spai=spai,
        spai_pour_mille=spai_pm,
        n_etablissements_accessibles=n_acc,
    )


# ---------------------------------------------------------------------------
# Restitution en DataFrame
# ---------------------------------------------------------------------------

def construire_table_spai(
    donnees: DonneesModele,
    resultats: ResultatsMH3SFCA,
) -> pd.DataFrame:
    """Construit une table de synthèse (une ligne par commune)."""
    df = donnees.communes.copy()
    df["spai"] = resultats.spai
    df["spai_pour_mille"] = resultats.spai_pour_mille
    df["n_etab_accessibles"] = resultats.n_etablissements_accessibles
    df["rang"] = df["spai_pour_mille"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rang").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    from data_loader import charger_donnees_pfe

    print("=" * 78)
    print(" Calcul MH3SFCA-λ — Béni Mellal-Khénifra ".center(78, "="))
    print("=" * 78)
    print(f"Paramètres : dmax={D_MAX} min, β={BETA}, α={ALPHA}, "
          f"λ_privé={LAMBDA_PRIVE}, λ_public={LAMBDA_PUBLIC}")
    print()

    df_long = charger_donnees_pfe()
    donnees = preparer_donnees(df_long)
    resultats = executer_modele(donnees)

    table = construire_table_spai(donnees, resultats)
    print("Top 5 communes (meilleure accessibilité) :")
    print(table.head(5).to_string(index=False))
    print()
    print("5 communes les moins bien desservies (SPAI=0 incluses) :")
    print(table.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
