"""
Constantes de configuration du modèle MH3SFCA-λ — Béni Mellal-Khénifra.

Les valeurs ci-dessous sont celles justifiées dans le mémoire (voir
``CLAUDE MH3SFCA.md``). Toute modification doit être documentée.

Pour les analyses de sensibilité et l'application interactive, utiliser
la dataclass :class:`Parametres` qui permet de modifier dynamiquement
les paramètres sans recharger le module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Paramètres du modèle (NE PAS MODIFIER sans justification académique)
# ---------------------------------------------------------------------------

# Seuil maximal de temps de trajet (en minutes). Au-delà, on considère
# qu'un établissement n'est pas accessible depuis une commune.
# Justification : médiane des temps de trajet observés dans la région BMK.
D_MAX = 90.0

# Paramètre de friction spatiale (calibré sur D_MAX).
# Source : Jörg et al. (2019, p. 29), formule reliant β à dₘₐₓ.
BETA = 1758.89

# Élasticité de l'attractivité par rapport à la capacité (nombre de lits).
# α = 1 correspond à une attractivité linéaire (Huff, 1963).
ALPHA = 1.0

# Coefficient de préférence sectorielle λ (innovation de la méthode MH3SFCA-λ).
# Justification : répartition des dépenses AMO entre secteur privé et public
# au Maroc (≈ 90 % privé / 10 % public).
LAMBDA_PRIVE = 0.9
LAMBDA_PUBLIC = 0.1

# Taux de couverture AMO (Assurance Maladie Obligatoire) au Maroc.
# Sert à convertir la "demande potentielle" en "demande solvable" pour
# l'analyse de Pareto (cf. README de Pareto_MH3SFCA.xlsx).
TAUX_AMO = 0.86

# ---------------------------------------------------------------------------
# Étiquettes utilisées pour la sectorisation
# ---------------------------------------------------------------------------

# Mots-clés caractérisant les établissements du secteur privé.
# La détection se fait par recherche de sous-chaîne (insensible à la casse).
MOTS_CLES_PRIVE = ("clinique",)
MOTS_CLES_PUBLIC = ("hôpital", "hopital", "chu", "chr", "chp")

# ---------------------------------------------------------------------------
# Valeurs de référence (cible de validation)
# ---------------------------------------------------------------------------

SPAI_MOYEN_REF = 0.137
SPAI_ECART_TYPE_REF = 0.171
SPAI_MAX_REF = 0.707
NB_DESERTS_MEDICAUX_REF = 21  # communes avec SPAI = 0


# ---------------------------------------------------------------------------
# Conteneur dynamique pour l'application interactive
# ---------------------------------------------------------------------------

@dataclass
class Parametres:
    """Paramètres du modèle MH3SFCA-λ modifiables au runtime.

    β est dérivé automatiquement de :attr:`d_max` et :attr:`seuil_friction`
    suivant la calibration de Jörg et al. (2019) :
        β = −(dₘₐₓ²) / ln(seuil)
    """

    d_max: float = D_MAX
    seuil_friction: float = 0.01
    alpha: float = ALPHA
    lambda_prive: float = LAMBDA_PRIVE
    lambda_public: float = LAMBDA_PUBLIC
    taux_amo: float = TAUX_AMO

    @property
    def beta(self) -> float:
        """Paramètre de friction β recalculé à partir de d_max et du seuil."""
        return -(self.d_max ** 2) / math.log(self.seuil_friction)


PARAMS_DEFAUT = Parametres()
