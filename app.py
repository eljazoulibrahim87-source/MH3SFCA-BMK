"""
Application interactive MH3SFCA-λ — Béni Mellal-Khénifra.

Lancement :
    streamlit run app.py

5 onglets : Carte • Données • Résultats • Simulation • Méthode
Sidebar  : paramètres du modèle + bouton de recalcul.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

# Permet l'import des modules du dossier src/
RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "src"))

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from analytics import (
    BORNES_CLASSES,
    COULEURS_CLASSES,
    LIBELLES_CLASSES,
    classer_spai,
    courbe_lorenz,
    kpi_synthetiques,
    repartition_classes,
    spai_par_province,
)
from config import Parametres
from data_loader import (
    charger_communes,
    charger_donnees_pfe,
    charger_etablissements,
    charger_od_sites_candidats,
)
from mh3sfca import construire_table_spai, executer_modele, preparer_donnees
from simulation import (
    LITS_CLINIQUE_SIMULEE,
    SITES_PARETO,
    construire_lignes_pour_site,
    _construire_lignes_simulees,
)

# Coordonnées des sites Pareto (utilisé pour l'affichage carte + lignes site)
COORDS_SITES_PARETO = {
    "Khenifra":              (32.938293, -5.666568),
    "Souk Sebt Ouled Nemma": (32.293866, -6.702381),
    "Demnate":               (31.733306, -7.003886),
}


def generer_template_od(communes_df: pd.DataFrame) -> bytes:
    """Génère un fichier Excel vierge à remplir avec la matrice OD du site.

    Le fichier contient les 135 communes ordonnées, l'utilisateur n'a qu'à
    remplir la colonne ``TEMPS_TRAJET_MINUTES``.
    """
    template = pd.DataFrame({
        "Commune": communes_df["nom"].values,
        "TEMPS_TRAJET_MINUTES": [None] * len(communes_df),
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template.to_excel(writer, index=False, sheet_name="OD")
    return buffer.getvalue()


# ===========================================================================
# Classification des variables affichées sur la carte
# ===========================================================================

# Tranches population (seuils choisis selon la distribution des communes BMK)
BORNES_POPULATION = [0, 20_000, 40_000, 60_000, 80_000, float("inf")]
LIBELLES_POPULATION = [
    "< 20 000 hab.",
    "20 000 – 40 000 hab.",
    "40 000 – 60 000 hab.",
    "60 000 – 80 000 hab.",
    "> 80 000 hab.",
]
COULEURS_POPULATION = ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"]

# Tranches nombre d'établissements accessibles
BORNES_N_ETAB = [0, 5, 10, 15, 20, 25, float("inf")]
LIBELLES_N_ETAB = [
    "0 – 5 établissements",
    "5 – 10",
    "10 – 15",
    "15 – 20",
    "20 – 25",
    "> 25",
]
COULEURS_N_ETAB = ["#ffffd9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8"]


def _config_variable(variable: str):
    """Retourne (colonne, bornes, libelles, couleurs, titre) selon la variable.

    Sert à rendre la carte choroplèthe ET la légende cohérentes quelle que
    soit la variable affichée.
    """
    if variable == "SPAI×1000":
        return ("spai_pour_mille", BORNES_CLASSES, LIBELLES_CLASSES,
                COULEURS_CLASSES, "SPAI×1000 (lits / 1 000 hab.)")
    if variable == "Population":
        return ("Populati_1", BORNES_POPULATION, LIBELLES_POPULATION,
                COULEURS_POPULATION, "Population (habitants)")
    # "Nb étab. accessibles"
    return ("n_etab_accessibles", BORNES_N_ETAB, LIBELLES_N_ETAB,
            COULEURS_N_ETAB, "Établissements accessibles")


# ===========================================================================
# Configuration de la page
# ===========================================================================

st.set_page_config(
    page_title="Aménagement sanitaire • Béni Mellal-Khénifra",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Filigrane thématique pour le bandeau d'en-tête : pin de carte, croix
# médicale, rose des vents, ECG — en blanc semi-transparent sur le gradient.
_SVG_BANDEAU = """<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'>
<g fill='white' fill-opacity='0.5'>
<g transform='translate(14, 12)'><path d='M12 0 C5.4 0 0 5.4 0 12 C0 22 12 38 12 38 C12 38 24 22 24 12 C24 5.4 18.6 0 12 0 Z'/><circle cx='12' cy='12' r='4' fill='white' fill-opacity='1'/></g>
<g transform='translate(116, 22)'><rect x='7' y='0' width='6' height='22' rx='1'/><rect x='0' y='8' width='22' height='6' rx='1'/></g>
</g>
<g transform='translate(102, 102)' fill='none' stroke='white' stroke-opacity='0.45' stroke-width='1.2'>
<circle cx='20' cy='20' r='18'/>
<line x1='20' y1='3' x2='20' y2='37'/>
<line x1='3' y1='20' x2='37' y2='20'/>
<path d='M20 4 L24 20 L20 36 L16 20 Z' fill='white' fill-opacity='0.35'/>
</g>
<g stroke='white' stroke-opacity='0.42' stroke-width='1.5' fill='none' stroke-linejoin='round' stroke-linecap='round'>
<path d='M2 128 L24 128 L31 113 L42 145 L52 103 L62 128 L86 128'/>
</g>
</svg>"""

_BANDEAU_BG_URI = (
    "data:image/svg+xml;base64,"
    + base64.b64encode(_SVG_BANDEAU.encode("utf-8")).decode("ascii")
)

CSS = f"""
<style>
    /* === Bandeau d'en-tête vif === */
    .bandeau-mh3sfca {{
        position: relative;
        background:
            linear-gradient(rgba(20, 30, 60, 0.22), rgba(20, 30, 60, 0.22)),
            linear-gradient(135deg,
                #67000d 0%, #b30000 14%, #d73027 28%, #fc8d59 42%,
                #fee08b 54%, #a6d96a 68%, #1a9850 84%, #006837 100%);
        border-radius: 16px;
        padding: 22px 30px 16px 30px;
        margin-bottom: 18px;
        color: white;
        box-shadow: 0 6px 24px rgba(30, 58, 95, 0.20);
        overflow: hidden;
    }}
    .bandeau-mh3sfca::before {{
        content: '';
        position: absolute;
        inset: 0;
        background-image: url("{_BANDEAU_BG_URI}");
        background-repeat: repeat;
        opacity: 0.55;
        pointer-events: none;
    }}
    .bandeau-contenu {{
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        flex-wrap: wrap;
    }}
    .bandeau-titre {{
        display: flex;
        align-items: center;
        gap: 18px;
        flex: 1;
        min-width: 0;
    }}
    .bandeau-icone {{
        font-size: 2.8rem;
        line-height: 1;
        filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.35));
    }}
    .bandeau-titre h1 {{
        font-size: 1.65rem;
        margin: 0;
        color: white !important;
        text-shadow: 0 1px 6px rgba(0, 0, 0, 0.45);
        font-weight: 700;
        letter-spacing: 0.3px;
        line-height: 1.25;
    }}
    .bandeau-soustitre {{
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        color: white;
        opacity: 0.95;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    }}
    .bandeau-badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 22px;
        font-weight: 600;
        font-size: 0.88rem;
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        white-space: nowrap;
        color: white;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
    }}
    .bandeau-params {{
        position: relative;
        z-index: 1;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.28);
        font-size: 0.82rem;
        color: white;
        opacity: 0.95;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        letter-spacing: 0.2px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
    }}

    /* === Reste du style général === */
    .block-container {{ padding-top: 1.5rem; padding-bottom: 1rem; }}
    .stMetric {{ background: #f8f9fa; border-radius: 12px; padding: 14px 18px;
                border-left: 5px solid #4575b4; }}
    div[data-testid="stMetricLabel"] > div {{ font-size: 0.75rem;
                color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.9rem; font-weight: 700; }}
    h1, h2, h3 {{ color: #1e3a5f; }}
    .badge-pareto {{ background: #d62728; color: white; padding: 2px 8px;
                    border-radius: 12px; font-size: 0.8rem; font-weight: 600; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ===========================================================================
# Chargement (mis en cache)
# ===========================================================================

@st.cache_data(show_spinner="Chargement des données…")
def charger_tout_en_cache():
    """Charge tous les jeux de données. Mis en cache pour rapidité."""
    return {
        "communes": charger_communes(),
        "etablissements": charger_etablissements(),
        "donnees_pfe": charger_donnees_pfe(),
        "od_candidats": charger_od_sites_candidats(),
    }


@st.cache_data(show_spinner="Calcul du modèle MH3SFCA-λ…")
def calculer_resultats(
    params_dict: dict,
    sites_pareto_config: tuple = (),
    site_perso_sig: tuple | None = None,
    site_perso_od_tuple: tuple | None = None,
) -> dict:
    """Recalcule le SPAI pour les paramètres et sites simulés donnés.

    Parameters
    ----------
    params_dict : dict
        Paramètres du modèle (sera converti en :class:`Parametres`).
    sites_pareto_config : tuple
        Tuple de tuples ``((site_name, capacity), ...)`` — capacité variable
        par site Pareto.
    site_perso_sig : tuple | None
        Métadonnées du site personnalisé ``(nom, lat, lon, capacite)`` ou None.
    site_perso_od_tuple : tuple | None
        Matrice OD du site personnalisé sous forme ``((commune, temps), ...)``.
    """
    params = Parametres(**params_dict)
    df = charger_donnees_pfe()
    com = charger_communes()

    # --- Sites Pareto (capacités variables) ---
    if sites_pareto_config:
        od_cand = charger_od_sites_candidats()
        for site_name, capacity in sites_pareto_config:
            if site_name not in SITES_PARETO or site_name not in od_cand.columns:
                continue
            od_series = od_cand.set_index("Commune")[site_name]
            lat, lon = COORDS_SITES_PARETO.get(site_name, (np.nan, np.nan))
            lignes = construire_lignes_pour_site(
                nom_etab=SITES_PARETO[site_name],
                capacite_lits=int(capacity),
                od_par_commune=od_series,
                table_communes=com,
                coord_lat=lat, coord_lon=lon,
                circ_sanitaire=site_name,
            )
            df = pd.concat([df, lignes], ignore_index=True)

    # --- Site personnalisé ---
    if site_perso_sig is not None and site_perso_od_tuple:
        nom, lat, lon, capacite = site_perso_sig
        od_series = pd.Series(dict(site_perso_od_tuple))
        lignes = construire_lignes_pour_site(
            nom_etab=f"[Perso] {nom}",
            capacite_lits=int(capacite),
            od_par_commune=od_series,
            table_communes=com,
            coord_lat=float(lat), coord_lon=float(lon),
            circ_sanitaire=nom,
        )
        df = pd.concat([df, lignes], ignore_index=True)

    donnees = preparer_donnees(df, params)
    resultats = executer_modele(donnees, params, verbeux=False)
    table = construire_table_spai(donnees, resultats)
    return {
        "table_spai": table,
        "etablissements": donnees.etablissements,
        "n_acc": resultats.n_etablissements_accessibles,
    }


DATA = charger_tout_en_cache()


# ===========================================================================
# État de session
# ===========================================================================

if "params" not in st.session_state:
    st.session_state.params = Parametres()
if "sites_pareto_config" not in st.session_state:
    # {nom_site: capacite_lits} — capacité variable par site
    st.session_state.sites_pareto_config = {}
if "site_perso_config" not in st.session_state:
    # {nom, lat, lon, capacite, od_dict} ou None
    st.session_state.site_perso_config = None


# ===========================================================================
# Sidebar — Paramètres
# ===========================================================================

with st.sidebar:
    st.markdown("## 🏥 Aménagement sanitaire")
    st.caption("Béni Mellal-Khénifra — Modèle MH3SFCA-λ")
    st.divider()

    st.markdown("### Paramètres du modèle")
    d_max = st.slider("dₘₐₓ — seuil de temps (min)", 30, 180,
                       int(st.session_state.params.d_max), step=5)
    seuil = st.number_input("Seuil f(dₘₐₓ)", 0.001, 0.5,
                             float(st.session_state.params.seuil_friction),
                             step=0.005, format="%.3f")

    col_a, col_b = st.columns(2)
    with col_a:
        alpha = st.number_input("α (attractivité)", 0.1, 3.0,
                                 float(st.session_state.params.alpha), step=0.1)
    with col_b:
        beta_calc = -(d_max ** 2) / np.log(seuil)
        st.metric("β (calculé)", f"{beta_calc:.1f}")

    col_c, col_d = st.columns(2)
    with col_c:
        lambda_prive = st.number_input("λ privé", 0.0, 1.0,
                                         float(st.session_state.params.lambda_prive),
                                         step=0.05)
    with col_d:
        lambda_public = st.number_input("λ public", 0.0, 1.0,
                                          float(st.session_state.params.lambda_public),
                                          step=0.05)

    st.divider()
    st.markdown("### Scénario simulé")

    # ----- Sites Pareto avec capacité variable -----
    st.markdown("**Sites Pareto-optimaux**")
    st.caption("Cocher les sites à ajouter et fixer leur capacité (en lits).")

    sites_pareto_actifs: dict[str, int] = {}
    for site_name in SITES_PARETO.keys():
        col_chk, col_cap = st.columns([3, 2])
        with col_chk:
            actif = st.checkbox(
                site_name,
                value=(site_name in st.session_state.sites_pareto_config),
                key=f"chk_{site_name}",
            )
        with col_cap:
            cap_defaut = st.session_state.sites_pareto_config.get(site_name, 40)
            cap = st.number_input(
                "lits", min_value=5, max_value=500, step=5,
                value=int(cap_defaut),
                key=f"cap_{site_name}",
                label_visibility="collapsed",
                disabled=not actif,
            )
            if actif:
                sites_pareto_actifs[site_name] = int(cap)

    # ----- Site personnalisé -----
    st.markdown("---")
    st.markdown("**Site personnalisé**")
    site_perso_actif = st.checkbox(
        "Ajouter un autre lieu d'implantation",
        value=(st.session_state.site_perso_config is not None),
        key="chk_site_perso",
    )

    site_perso_data = None
    if site_perso_actif:
        nom_perso = st.text_input(
            "Nom du site",
            value=(st.session_state.site_perso_config or {}).get(
                "nom", "Nouveau site"),
            key="nom_perso",
        )
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat_perso = st.number_input(
                "Latitude", min_value=30.0, max_value=35.0,
                value=float((st.session_state.site_perso_config or {})
                            .get("lat", 32.5)),
                step=0.001, format="%.4f", key="lat_perso",
            )
        with col_lon:
            lon_perso = st.number_input(
                "Longitude", min_value=-10.0, max_value=-4.0,
                value=float((st.session_state.site_perso_config or {})
                            .get("lon", -6.5)),
                step=0.001, format="%.4f", key="lon_perso",
            )
        cap_perso = st.number_input(
            "Capacité (lits)", min_value=5, max_value=500, step=5,
            value=int((st.session_state.site_perso_config or {})
                      .get("capacite", 40)),
            key="cap_perso",
        )

        st.download_button(
            "📥 Télécharger le modèle Excel",
            data=generer_template_od(DATA["communes"]),
            file_name="modele_OD_site_personnalise.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Le modèle contient les 135 communes — remplir la colonne "
                 "TEMPS_TRAJET_MINUTES avec le temps en minutes depuis le site.",
            use_container_width=True,
        )

        fichier_od = st.file_uploader(
            "Matrice OD du site (Excel)",
            type=["xlsx", "xls"],
            help="Fichier Excel à 2 colonnes : Commune | TEMPS_TRAJET_MINUTES.",
            key="upload_od_perso",
        )

        if fichier_od is not None:
            try:
                df_od_perso = pd.read_excel(fichier_od)
                if df_od_perso.shape[1] < 2:
                    st.error("⚠ Le fichier doit avoir au moins 2 colonnes.")
                else:
                    df_od_perso = df_od_perso.iloc[:, :2].copy()
                    df_od_perso.columns = ["Commune", "TEMPS_TRAJET_MINUTES"]
                    df_od_perso["Commune"] = df_od_perso["Commune"].astype(str).str.strip()
                    df_od_perso["TEMPS_TRAJET_MINUTES"] = pd.to_numeric(
                        df_od_perso["TEMPS_TRAJET_MINUTES"], errors="coerce")
                    df_od_perso = df_od_perso.dropna(subset=["TEMPS_TRAJET_MINUTES"])

                    noms_attendus = set(DATA["communes"]["nom"].astype(str).str.strip())
                    noms_fournis = set(df_od_perso["Commune"])
                    nb_apparies = len(noms_fournis & noms_attendus)
                    manquants = noms_attendus - noms_fournis

                    if nb_apparies >= 100:
                        st.success(
                            f"✓ Fichier validé : **{nb_apparies}** communes appariées."
                        )
                        if manquants:
                            with st.expander(f"⚠ {len(manquants)} commune(s) manquante(s)"):
                                st.caption(", ".join(sorted(list(manquants))[:30]))
                        od_filtre = df_od_perso[
                            df_od_perso["Commune"].isin(noms_attendus)
                        ]
                        site_perso_data = {
                            "nom": nom_perso,
                            "lat": float(lat_perso),
                            "lon": float(lon_perso),
                            "capacite": int(cap_perso),
                            "od_dict": dict(zip(
                                od_filtre["Commune"],
                                od_filtre["TEMPS_TRAJET_MINUTES"].astype(float),
                            )),
                        }
                    else:
                        st.error(
                            f"❌ Seulement **{nb_apparies}** communes appariées "
                            f"(minimum requis : 100). Vérifiez l'orthographe des noms."
                        )
            except Exception as e:
                st.error(f"❌ Erreur lecture du fichier : {e}")
        else:
            # Pas de fichier uploadé : on garde la config si elle existe en session
            if st.session_state.site_perso_config is not None:
                site_perso_data = {
                    **st.session_state.site_perso_config,
                    "nom": nom_perso,
                    "lat": float(lat_perso),
                    "lon": float(lon_perso),
                    "capacite": int(cap_perso),
                }

    st.divider()
    calculer = st.button("▶ Calculer MH3SFCA-λ", type="primary",
                          use_container_width=True)

    if calculer:
        st.session_state.params = Parametres(
            d_max=float(d_max), seuil_friction=float(seuil),
            alpha=float(alpha), lambda_prive=float(lambda_prive),
            lambda_public=float(lambda_public),
        )
        st.session_state.sites_pareto_config = sites_pareto_actifs
        st.session_state.site_perso_config = site_perso_data
        st.toast("Calcul terminé ✓", icon="✅")


# ===========================================================================
# Calcul des résultats actuels
# ===========================================================================

params_courants = st.session_state.params
sites_pareto_courant = st.session_state.sites_pareto_config or {}
site_perso_courant = st.session_state.site_perso_config

params_dict = {
    "d_max": params_courants.d_max,
    "seuil_friction": params_courants.seuil_friction,
    "alpha": params_courants.alpha,
    "lambda_prive": params_courants.lambda_prive,
    "lambda_public": params_courants.lambda_public,
}

# Convertit les structures non-hashables en tuples (pour @st.cache_data)
sites_pareto_tuple = tuple(sorted(sites_pareto_courant.items()))
perso_sig = None
perso_od_tuple = None
if site_perso_courant and site_perso_courant.get("od_dict"):
    perso_sig = (
        site_perso_courant["nom"],
        site_perso_courant["lat"],
        site_perso_courant["lon"],
        site_perso_courant["capacite"],
    )
    perso_od_tuple = tuple(sorted(site_perso_courant["od_dict"].items()))

# Toujours calculer l'état actuel
result_courant = calculer_resultats(
    params_dict, sites_pareto_tuple, perso_sig, perso_od_tuple,
)
table_courante = result_courant["table_spai"]
etabs_courants = result_courant["etablissements"]

# État initial (sans aucune simulation) pour comparaison
result_initial = calculer_resultats(params_dict, (), None, None)
table_initial = result_initial["table_spai"]

# Indicateur global : un scénario simulé est-il actif ?
nb_sites_simules = len(sites_pareto_courant) + (1 if site_perso_courant else 0)


# ===========================================================================
# En-tête
# ===========================================================================

# Bandeau d'en-tête vif (gradient SPAI + filigrane thématique + statut)
if nb_sites_simules > 0:
    statut_text = (
        f"🔵 Scénario simulé "
        f"({nb_sites_simules} site{'s' if nb_sites_simules > 1 else ''})"
    )
else:
    statut_text = "🟢 État initial"

params_resume = (
    f"d<sub>max</sub> = {params_courants.d_max:.0f} min &nbsp;•&nbsp; "
    f"β = {params_courants.beta:.1f} &nbsp;•&nbsp; "
    f"α = {params_courants.alpha:.2f} &nbsp;•&nbsp; "
    f"λ<sub>privé</sub> = {params_courants.lambda_prive:.2f} &nbsp;•&nbsp; "
    f"λ<sub>public</sub> = {params_courants.lambda_public:.2f}"
)

st.markdown(f"""
<div class="bandeau-mh3sfca">
  <div class="bandeau-contenu">
    <div class="bandeau-titre">
      <span class="bandeau-icone">🏥</span>
      <div>
        <h1>Aide à la décision pour l'aménagement sanitaire</h1>
        <p class="bandeau-soustitre">Béni Mellal-Khénifra &nbsp;—&nbsp; Modèle MH3SFCA-λ</p>
      </div>
    </div>
    <div class="bandeau-status"><span class="bandeau-badge">{statut_text}</span></div>
  </div>
  <div class="bandeau-params">{params_resume}</div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# Onglets
# ===========================================================================

tab_carte, tab_donnees, tab_resultats, tab_simulation, tab_methode = st.tabs([
    "🗺️  Carte", "📊  Données", "📈  Résultats",
    "🧪  Simulation", "ℹ️  Méthode",
])


# ---------------------------------------------------------------------------
# Onglet 1 — CARTE
# ---------------------------------------------------------------------------

with tab_carte:
    col_options, col_carte = st.columns([1, 4])

    with col_options:
        st.markdown("**Affichage**")
        variable = st.selectbox("Variable",
                                 ["SPAI×1000", "Nb étab. accessibles", "Population"])
        afficher_etabs = st.selectbox("Établissements",
                                        ["Tous", "Cliniques privées",
                                         "Hôpitaux publics", "Simulées", "Aucun"])
        choro = st.checkbox("Polygones de commune", value=True,
                             help="Décocher pour afficher des cercles aux centroïdes")
        cluster_etabs = st.checkbox("Grouper les établissements", value=False)

    with col_carte:
        # Centre carte sur la région
        carte = folium.Map(
            location=[32.5, -6.3], zoom_start=8,
            tiles="OpenStreetMap", control_scale=True,
        )

        # --- Couche communes : polygones choroplèthes OU bulles ---
        com_df = DATA["communes"].copy()
        com_df["code_norm"] = com_df["Code_Commu"].astype(str).str.rstrip(".")
        valeurs = table_courante[["code", "nom", "population",
                                   "spai_pour_mille", "n_etab_accessibles"]].copy()
        valeurs["code_norm"] = valeurs["code"].astype(str).str.rstrip(".")
        com_df = com_df.merge(valeurs.drop(columns=["nom", "population"]),
                                on="code_norm", how="left")

        # Configuration de la variable affichée (colonne + tranches + légende)
        col_valeur, bornes_v, libelles_v, couleurs_v, titre_legende = \
            _config_variable(variable)

        def _couleur_pour(val):
            """Détermine la couleur d'une commune selon la tranche de la variable."""
            if pd.isna(val):
                return "#cccccc"
            for i in range(len(bornes_v) - 1):
                if val < bornes_v[i + 1]:
                    return couleurs_v[i]
            return couleurs_v[-1]

        if choro:
            # Polygones (charge le shapefile)
            from data_loader import charger_geometries_communes
            gdf = charger_geometries_communes()
            gdf["code_norm"] = gdf["Code_Commu"].astype(str).str.rstrip(".")
            gdf = gdf.merge(valeurs.drop(columns=["nom", "population"]),
                              on="code_norm", how="left")

            for _, ligne in gdf.iterrows():
                couleur = _couleur_pour(ligne[col_valeur])
                popup = (
                    f"<b>{ligne['nom']}</b><br>"
                    f"Code : {ligne['Code_Commu']}<br>"
                    f"Population : {int(ligne['Populati_1']):,} hab.<br>"
                    f"SPAI×1000 : {ligne['spai_pour_mille']:.4f}<br>"
                    f"Étab. accessibles : {int(ligne['n_etab_accessibles'])}"
                ).replace(",", " ")
                folium.GeoJson(
                    ligne["geometry"].__geo_interface__,
                    style_function=lambda x, c=couleur: {
                        "fillColor": c, "color": "white",
                        "weight": 0.7, "fillOpacity": 0.78,
                    },
                    tooltip=folium.Tooltip(ligne["nom"], sticky=True),
                    popup=folium.Popup(popup, max_width=280),
                ).add_to(carte)
        else:
            # Bulles aux centroïdes
            for _, ligne in com_df.iterrows():
                val = ligne[col_valeur]
                if pd.isna(val):
                    continue
                couleur = _couleur_pour(val)
                rayon = 4 + (ligne["Populati_1"] / com_df["Populati_1"].max()) * 18
                popup = (
                    f"<b>{ligne['nom']}</b><br>"
                    f"Pop. : {int(ligne['Populati_1']):,} hab.<br>"
                    f"SPAI×1000 : {ligne['spai_pour_mille']:.4f}"
                ).replace(",", " ")
                folium.CircleMarker(
                    location=[ligne["latitude"], ligne["longitude"]],
                    radius=rayon, color="white", weight=1,
                    fill=True, fill_color=couleur, fill_opacity=0.85,
                    popup=folium.Popup(popup, max_width=260),
                    tooltip=ligne["nom"],
                ).add_to(carte)

        # --- Couche établissements ---
        etabs = DATA["etablissements"].copy()
        # Inclure aussi les sites Pareto sélectionnés (capacité variable)
        for site_name, cap in sites_pareto_courant.items():
            lat, lon = COORDS_SITES_PARETO[site_name]
            etabs = pd.concat([etabs, pd.DataFrame([{
                "Nom_etablissement": f"[Simulée] {site_name} ({cap} lits)",
                "Nombre_lits": cap,
                "latitude": lat, "longitude": lon,
                "type_etablissement": "clinique simulée",
            }])], ignore_index=True)
        # Site personnalisé
        if site_perso_courant:
            etabs = pd.concat([etabs, pd.DataFrame([{
                "Nom_etablissement": f"[Perso] {site_perso_courant['nom']} "
                                      f"({site_perso_courant['capacite']} lits)",
                "Nombre_lits": site_perso_courant["capacite"],
                "latitude": site_perso_courant["lat"],
                "longitude": site_perso_courant["lon"],
                "type_etablissement": "clinique simulée",
            }])], ignore_index=True)

        if afficher_etabs != "Aucun":
            filtre = {
                "Tous": etabs,
                "Cliniques privées": etabs[etabs["type_etablissement"].str.contains(
                    "clinique priv", case=False, na=False)],
                "Hôpitaux publics": etabs[etabs["type_etablissement"].str.contains(
                    "hopital|hôpital", case=False, na=False)],
                "Simulées": etabs[etabs["type_etablissement"].str.contains(
                    "simul", case=False, na=False)],
            }[afficher_etabs]

            cible = MarkerCluster().add_to(carte) if cluster_etabs else carte
            for _, etab in filtre.iterrows():
                if pd.isna(etab["latitude"]) or pd.isna(etab["longitude"]):
                    continue
                type_e = str(etab.get("type_etablissement", "")).lower()
                if "simul" in type_e:
                    icone = folium.Icon(color="orange", icon="star", prefix="fa")
                elif "priv" in type_e:
                    icone = folium.Icon(color="blue", icon="plus-square", prefix="fa")
                else:
                    icone = folium.Icon(color="red", icon="hospital", prefix="fa")
                folium.Marker(
                    location=[etab["latitude"], etab["longitude"]],
                    popup=folium.Popup(
                        f"<b>{etab['Nom_etablissement']}</b><br>"
                        f"Type : {etab.get('type_etablissement', '?')}<br>"
                        f"Lits : {etab.get('Nombre_lits', '?')}",
                        max_width=260,
                    ),
                    tooltip=etab["Nom_etablissement"],
                    icon=icone,
                ).add_to(cible)

        # Légende personnalisée — dynamique selon la variable affichée
        legende_html = f"""
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,.18); font-family: sans-serif;
                    font-size: 12px;">
          <b>{titre_legende}</b><br>
          <div style="margin-top:6px;">
        """
        for libelle, couleur in zip(libelles_v, couleurs_v):
            legende_html += (
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:{couleur};border:1px solid #999;margin-right:6px;'
                f'vertical-align:middle;"></span>{libelle}<br>'
            )
        legende_html += "</div></div>"
        carte.get_root().html.add_child(folium.Element(legende_html))

        st_folium(carte, use_container_width=True, height=620,
                   returned_objects=[])


# ---------------------------------------------------------------------------
# Onglet 2 — DONNÉES
# ---------------------------------------------------------------------------

with tab_donnees:
    sous_tab_com, sous_tab_etab, sous_tab_spai = st.tabs(
        ["Communes", "Établissements", "SPAI par commune"]
    )
    with sous_tab_com:
        st.dataframe(DATA["communes"], use_container_width=True, height=460,
                      hide_index=True)
    with sous_tab_etab:
        st.dataframe(DATA["etablissements"], use_container_width=True,
                      height=460, hide_index=True)
    with sous_tab_spai:
        st.dataframe(table_courante, use_container_width=True, height=460,
                      hide_index=True)
        st.download_button(
            "📥 Télécharger en CSV",
            data=table_courante.to_csv(index=False).encode("utf-8"),
            file_name="spai_par_commune.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Onglet 3 — RÉSULTATS
# ---------------------------------------------------------------------------

with tab_resultats:
    kpi = kpi_synthetiques(table_courante)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SPAI moyen ×1000", f"{kpi['spai_moyen']:.4f}",
                   help="lits disponibles pour 1 000 habitants")
    with col2:
        st.metric("Déserts médicaux", f"{kpi['deserts']}",
                   delta=f"{kpi['deserts'] - kpi_synthetiques(table_initial)['deserts']}",
                   delta_color="inverse",
                   help="communes avec SPAI = 0")
    with col3:
        st.metric("Gini d'accessibilité", f"{kpi['gini']:.3f}",
                   help="0 = égalité parfaite • 1 = inégalité maximale")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Médiane SPAI", f"{kpi['spai_median']:.4f}")
    with col5:
        st.metric("Maximum SPAI", f"{kpi['spai_max']:.4f}",
                   help=f"Atteint par {kpi['spai_max_commune']}")
    with col6:
        pop_k = kpi["pop_dans_deserts"] / 1000
        st.metric("Pop. en désert", f"{pop_k:.0f}k",
                   help="habitants dans des communes SPAI = 0")

    st.divider()

    col_donut, col_bars, col_provs = st.columns([1.1, 1.4, 1.2])

    with col_donut:
        st.markdown("**Distribution des classes**")
        repart = repartition_classes(table_courante["spai_pour_mille"])
        fig = px.pie(repart, values="nb_communes", names="classe",
                      hole=0.5, color="classe",
                      color_discrete_map=dict(zip(repart["classe"],
                                                    repart["couleur"])))
        fig.update_traces(textposition="inside", textinfo="value")
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
                           height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_bars:
        st.markdown("**Top 15 communes — SPAI×1000**")
        top = table_courante.head(15).copy()
        top["nom_court"] = top["nom"].str.replace("Commune de ", "")\
                                       .str.replace("Commune d'", "")
        fig = px.bar(top.sort_values("spai_pour_mille"),
                      x="spai_pour_mille", y="nom_court",
                      orientation="h", color="spai_pour_mille",
                      color_continuous_scale="YlGnBu")
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340,
                           xaxis_title="SPAI×1000", yaxis_title="",
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_provs:
        st.markdown("**SPAI moyen par province**")
        par_prov = spai_par_province(table_courante, DATA["communes"])
        fig = px.bar(par_prov, x="province", y="spai_moyen_pondere",
                      color="province",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340,
                           xaxis_title="", yaxis_title="SPAI×1000 (pondéré)",
                           showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Courbe de Lorenz
    st.markdown(f"**Courbe de Lorenz** — Gini = {kpi['gini']:.3f}")
    lorenz = courbe_lorenz(table_courante["spai_pour_mille"].values,
                            table_courante["population"].values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lorenz["pop_cum"], y=lorenz["spai_cum"],
                              mode="lines", name="Lorenz observée",
                              line=dict(color="#d62728", width=3),
                              fill="tozeroy", fillcolor="rgba(214,39,40,0.15)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="Égalité parfaite",
                              line=dict(color="#888", width=2, dash="dash")))
    fig.update_layout(xaxis_title="% population cumulée",
                       yaxis_title="% SPAI cumulé",
                       height=380, margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Onglet 4 — SIMULATION
# ---------------------------------------------------------------------------

with tab_simulation:
    st.markdown("### Comparaison État initial ↔ Scénario simulé")
    if nb_sites_simules == 0:
        st.info("👉 Sélectionne un ou plusieurs sites dans la sidebar et clique "
                 "**Calculer** pour activer la comparaison.")
    else:
        kpi_i = kpi_synthetiques(table_initial)
        kpi_s = kpi_synthetiques(table_courante)

        # Récapitulatif des sites injectés (capacité variable + site perso)
        lits_pareto = sum(sites_pareto_courant.values())
        lits_perso = site_perso_courant["capacite"] if site_perso_courant else 0
        total_lits = lits_pareto + lits_perso

        descriptions = []
        for site_name, cap in sites_pareto_courant.items():
            descriptions.append(f"{site_name} ({cap} lits)")
        if site_perso_courant:
            descriptions.append(
                f"{site_perso_courant['nom']} ({site_perso_courant['capacite']} lits, "
                f"personnalisé)"
            )
        st.markdown(
            f"**Sites injectés** ({nb_sites_simules}) — **{total_lits} lits ajoutés** :  \n"
            f"{' • '.join(descriptions)}"
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            d = (kpi_s["spai_moyen"] - kpi_i["spai_moyen"]) / max(kpi_i["spai_moyen"], 1e-9) * 100
            st.metric("SPAI moyen", f"{kpi_s['spai_moyen']:.4f}",
                       delta=f"{d:+.1f}%")
        with c2:
            st.metric("Déserts médicaux",
                       f"{kpi_s['deserts']}",
                       delta=f"{kpi_s['deserts'] - kpi_i['deserts']:+d}",
                       delta_color="inverse")
        with c3:
            d = (kpi_s["gini"] - kpi_i["gini"]) / max(kpi_i["gini"], 1e-9) * 100
            st.metric("Gini", f"{kpi_s['gini']:.3f}",
                       delta=f"{d:+.1f}%", delta_color="inverse")
        with c4:
            pop_diff = kpi_s["pop_dans_deserts"] - kpi_i["pop_dans_deserts"]
            st.metric("Pop. en désert", f"{kpi_s['pop_dans_deserts']/1000:.0f}k",
                       delta=f"{pop_diff/1000:+.0f}k", delta_color="inverse")

        # Comparaison distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=table_initial["spai_pour_mille"],
                                    name="Initial", opacity=0.55,
                                    marker_color="#d73027", nbinsx=24))
        fig.add_trace(go.Histogram(x=table_courante["spai_pour_mille"],
                                    name="Simulé", opacity=0.55,
                                    marker_color="#1a9850", nbinsx=24))
        fig.update_layout(barmode="overlay",
                           title="Distribution du SPAI × 1000",
                           xaxis_title="SPAI × 1000 (lits / 1 000 hab.)",
                           yaxis_title="Nombre de communes",
                           height=400, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Communes sortant du désert
        fusion = table_initial[["code", "nom", "population", "spai_pour_mille"]].merge(
            table_courante[["code", "spai_pour_mille"]], on="code",
            suffixes=("_initial", "_simule"))
        sorties = fusion[(fusion["spai_pour_mille_initial"] == 0)
                          & (fusion["spai_pour_mille_simule"] > 0)]\
                    .sort_values("spai_pour_mille_simule", ascending=False)
        st.markdown(f"**{len(sorties)} commune(s) sorties du désert médical**")
        if not sorties.empty:
            st.dataframe(
                sorties[["code", "nom", "population", "spai_pour_mille_simule"]]
                .rename(columns={"spai_pour_mille_simule": "SPAI×1000 simulé"}),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Onglet 5 — MÉTHODE
# ---------------------------------------------------------------------------

with tab_methode:
    st.markdown("## La méthode MH3SFCA-λ")
    st.markdown(
        "**M**odified **H**uff **3**-**S**tep **F**loating **C**atchment **A**rea "
        "avec coefficient de préférence sectorielle **λ**."
    )
    st.markdown(
        "Cette méthode adapte les FCA (*Floating Catchment Area*) en intégrant "
        "deux raffinements majeurs : "
        "**(1)** le **modèle de Huff** qui pondère la probabilité que les "
        "habitants d'une commune utilisent chaque établissement, et "
        "**(2)** le **coefficient sectoriel λ** qui reflète la préférence des "
        "habitants entre cliniques privées (~90 % des dépenses AMO au Maroc) "
        "et hôpitaux publics (~10 %)."
    )

    st.divider()

    # =======================================================================
    # ÉTAPE 1
    # =======================================================================
    st.markdown("### Étape 1 — Poids de Huff (probabilité d'utilisation)")
    st.markdown(
        r"Pour chaque paire (commune $i$, établissement $j$ accessible "
        r"depuis $i$), on calcule la **probabilité** que les habitants de $i$ "
        r"utilisent l'établissement $j$ parmi tous ceux à portée."
    )
    st.latex(r"""
        \text{Huff}_{ij} \;=\;
        \frac{\lambda_j \,\cdot\, S_j^{\,\alpha} \,\cdot\, e^{-d_{ij}^{\,2}/\beta}}
             {\displaystyle\sum_{k\,:\,d_{ik}\,\le\,d_{\max}}
              \lambda_k \,\cdot\, S_k^{\,\alpha} \,\cdot\, e^{-d_{ik}^{\,2}/\beta}}
    """)
    st.markdown(
        r"**Interprétation** — un établissement attire d'autant plus que :"
    )
    st.markdown(
        r"""
        - il est **gros** : $S_j$ élevé (nombre de lits)
        - il est **proche** : friction $e^{-d_{ij}^{2}/\beta}$ proche de 1
        - il est du **secteur préféré** : $\lambda_j$ élevé (privé ici)
        """
    )
    st.markdown(r"**Propriété fondamentale** — conservation de la demande :")
    st.latex(r"\sum_{j} \text{Huff}_{ij} \;=\; 1 \qquad \forall\, i")
    st.caption(
        "Chaque commune répartit l'intégralité de sa population entre les "
        "établissements accessibles (au plus à dₘₐₓ minutes)."
    )

    st.divider()

    # =======================================================================
    # ÉTAPE 2
    # =======================================================================
    st.markdown("### Étape 2 — Ratio offre / demande pondéré")
    st.markdown(
        r"Pour chaque établissement $j$, on calcule le **nombre de lits "
        r"effectivement disponibles par habitant qui le sollicite** (la "
        r"demande étant pondérée par le poids de Huff de l'étape 1)."
    )
    st.latex(r"""
        R_j \;=\; \frac{\lambda_j \,\cdot\, S_j}
                       {\displaystyle\sum_{i\,:\,d_{ij}\,\le\,d_{\max}}
                        \text{Huff}_{ij} \,\cdot\, P_i}
    """)
    st.markdown(
        r"""
        **Interprétation** — $R_j$ mesure la **rareté de l'offre** :
        - $R_j$ **élevé** → établissement peu sollicité par rapport à ses capacités
        - $R_j$ **faible** → établissement **saturé** (forte demande, peu de lits)

        L'innovation de la méthode MH3SFCA-λ par rapport au modèle M3SFCA
        classique tient à la pondération de l'offre par $\lambda_j$ au
        numérateur : seule la fraction sectoriellement préférée est
        comptabilisée comme "offre effective".
        """
    )

    st.divider()

    # =======================================================================
    # ÉTAPE 3
    # =======================================================================
    st.markdown("### Étape 3 — Indice d'accessibilité SPAI")
    st.markdown(
        r"Pour chaque commune $i$, on **agrège** les ratios $R_j$ de tous "
        r"les établissements accessibles, en les pondérant par le poids de "
        r"Huff et la friction spatiale."
    )
    st.latex(r"""
        \text{SPAI}_i \;=\;
        \sum_{j\,:\,d_{ij}\,\le\,d_{\max}}
        \text{Huff}_{ij} \,\cdot\, R_j \,\cdot\, e^{-d_{ij}^{\,2}/\beta}
    """)
    st.markdown(
        r"""
        **Interprétation** — SPAI mesure le nombre de **lits réellement
        accessibles par habitant** de la commune $i$. La double pondération
        (Huff + friction) garantit que :
        - les **établissements proches** comptent davantage
        - les **établissements préférés** (privé si λ_privé > λ_public) comptent davantage
        - les **établissements saturés** ($R_j$ faible) sont sous-pondérés

        L'indice est ensuite multiplié par **1 000** pour s'exprimer en
        **lits disponibles pour 1 000 habitants**, l'unité conventionnelle
        de la littérature en accessibilité aux soins (OMS, OCDE).
        """
    )

    st.divider()

    # =======================================================================
    # PARAMÈTRES
    # =======================================================================
    st.markdown("### Paramètres du modèle")
    st.markdown(
        "Les paramètres sont modifiables dans la sidebar (◀ à gauche). "
        "Voici leur signification et leur impact :"
    )

    with st.expander("📐  $d_{\\max}$  — Seuil de temps de trajet (90 min)"):
        st.markdown(
            r"""
            **Rôle** — seuil maximal au-delà duquel un établissement est
            considéré **inaccessible** depuis une commune. Toutes les paires
            $(i, j)$ avec $d_{ij} > d_{\max}$ sont **exclues** des trois
            étapes du calcul.

            **Justification de la valeur 90 minutes** — médiane des temps de
            trajet observés dans la région Béni Mellal-Khénifra (matrice
            origine-destination calculée via OpenStreetMap et le plugin OD
            de QGIS sur 4 760 paires commune × établissement).

            **Sensibilité** :
            """
        )
        st.markdown(
            r"""
            - $d_{\max}$ **plus petit** (ex : 60 min) → davantage de
              **déserts médicaux** (communes isolées non rattachées à un
              établissement)
            - $d_{\max}$ **plus grand** (ex : 120 min) → SPAI plus **lissé**
              géographiquement, mais hypothèse réaliste affaiblie
            """
        )

    with st.expander("📉  $\\beta$  — Paramètre de friction spatiale (1758.89)"):
        st.markdown(
            r"""
            **Rôle** — contrôle la **vitesse de décroissance** de
            l'attractivité d'un établissement avec la distance. Apparaît
            dans la fonction de friction gaussienne :
            """
        )
        st.latex(r"f(d) \;=\; e^{-d^{2}/\beta}")
        st.markdown(r"**Calibrage automatique** depuis $d_{\max}$ :")
        st.latex(r"\beta \;=\; -\frac{d_{\max}^{\,2}}{\ln(\text{seuil})}")
        st.markdown(
            r"""
            avec **seuil $= 0.01$**, c'est-à-dire que la friction au seuil
            $d_{\max}$ vaut $f(d_{\max}) = 1\%$ (probabilité résiduelle
            d'utilisation très faible). Cette calibration suit
            **Jörg et al. (2019, p. 29)**.

            **Sensibilité** :
            """
        )
        st.markdown(
            r"""
            - $\beta$ **petit** → résistance **forte** aux longs trajets, les
              habitants restent près de chez eux
            - $\beta$ **grand** → effet de la distance **atténué**, la portée
              des établissements s'étend
            """
        )

    with st.expander("📈  $\\alpha$  — Élasticité d'attractivité (1.0)"):
        st.markdown(
            r"""
            **Rôle** — élasticité de l'attractivité d'un établissement par
            rapport à sa **capacité** (nombre de lits $S_j$). Apparaît à
            l'étape 1 sous la forme $S_j^{\alpha}$.

            **Justification de $\alpha = 1$** — attractivité **linéaire** :
            doubler la capacité double l'attractivité. C'est la
            recommandation originale de **Huff (1963)** pour les modèles
            d'interaction spatiale.

            **Sensibilité** :
            """
        )
        st.markdown(
            r"""
            - $\alpha < 1$ : **concavité** — les très gros établissements
              voient leur attractivité **relative** diminuer (rendements
              décroissants de la taille)
            - $\alpha > 1$ : **convexité** — les gros établissements
              dominent (effet *winner-take-all*)
            """
        )

    with st.expander("⚖️  $\\lambda_{\\text{privé}}$ et $\\lambda_{\\text{public}}$  — Coefficients de préférence sectorielle"):
        st.markdown(
            r"""
            **Rôle** — poids sectoriels reflétant la **préférence
            d'utilisation** des habitants entre secteur privé (cliniques) et
            secteur public (hôpitaux). C'est l'**innovation majeure** de la
            méthode MH3SFCA-λ par rapport au M3SFCA classique.

            **Valeurs par défaut** :
            """
        )
        st.latex(r"\lambda_{\text{privé}} = 0.9 \qquad \lambda_{\text{public}} = 0.1")
        st.markdown(
            r"""
            **Justification empirique** — selon les rapports de la CNSS
            (Caisse Nationale de Sécurité Sociale), **90 % des dépenses
            de l'AMO** (Assurance Maladie Obligatoire) au Maroc sont
            orientées vers le **secteur privé**. Les 10 % restants vont
            au public.

            **Effet dans le modèle** — $\lambda_j$ intervient :
            1. À l'**étape 1** : multiplie l'attractivité (un établissement
               privé attire 9 × plus qu'un public à capacité égale)
            2. À l'**étape 2** : pondère l'offre effective $S_j$ comptée
               dans le ratio $R_j$

            **Sensibilité** :
            """
        )
        st.markdown(
            r"""
            - $\lambda_{\text{privé}} = 1$, $\lambda_{\text{public}} = 0$ :
              modèle **"tout privé"** (les hôpitaux publics sont ignorés)
            - $\lambda_{\text{privé}} = 0.5$, $\lambda_{\text{public}} = 0.5$ :
              **neutralité sectorielle** (revient au M3SFCA classique sans
              préférence)
            - $\lambda_{\text{privé}} = 0.1$, $\lambda_{\text{public}} = 0.9$ :
              modèle **"tout public"** (renversement de la préférence)
            """
        )

    with st.expander("👥  $P_i$ et $S_j$  — Demande et offre"):
        st.markdown(
            r"""
            **$P_i$ — Population de la commune $i$** : nombre d'habitants
            résidant dans la commune. Source : **RGPH 2024** (Recensement
            Général de la Population et de l'Habitat), Haut-Commissariat
            au Plan. Permet de ramener l'offre à la demande effective dans
            l'étape 2.

            **$S_j$ — Capacité de l'établissement $j$** : **nombre de lits**
            disponibles. Source : **Carte Sanitaire 2025**, Ministère de la
            Santé et de la Protection Sociale. C'est l'indicateur standard
            utilisé dans les méthodes FCA en santé publique
            (**Wang, 2012**), reflétant la capacité d'hospitalisation.
            """
        )

    st.divider()

    # =======================================================================
    # INDICATEURS D'ÉQUITÉ
    # =======================================================================
    st.markdown("### Indicateurs d'équité spatiale")
    st.markdown(
        r"""
        Au-delà du SPAI moyen, deux indicateurs synthétiques caractérisent
        la **distribution** de l'accessibilité entre les 135 communes :

        **Coefficient de Gini** — mesure l'inégalité spatiale du SPAI
        (pondérée par la population) :
        """
    )
    st.latex(r"G \;=\; 1 - 2 \int_{0}^{1} L(p)\, dp")
    st.markdown(
        r"""
        où $L(p)$ est la courbe de Lorenz.
        - $G = 0$ : **égalité parfaite** (toutes les communes ont le même SPAI)
        - $G = 1$ : **inégalité maximale** (toute l'accessibilité concentrée
          sur une seule commune)

        **Courbe de Lorenz** — représentation graphique : axe $x$ = part
        cumulée de population (triée par SPAI croissant), axe $y$ = part
        cumulée du SPAI. Plus la courbe s'écarte de la diagonale,
        plus l'inégalité est forte.
        """
    )

    st.divider()

    # =======================================================================
    # SOURCES
    # =======================================================================
    st.markdown("### Sources des données")
    st.markdown(
        """
        | Donnée | Source | Année |
        |---|---|---|
        | Population des communes | Haut-Commissariat au Plan (RGPH) | 2024 |
        | Centroïdes communaux | Haut-Commissariat au Plan / QGIS | 2024 |
        | Liste et capacité des établissements | Carte Sanitaire (MSPS) | 2025 |
        | Coordonnées des établissements | Géolocalisation MSPS / OSM | 2025 |
        | Matrice origine-destination (135 × 35) | OpenStreetMap + plugin OD QGIS | 2025 |
        | Taux d'AMO et répartition sectorielle | CNSS, Rapport annuel | 2024 |
        """
    )

    st.markdown("### Références bibliographiques")
    st.markdown(
        """
        - **Huff, D. L. (1963)**. *A probabilistic analysis of shopping center
          trade areas*. Land Economics, 39(1), 81–90.
        - **Wang, F. (2012)**. *Measurement, optimization, and impact of
          health care accessibility: a methodological review*. Annals of the
          Association of American Geographers, 102(5), 1104–1112.
        - **Jörg, R., Lenz, N., Wetz, S. & Widmer, M. (2019)**. *A modified
          two-step floating catchment area method to incorporate
          within-region attractiveness*.
        - **Subal, J., Paal, P. & Krisp, J. M. (2021)**. *Quantifying spatial
          accessibility of general practitioners by applying a modified Huff
          three-step floating catchment area*.
        """
    )

    st.caption(
        "Application développée dans le cadre du Projet de Fin d'Études "
        "de **EL JAZOULI Brahim**, Faculté des Sciences Économiques et de "
        "Gestion, Béni Mellal."
    )
