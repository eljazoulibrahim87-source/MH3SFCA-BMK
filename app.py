"""
Application interactive MH3SFCA-λ — Béni Mellal-Khénifra.

Lancement :
    streamlit run app.py

5 onglets : Carte • Données • Résultats • Simulation • Méthode
Sidebar  : paramètres du modèle + bouton de recalcul.
"""

from __future__ import annotations

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
    _construire_lignes_simulees,
)


# ===========================================================================
# Configuration de la page
# ===========================================================================

st.set_page_config(
    page_title="MH3SFCA-λ • Béni Mellal-Khénifra",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 12px; padding: 14px 18px;
                border-left: 5px solid #4575b4; }
    div[data-testid="stMetricLabel"] > div { font-size: 0.75rem;
                color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
    h1, h2, h3 { color: #1e3a5f; }
    .badge-pareto { background: #d62728; color: white; padding: 2px 8px;
                    border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
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
def calculer_resultats(params_dict: dict, sites_simules: tuple = ()) -> dict:
    """Recalcule le SPAI pour les paramètres donnés.

    ``params_dict`` (dict hashable pour le cache) → :class:`Parametres`.
    ``sites_simules`` : tuple des noms de colonnes à injecter (vide = initial).
    """
    params = Parametres(**params_dict)
    df = charger_donnees_pfe()

    if sites_simules:
        # Injecte les sites simulés sélectionnés
        od_cand = charger_od_sites_candidats()
        com = charger_communes()
        mapping = {site: SITES_PARETO[site]
                   for site in sites_simules if site in SITES_PARETO}
        if mapping:
            lignes = _construire_lignes_simulees(od_cand, com, mapping)
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
if "sites_simules" not in st.session_state:
    st.session_state.sites_simules = ()


# ===========================================================================
# Sidebar — Paramètres
# ===========================================================================

with st.sidebar:
    st.markdown("## 🏥 MH3SFCA-λ")
    st.caption("Accessibilité spatiale aux soins — Béni Mellal-Khénifra")
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
    sites_choisis = st.multiselect(
        "Cliniques à ajouter (40 lits chacune)",
        options=list(SITES_PARETO.keys()),
        default=list(st.session_state.sites_simules),
        help="Sélectionne 0, 1, 2 ou 3 sites du front de Pareto à injecter.",
    )

    calculer = st.button("▶ Calculer MH3SFCA-λ", type="primary",
                          use_container_width=True)

    if calculer:
        st.session_state.params = Parametres(
            d_max=float(d_max), seuil_friction=float(seuil),
            alpha=float(alpha), lambda_prive=float(lambda_prive),
            lambda_public=float(lambda_public),
        )
        st.session_state.sites_simules = tuple(sites_choisis)
        st.toast("Calcul terminé ✓", icon="✅")


# ===========================================================================
# Calcul des résultats actuels
# ===========================================================================

params_courants = st.session_state.params
sites_courants = st.session_state.sites_simules

params_dict = {
    "d_max": params_courants.d_max,
    "seuil_friction": params_courants.seuil_friction,
    "alpha": params_courants.alpha,
    "lambda_prive": params_courants.lambda_prive,
    "lambda_public": params_courants.lambda_public,
}

# Toujours calculer l'état actuel
result_courant = calculer_resultats(params_dict, sites_courants)
table_courante = result_courant["table_spai"]
etabs_courants = result_courant["etablissements"]

# Calculer aussi l'état initial (sans sites simulés) pour comparaison
result_initial = calculer_resultats(params_dict, ())
table_initial = result_initial["table_spai"]


# ===========================================================================
# En-tête
# ===========================================================================

col_logo, col_titre, col_status = st.columns([1, 5, 2])
with col_logo:
    st.markdown("# 🏥")
with col_titre:
    st.markdown("## MH3SFCA-λ — Béni Mellal-Khénifra")
    st.caption(f"dₘₐₓ = {params_courants.d_max:.0f} min  •  "
                f"β = {params_courants.beta:.1f}  •  "
                f"α = {params_courants.alpha:.2f}  •  "
                f"λ_privé = {params_courants.lambda_prive:.2f}  •  "
                f"λ_public = {params_courants.lambda_public:.2f}")
with col_status:
    if sites_courants:
        st.markdown(f"🔵 **Scénario simulé** ({len(sites_courants)} site(s))")
    else:
        st.markdown("🟢 **État initial**")

st.divider()


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

        if variable == "SPAI×1000":
            col_valeur = "spai_pour_mille"
            unite = "lits / 1000 hab."
        elif variable == "Nb étab. accessibles":
            col_valeur = "n_etab_accessibles"
            unite = "établissements"
        else:
            col_valeur = "Populati_1"
            unite = "habitants"

        if choro:
            # Polygones (charge le shapefile)
            from data_loader import charger_geometries_communes
            gdf = charger_geometries_communes()
            gdf["code_norm"] = gdf["Code_Commu"].astype(str).str.rstrip(".")
            gdf = gdf.merge(valeurs.drop(columns=["nom", "population"]),
                              on="code_norm", how="left")

            def _couleur(val):
                if pd.isna(val):
                    return "#cccccc"
                if col_valeur == "spai_pour_mille":
                    for i, borne_sup in enumerate(BORNES_CLASSES[1:]):
                        if val < borne_sup:
                            return COULEURS_CLASSES[i]
                    return COULEURS_CLASSES[-1]
                # Pour autres variables : gradient continu
                vmin, vmax = gdf[col_valeur].min(), gdf[col_valeur].max()
                if vmax > vmin:
                    norm = (val - vmin) / (vmax - vmin)
                else:
                    norm = 0
                from matplotlib import colormaps
                cmap = colormaps["YlGnBu"]
                rgba = cmap(norm)
                return f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"

            for _, ligne in gdf.iterrows():
                couleur = _couleur(ligne[col_valeur])
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
                if col_valeur == "spai_pour_mille":
                    couleur = COULEURS_CLASSES[
                        max(0, min(len(COULEURS_CLASSES) - 1,
                                   sum(val >= b for b in BORNES_CLASSES[1:])))
                    ]
                else:
                    couleur = "#4575b4"
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
        # Inclure aussi les sites simulés actifs
        if sites_courants:
            for site in sites_courants:
                from simulation import LITS_CLINIQUE_SIMULEE
                coords_sim = {
                    "Khenifra":              (32.938293, -5.666568),
                    "Souk Sebt Ouled Nemma": (32.293866, -6.702381),
                    "Demnate":               (31.733306, -7.003886),
                }
                lat, lon = coords_sim[site]
                etabs = pd.concat([etabs, pd.DataFrame([{
                    "Nom_etablissement": f"[Simulée] {site}",
                    "Nombre_lits": LITS_CLINIQUE_SIMULEE,
                    "latitude": lat, "longitude": lon,
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

        # Légende personnalisée
        legende_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,.18); font-family: sans-serif;
                    font-size: 12px;">
          <b>SPAI×1000 (lits / 1 000 hab.)</b><br>
          <div style="margin-top:6px;">
        """
        for libelle, couleur in zip(LIBELLES_CLASSES, COULEURS_CLASSES):
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
    if not sites_courants:
        st.info("👉 Sélectionne un ou plusieurs sites dans la sidebar et clique "
                 "**Calculer** pour activer la comparaison.")
    else:
        kpi_i = kpi_synthetiques(table_initial)
        kpi_s = kpi_synthetiques(table_courante)

        st.markdown(f"**Sites injectés** : "
                     f"{', '.join(sites_courants)}  •  "
                     f"{len(sites_courants) * LITS_CLINIQUE_SIMULEE} lits ajoutés")

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
    st.markdown(r"""
    ## La méthode MH3SFCA-λ

    **M**odified **H**uff **3**-**S**tep **F**loating **C**atchment **A**rea
    avec coefficient de préférence sectorielle **λ**.

    ### Étape 1 — Poids de Huff

    Pour chaque paire (commune $i$, établissement $j$) :

    $$ \text{Huff}_{ij} = \frac{\lambda_j \cdot S_j^{\alpha} \cdot e^{-d_{ij}^2/\beta}}
        {\sum_k \lambda_k \cdot S_k^{\alpha} \cdot e^{-d_{ik}^2/\beta}} $$

    Propriété : $\sum_j \text{Huff}_{ij} = 1$ (conservation de la demande).

    ### Étape 2 — Ratio offre / demande pondéré

    $$ R_j = \frac{\lambda_j \cdot S_j}{\sum_i \text{Huff}_{ij} \cdot P_i} $$

    ### Étape 3 — Indice SPAI

    $$ \text{SPAI}_i = \sum_j \text{Huff}_{ij} \cdot R_j \cdot e^{-d_{ij}^2/\beta} $$

    L'indice est exprimé en **lits disponibles pour 1 000 habitants**
    (×1000).

    ---

    ### Paramètres et signification

    | Symbole | Valeur défaut | Rôle |
    |---|---|---|
    | $d_\max$ | 90 min | Seuil maximal de temps de trajet (médiane régionale BMK) |
    | $\beta$ | 1758.89 | Friction spatiale (calibré sur $d_\max$, seuil 0.01) |
    | $\alpha$ | 1.0 | Élasticité d'attractivité par rapport à la capacité |
    | $\lambda_\text{privé}$ | 0.9 | 90 % des dépenses AMO vers le secteur privé |
    | $\lambda_\text{public}$ | 0.1 | 10 % vers le secteur public |
    | $S_j$ | nb lits | Capacité de l'établissement $j$ |
    | $P_i$ | RGPH 2024 | Population de la commune $i$ |

    ### Indicateurs d'équité

    - **Coefficient de Gini** : mesure l'inégalité de distribution du SPAI
      entre communes (0 = égalité, 1 = inégalité totale).
    - **Courbe de Lorenz** : visualise l'écart à l'égalité parfaite.

    ---

    ### Sources

    - **Données démographiques** : Haut-Commissariat au Plan, RGPH 2024.
    - **Établissements de santé** : Carte Sanitaire 2025, MSPS.
    - **Matrice OD** : OpenStreetMap + plugin OD QGIS.
    - **Méthode** : Jörg et al. (2019) ; Subal et al. (2021).

    ---

    *Application développée dans le cadre du PFE de EL JAZOULI Brahim,
    FSEG Béni Mellal.*
    """)
