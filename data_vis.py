import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf



# ============================================================
# 1) Vérification minimale du DataFrame
# ============================================================
def check_var_dataframe(var_df):
    """
    Vérifie que le DataFrame contient bien les colonnes attendues.

    Paramètres
    ----------
    var_df : pd.DataFrame
        DataFrame contenant les séries du VAR.

    Retour
    ------
    None
    """
    expected_columns = ["date", "gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]

    missing_cols = [col for col in expected_columns if col not in var_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans var_df : {missing_cols}. "
            f"Colonnes disponibles : {list(var_df.columns)}"
        )


# ============================================================
# 2) Vérification minimale du DataFrame de prix
# ============================================================
def check_price_dataframe(price_df):
    """
    Vérifie que le DataFrame contient bien les colonnes attendues
    pour les séries de prix bruts.

    Paramètres
    ----------
    price_df : pd.DataFrame
        DataFrame contenant les prix bruts.

    Retour
    ------
    None
    """
    expected_columns = ["date", "gold_price", "dxy_price", "sp500_price", "vix_price"]

    missing_cols = [col for col in expected_columns if col not in price_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans price_df : {missing_cols}. "
            f"Colonnes disponibles : {list(price_df.columns)}"
        )


# ============================================================
# 3) Préparation simple de la date
# ============================================================
def prepare_dates(df):
    """
    S'assure que la colonne date est bien au format datetime.

    Paramètres
    ----------
    df : pd.DataFrame

    Retour
    ------
    pd.DataFrame
        Copie du DataFrame avec une colonne date au bon format.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


# ============================================================
# 4) Formatage des dates sur les axes matplotlib
# ============================================================
def format_date_axis(ax, major="year", date_format="%Y"):
    """
    Formate l'axe des dates pour améliorer la lisibilité.

    Paramètres
    ----------
    ax : matplotlib axis
        Axe à formatter.
    major : str
        Fréquence des ticks majeurs :
        - "year"
        - "2year"
        - "month"
        - "quarter"
    date_format : str
        Format d'affichage des dates.

    Retour
    ------
    None
    """
    if major == "year":
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    elif major == "2year":
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    elif major == "month":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    elif major == "quarter":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# ============================================================
# 5) Tracé de toutes les séries de rendements
# ============================================================
def plot_all_returns(var_df, figsize=(14, 8), major="2year", date_format="%Y"):
    """
    Trace les 4 séries de rendements sur un même graphique.
    """
    check_var_dataframe(var_df)
    df = prepare_dates(var_df)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df["date"], df["gold_ret"], label="Gold returns")
    ax.plot(df["date"], df["dxy_ret"], label="DXY returns")
    ax.plot(df["date"], df["sp500_ret"], label="S&P 500 returns")
    ax.plot(df["date"], df["vix_ret"], label="VIX log-diff")

    ax.set_title("Daily transformed series for VAR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log-returns / log-difference")
    ax.legend()
    ax.grid(True)

    format_date_axis(ax, major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()


# ============================================================
# 6) Tracé individuel d'une seule série
# ============================================================
def plot_single_series(df, series_col, title=None, figsize=(14, 5), major="2year", date_format="%Y"):
    """
    Trace une seule série temporelle.
    """
    if "date" not in df.columns:
        raise ValueError("La colonne 'date' est absente du DataFrame.")
    if series_col not in df.columns:
        raise ValueError(f"La colonne '{series_col}' est absente du DataFrame.")

    df = prepare_dates(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["date"], df[series_col])

    if title is None:
        title = f"Time series plot of {series_col}"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(series_col)
    ax.grid(True)

    format_date_axis(ax, major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()


# ============================================================
# 7) Tracé des 4 séries de rendements séparément
# ============================================================
def plot_returns_separately(var_df, figsize=(14, 10), major="2year", date_format="%Y"):
    """
    Trace les 4 séries dans 4 graphiques superposés.
    """
    check_var_dataframe(var_df)
    df = prepare_dates(var_df)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].plot(df["date"], df["gold_ret"])
    axes[0].set_title("Gold returns")
    axes[0].grid(True)

    axes[1].plot(df["date"], df["dxy_ret"])
    axes[1].set_title("DXY returns")
    axes[1].grid(True)

    axes[2].plot(df["date"], df["sp500_ret"])
    axes[2].set_title("S&P 500 returns")
    axes[2].grid(True)

    axes[3].plot(df["date"], df["vix_ret"])
    axes[3].set_title("VIX log-difference")
    axes[3].grid(True)
    axes[3].set_xlabel("Date")

    format_date_axis(axes[3], major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()


# ============================================================
# 8) Fusion des prix bruts
# ============================================================
def build_price_dataframe(cleaned_data):
    """
    Fusionne les séries de prix bruts nettoyées sur la date.
    """
    gold = cleaned_data["gold"][["date", "gold_price"]].copy()
    dxy = cleaned_data["dxy"][["date", "dxy_price"]].copy()
    sp500 = cleaned_data["sp500"][["date", "sp500_price"]].copy()
    vix = cleaned_data["vix"][["date", "vix_price"]].copy()

    price_df = gold.merge(dxy, on="date", how="inner")
    price_df = price_df.merge(sp500, on="date", how="inner")
    price_df = price_df.merge(vix, on="date", how="inner")

    price_df = price_df.sort_values("date").reset_index(drop=True)

    return price_df


# ============================================================
# 9) Tracé de tous les prix bruts sur un même graphique
# ============================================================
def plot_all_prices(price_df, figsize=(14, 8), major="2year", date_format="%Y"):
    """
    Trace les 4 séries de prix bruts sur un même graphique.
    """
    check_price_dataframe(price_df)
    df = prepare_dates(price_df)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df["date"], df["gold_price"], label="Gold price")
    ax.plot(df["date"], df["dxy_price"], label="DXY")
    ax.plot(df["date"], df["sp500_price"], label="S&P 500")
    ax.plot(df["date"], df["vix_price"], label="VIX")

    ax.set_title("Raw price series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Level")
    ax.legend()
    ax.grid(True)

    format_date_axis(ax, major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()


# ============================================================
# 10) Tracé des prix bruts séparément
# ============================================================
def plot_prices_separately(price_df, figsize=(14, 10), major="2year", date_format="%Y"):
    """
    Trace les 4 séries de prix bruts dans 4 graphiques superposés.
    """
    check_price_dataframe(price_df)
    df = prepare_dates(price_df)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].plot(df["date"], df["gold_price"])
    axes[0].set_title("Gold price")
    axes[0].grid(True)

    axes[1].plot(df["date"], df["dxy_price"])
    axes[1].set_title("DXY")
    axes[1].grid(True)

    axes[2].plot(df["date"], df["sp500_price"])
    axes[2].set_title("S&P 500")
    axes[2].grid(True)

    axes[3].plot(df["date"], df["vix_price"])
    axes[3].set_title("VIX")
    axes[3].grid(True)
    axes[3].set_xlabel("Date")

    format_date_axis(axes[3], major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()

def plot_gold_price_actual_vs_predicted_test(
    forecast_price_df,
    figsize=(14, 6)
):
    """
    Trace sur la période de test :
    - le prix observé de l'or
    - le prix prédit de l'or

    Paramètres
    ----------
    forecast_price_df : pd.DataFrame
        Doit contenir :
        - date
        - gold_price_actual
        - gold_price_pred
    """
    required_columns = ["date", "gold_price_actual", "gold_price_pred"]
    missing_cols = [col for col in required_columns if col not in forecast_price_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans forecast_price_df : {missing_cols}. "
            f"Colonnes disponibles : {list(forecast_price_df.columns)}"
        )

    df = forecast_price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    plt.figure(figsize=figsize)
    plt.plot(df["date"], df["gold_price_actual"], label="Prix observé de l'or")
    plt.plot(df["date"], df["gold_price_pred"], label="Prix prédit de l'or", linestyle="--")
    plt.title("Prix observé vs prix prédit de l'or sur le test")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) Rendements observés vs rendements prédits sur le test
# ============================================================
def plot_gold_returns_actual_vs_predicted_test(
    forecast_df,
    figsize=(14, 6)
):
    """
    Trace sur la période de test :
    - le rendement observé de l'or
    - le rendement prédit de l'or

    Paramètres
    ----------
    forecast_df : pd.DataFrame
        Doit contenir :
        - date
        - gold_ret_actual
        - gold_ret_pred
    """
    required_columns = ["date", "gold_ret_actual", "gold_ret_pred"]
    missing_cols = [col for col in required_columns if col not in forecast_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans forecast_df : {missing_cols}. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    plt.figure(figsize=figsize)
    plt.plot(df["date"], df["gold_ret_actual"], label="Rendement observé de l'or")
    plt.plot(df["date"], df["gold_ret_pred"], label="Rendement prédit de l'or", linestyle="--")
    plt.title("Rendements observés vs rendements prédits de l'or sur le test")
    plt.xlabel("Date")
    plt.ylabel("Rendement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




# ============================================================
# 12) Vérification du DataFrame des résidus VAR de l'or
# ============================================================
def check_gold_residuals_dataframe(gold_resid_df):
    """
    Vérifie que le DataFrame contient bien les colonnes attendues.

    Paramètres
    ----------
    gold_resid_df : pd.DataFrame
        DataFrame contenant :
        - date
        - gold_var_resid
    """
    required_columns = ["date", "gold_var_resid"]

    missing_cols = [col for col in required_columns if col not in gold_resid_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans gold_resid_df : {missing_cols}. "
            f"Colonnes disponibles : {list(gold_resid_df.columns)}"
        )


# ============================================================
# 13) Graphique des résidus de l'or
# ============================================================
def plot_gold_var_residuals(gold_resid_df, figsize=(14, 5)):
    """
    Trace la série temporelle des résidus de l'équation de l'or.
    """
    check_gold_residuals_dataframe(gold_resid_df)

    df = gold_resid_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    plt.figure(figsize=figsize)
    plt.plot(df["date"], df["gold_var_resid"])
    plt.title("Résidus de l'équation de l'or du VAR")
    plt.xlabel("Date")
    plt.ylabel("Résidu")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 14) Graphique des résidus au carré
# ============================================================
def plot_gold_var_squared_residuals(gold_resid_df, figsize=(14, 5)):
    """
    Trace la série temporelle des résidus au carré.
    """
    check_gold_residuals_dataframe(gold_resid_df)

    df = gold_resid_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["gold_var_resid_sq"] = df["gold_var_resid"] ** 2

    plt.figure(figsize=figsize)
    plt.plot(df["date"], df["gold_var_resid_sq"])
    plt.title("Résidus au carré de l'équation de l'or du VAR")
    plt.xlabel("Date")
    plt.ylabel("Résidu au carré")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 15) ACF des résidus au carré
# ============================================================
def plot_gold_var_squared_residuals_acf(gold_resid_df, lags=30, figsize=(10, 5)):
    """
    Trace l'ACF des résidus au carré.
    """
    check_gold_residuals_dataframe(gold_resid_df)

    df = gold_resid_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    squared_resid = (df["gold_var_resid"] ** 2).dropna()

    plt.figure(figsize=figsize)
    plot_acf(squared_resid, lags=lags)
    plt.title("ACF des résidus au carré de l'or")
    plt.tight_layout()
    plt.show()


####

"""
data_vis.py
===========
Visualisations pour le projet de prédiction du prix de l'or via VAR.

Fonctions exportées :
    - load_and_merge_data()          -> DataFrame fusionné mensuel
    - plot_timeseries_multi(df)      -> Séries temporelles multi-variables (Plotly + Matplotlib)
    - plot_correlation_heatmap(df)   -> Heatmap des corrélations (Plotly + Matplotlib)
    - plot_geopolitical_timeline(df) -> Timeline événementielle (Plotly + Matplotlib)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xlrd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Palette & style globaux
# ─────────────────────────────────────────────
COLORS = {
    "gold":  "#D4AF37",
    "dxy":   "#4A90D9",
    "sp500": "#27AE60",
    "vix":   "#E74C3C",
    "cpi":   "#9B59B6",
    "gpr":   "#E67E22",
}

PLOTLY_TEMPLATE = "plotly_dark"
MATPLOTLIB_STYLE = "dark_background"


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convertit un code hex (#RRGGBB) en chaîne rgba() pour Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "data"

# ─────────────────────────────────────────────
# 0) Chargement & fusion des données
# ─────────────────────────────────────────────

def _load_yahoo(name: str) -> pd.Series:
    """Charge un CSV Yahoo Finance et retourne la colonne 'Close' mensuelle."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    # Aplatir les colonnes multi-niveaux
    df.columns = ["_".join(col).strip() for col in df.columns]
    close_col = [c for c in df.columns if "Close" in c][0]
    series = df[close_col].dropna()
    series.index = pd.to_datetime(series.index)
    # Rééchantillonnage mensuel (dernier jour du mois)
    return series.resample("ME").last().rename(name)


def _load_cpi() -> pd.Series:
    """Charge le CPI depuis data/cpi.csv."""
    path = os.path.join(DATA_DIR, "cpi.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    series = df["cpi"].dropna()
    series.index = pd.to_datetime(series.index)
    return series.resample("ME").last().rename("cpi")


def _load_gpr() -> pd.Series:
    """Charge le GPR depuis le fichier Excel brut."""
    # Cherche le fichier brut (xls ou xlsx)
    raw_xls  = os.path.join(DATA_DIR, "gpr_raw.xls")
    raw_xlsx = os.path.join(DATA_DIR, "gpr_raw.xlsx")

    path = raw_xlsx if os.path.exists(raw_xlsx) else raw_xls
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier GPR introuvable dans {DATA_DIR}/")

    # Lecture robuste : on essaie plusieurs moteurs
    for engine in ("openpyxl", "xlrd", None):
        try:
            kwargs = {"engine": engine} if engine else {}
            xl = pd.read_excel(path, sheet_name=None, **kwargs)
            break
        except Exception:
            continue

    # On prend le premier onglet qui contient "GPR" dans ses colonnes
    gpr_series = None
    for sheet_name, sheet_df in xl.items():
        cols_lower = [str(c).lower() for c in sheet_df.columns]
        gpr_cols   = [c for c in sheet_df.columns if "gpr" in str(c).lower()]

        # Chercher une colonne date/year et une colonne GPR
        date_cols = [c for c in sheet_df.columns
                     if any(k in str(c).lower() for k in ("date", "year", "month"))]

        if gpr_cols and date_cols:
            tmp = sheet_df[[date_cols[0], gpr_cols[0]]].copy()
            tmp.columns = ["date", "gpr"]
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            tmp = tmp.dropna(subset=["date", "gpr"])
            tmp = tmp.set_index("date").sort_index()
            gpr_series = tmp["gpr"].resample("ME").last().rename("gpr")
            print(f"GPR chargé depuis l'onglet '{sheet_name}' ({len(gpr_series)} obs)")
            break

        # Fallback : si la feuille a year+month
        if "year" in cols_lower and "month" in cols_lower:
            tmp = sheet_df.copy()
            tmp.columns = [str(c).lower() for c in tmp.columns]
            gpr_col = next((c for c in tmp.columns if "gpr" in c), None)
            if gpr_col:
                tmp["date"] = pd.to_datetime(
                    tmp["year"].astype(int).astype(str) + "-" +
                    tmp["month"].astype(int).astype(str).str.zfill(2) + "-01",
                    errors="coerce"
                )
                tmp = tmp.dropna(subset=["date", gpr_col])
                tmp = tmp.set_index("date").sort_index()
                gpr_series = tmp[gpr_col].resample("ME").last().rename("gpr")
                print(f"GPR chargé (year+month) depuis '{sheet_name}' ({len(gpr_series)} obs)")
                break

    if gpr_series is None:
        raise ValueError("Impossible de parser le fichier GPR. Vérifiez la structure Excel.")

    return gpr_series


def load_and_merge_data() -> pd.DataFrame:
    """
    Charge toutes les sources et retourne un DataFrame mensuel fusionné.
    Colonnes : gold, dxy, sp500, vix, cpi, gpr
    """
    print("Chargement des données...")
    gold  = _load_yahoo("gold")
    dxy   = _load_yahoo("dxy")
    sp500 = _load_yahoo("sp500")
    vix   = _load_yahoo("vix")
    cpi   = _load_cpi()
    gpr   = _load_gpr()

    df = pd.concat([gold, dxy, sp500, vix, cpi, gpr], axis=1)
    df = df.dropna()
    df.index.name = "date"
    print(f"DataFrame fusionné : {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"Période : {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# ÉVÉNEMENTS GÉOPOLITIQUES (timeline)
# ─────────────────────────────────────────────

EVENTS = [
    ("2008-09-15", "Faillite\nLehman Brothers",  "#E74C3C"),
    ("2010-05-01", "Crise dette\neuropéenne",      "#E67E22"),
    ("2011-08-01", "Pic historique\nor ($1900)",   "#D4AF37"),
    ("2016-06-23", "Brexit",                        "#9B59B6"),
    ("2018-03-01", "Guerre\ncommerciale US-Chine", "#4A90D9"),
    ("2020-03-11", "COVID-19\nPandémie",            "#E74C3C"),
    ("2020-08-01", "ATH Or\n($2075)",               "#D4AF37"),
    ("2022-02-24", "Invasion\nUkraine",             "#C0392B"),
    ("2023-10-07", "Conflit\nGaza",                 "#E67E22"),
    ("2024-04-01", "Nouveau ATH\nOr ($2300+)",      "#D4AF37"),
]


# ─────────────────────────────────────────────
# 1) SÉRIES TEMPORELLES MULTI-VARIABLES
# ─────────────────────────────────────────────

def plot_timeseries_multi(df: pd.DataFrame, save: bool = True):
    """
    Graphique 6 panneaux avec les séries normalisées + panneau or brut.
    Versions Plotly (interactive) et Matplotlib (publication).
    """

    # ── Normalisation base 100 ──────────────────
    df_norm = (df / df.iloc[0]) * 100

    var_labels = {
        "gold":  "Or (GC=F)",
        "dxy":   "Dollar Index (DXY)",
        "sp500": "S&P 500",
        "vix":   "VIX",
        "cpi":   "CPI (Inflation)",
        "gpr":   "Indice GPR",
    }

    # ══════════════════════════════════════════
    # PLOTLY — version interactive
    # ══════════════════════════════════════════
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        subplot_titles=[var_labels[v] for v in df.columns],
        vertical_spacing=0.08,
        horizontal_spacing=0.07,
    )

    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)]

    for i, col in enumerate(df.columns):
        r, c = positions[i]
        fig.add_trace(
            go.Scatter(
                x=df_norm.index,
                y=df_norm[col],
                name=var_labels[col],
                line=dict(color=COLORS[col], width=1.8),
                fill="tozeroy",
                fillcolor="rgba({},{},{},0.12)".format(
                    int(COLORS[col][1:3], 16),
                    int(COLORS[col][3:5], 16),
                    int(COLORS[col][5:7], 16),
                ),
                hovertemplate="%{x|%b %Y}<br>%{y:.1f}<extra></extra>",
            ),
            row=r, col=c
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="<b>Évolution des variables du modèle VAR (base 100 = Jan 2005)</b>",
            font=dict(size=18, color="#D4AF37"),
            x=0.5,
        ),
        height=750,
        showlegend=False,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="#CCCCCC", family="Georgia"),
        margin=dict(t=80, b=40, l=50, r=30),
    )

    fig.update_xaxes(
        gridcolor="#1E1E1E", tickformat="%Y",
        tickfont=dict(size=10)
    )
    fig.update_yaxes(gridcolor="#1E1E1E", tickfont=dict(size=10))

    if save:
        path = os.path.join(OUTPUT_DIR, "timeseries_multi.html")
        fig.write_html(path)
        print(f"[Plotly] Séries temporelles → {path}")

    fig.show()


# ─────────────────────────────────────────────
# 2) HEATMAP DES CORRÉLATIONS
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    """
    Heatmap des corrélations de Pearson (niveaux + différences premières).
    Versions Plotly et Matplotlib.
    """

    var_labels = {
        "gold":  "Or",
        "dxy":   "DXY",
        "sp500": "S&P 500",
        "vix":   "VIX",
        "cpi":   "CPI",
        "gpr":   "GPR",
    }

    df_levels = df.rename(columns=var_labels)
    df_diff   = df.diff().dropna().rename(columns=var_labels)

    corr_lvl  = df_levels.corr()
    corr_diff = df_diff.corr()

    labels = list(var_labels.values())

    # ══════════════════════════════════════════
    # PLOTLY — deux heatmaps côte à côte
    # ══════════════════════════════════════════
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Corrélations — Niveaux</b>",
            "<b>Corrélations — Différences premières</b>",
        ],
        horizontal_spacing=0.12,
    )

    def _make_heatmap(corr_df, col):
        z    = corr_df.values
        text = [[f"{v:.2f}" for v in row] for row in z]
        fig.add_trace(
            go.Heatmap(
                z=z, x=labels, y=labels,
                text=text, texttemplate="%{text}",
                colorscale=[
                    [0.0,  "#2980B9"],
                    [0.5,  "#111111"],
                    [1.0,  "#D4AF37"],
                ],
                zmid=0, zmin=-1, zmax=1,
                showscale=(col == 2),
                colorbar=dict(
                    title="ρ",
                    tickfont=dict(color="#CCCCCC"),
                    title_font=dict(color="#CCCCCC"),
                ),
                hovertemplate="%{y} / %{x}<br>ρ = %{z:.3f}<extra></extra>",
            ),
            row=1, col=col
        )

    _make_heatmap(corr_lvl,  1)
    _make_heatmap(corr_diff, 2)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="<b>Matrice de corrélations des variables VAR</b>",
            font=dict(size=17, color="#D4AF37"), x=0.5
        ),
        height=500,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="#CCCCCC", family="Georgia"),
        margin=dict(t=80, b=40),
    )

    if save:
        path = os.path.join(OUTPUT_DIR, "correlation_heatmap.html")
        fig.write_html(path)
        print(f"[Plotly] Heatmap corrélations → {path}")

    fig.show()


# ─────────────────────────────────────────────
# 3) TIMELINE GÉOPOLITIQUE
# ─────────────────────────────────────────────

def plot_geopolitical_timeline(df: pd.DataFrame, save: bool = True):
    """
    Cours de l'or + GPR en fond + annotations des grands événements.
    Versions Plotly et Matplotlib.
    """

    gold_series = df["gold"]
    gpr_series  = df["gpr"]

    # ══════════════════════════════════════════
    # PLOTLY
    # ══════════════════════════════════════════
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # GPR en fond (area)
    fig.add_trace(
        go.Scatter(
            x=gpr_series.index, y=gpr_series.values,
            name="GPR", fill="tozeroy",
            line=dict(color="#E67E22", width=0),
            fillcolor="rgba(230,126,34,0.18)",
            hovertemplate="%{x|%b %Y}<br>GPR: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Or (ligne principale)
    fig.add_trace(
        go.Scatter(
            x=gold_series.index, y=gold_series.values,
            name="Prix de l'or (USD/oz)",
            line=dict(color="#D4AF37", width=2.5),
            hovertemplate="%{x|%b %Y}<br>Or: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Annotations événements
    for date_str, label, color in EVENTS:
        dt = pd.Timestamp(date_str)
        if dt < gold_series.index[0] or dt > gold_series.index[-1]:
            continue
        # Trouver la valeur la plus proche
        idx = gold_series.index.get_indexer([dt], method="nearest")[0]
        y_val = float(gold_series.iloc[idx])

        fig.add_vline(
            x=dt, line_width=1, line_dash="dot",
            line_color=color, opacity=0.6,
        )
        fig.add_annotation(
            x=dt, y=y_val * 1.04,
            text=label.replace("\n", "<br>"),
            showarrow=True, arrowhead=2,
            arrowcolor=color, arrowsize=0.8, arrowwidth=1.2,
            font=dict(size=9, color=color),
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor=color, borderwidth=1,
            borderpad=3,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="<b>Prix de l'Or & Indice GPR — Chronologie des chocs géopolitiques</b>",
            font=dict(size=17, color="#D4AF37"), x=0.5
        ),
        height=560,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#111111",
        font=dict(color="#CCCCCC", family="Georgia"),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=80, b=60, l=60, r=60),
        hovermode="x unified",
    )

    fig.update_yaxes(
        title_text="Prix de l'or (USD/oz)", secondary_y=False,
        gridcolor="#1E1E1E", tickprefix="$",
    )
    fig.update_yaxes(
        title_text="Indice GPR", secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#1E1E1E", tickformat="%Y")

    if save:
        path = os.path.join(OUTPUT_DIR, "geopolitical_timeline.html")
        fig.write_html(path)
        print(f"[Plotly] Timeline géopolitique → {path}")

    fig.show()

# ─────────────────────────────────────────────
# 4) ÉVOLUTION NORMALISÉE — STYLE CLUSTER LINES
# ─────────────────────────────────────────────

def plot_normalized_evolution(df: pd.DataFrame, save: bool = True):
    """
    Toutes les variables normalisées (base 100) sur un seul graphique,
    style 'cluster lines' : chaque variable est une ligne colorée avec
    marqueurs, fond sombre, annotations des valeurs finales.
    Versions Plotly (interactive) et Matplotlib (publication).
    """

    df_norm = (df / df.iloc[0]) * 100

    var_labels = {
        "gold":  "Or (GC=F)",
        "dxy":   "Dollar Index (DXY)",
        "sp500": "S&P 500",
        "vix":   "VIX",
        "cpi":   "CPI (Inflation)",
        "gpr":   "Indice GPR",
    }

    # ══════════════════════════════════════════
    # PLOTLY
    # ══════════════════════════════════════════
    fig = go.Figure()

    for col in df_norm.columns:
        series = df_norm[col]
        final_val = series.iloc[-1]

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=var_labels[col],
            line=dict(color=COLORS[col], width=2),
            hovertemplate=f"<b>{var_labels[col]}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}<extra></extra>",
        ))

        # Annotation valeur finale
        fig.add_annotation(
            x=series.index[-1],
            y=final_val,
            text=f"<b>{var_labels[col]}</b><br>{final_val:.0f}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            font=dict(size=9, color=COLORS[col]),
        )

    # Ligne de référence base 100
    fig.add_hline(y=100, line_dash="dot", line_color="#444444", line_width=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="<b>Évolution normalisée des variables du modèle VAR (base 100 = Jan 2005)</b>",
            font=dict(size=17, color="#D4AF37"), x=0.5,
        ),
        height=550,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#111111",
        font=dict(color="#CCCCCC", family="Georgia"),
        xaxis=dict(gridcolor="#1E1E1E", tickformat="%Y"),
        yaxis=dict(gridcolor="#1E1E1E", title="Base 100"),
        legend=dict(
            orientation="v", x=1.01, y=1,
            bgcolor="rgba(0,0,0,0.4)", bordercolor="#333",
            borderwidth=1, font=dict(size=10),
        ),
        margin=dict(t=70, b=50, l=60, r=180),
        hovermode="x unified",
    )

    if save:
        path = os.path.join(OUTPUT_DIR, "normalized_evolution.html")
        fig.write_html(path)
        print(f"[Plotly] Évolution normalisée → {path}")

    fig.show()


# ─────────────────────────────────────────────
# 5) SCATTER MATRIX (PAIRPLOT)
# ─────────────────────────────────────────────

def plot_scatter_matrix(df: pd.DataFrame, save: bool = True):
    """
    Scatter matrix complète : chaque variable vs chaque autre,
    diagonale = distribution (KDE), hors-diagonale = nuage de points
    coloré par l'or (gradient).
    Versions Plotly (interactive) et Matplotlib (publication).
    """

    var_labels = {
        "gold":  "Or",
        "dxy":   "DXY",
        "sp500": "S&P 500",
        "vix":   "VIX",
        "cpi":   "CPI",
        "gpr":   "GPR",
    }
    cols   = list(df.columns)
    labels = [var_labels[c] for c in cols]

    # ══════════════════════════════════════════
    # PLOTLY — splom natif
    # ══════════════════════════════════════════
    # Couleur des points = valeur normalisée de l'or
    gold_norm = (df["gold"] - df["gold"].min()) / (df["gold"].max() - df["gold"].min())

    dimensions = [
        dict(label=var_labels[c], values=df[c].values)
        for c in cols
    ]

    fig = go.Figure(go.Splom(
        dimensions=dimensions,
        showupperhalf=False,
        diagonal_visible=True,
        marker=dict(
            color=gold_norm.values,
            colorscale=[
                [0.0, "#1a1a2e"],
                [0.3, "#4A90D9"],
                [0.7, "#D4AF37"],
                [1.0, "#E74C3C"],
            ],
            size=3,
            opacity=0.65,
            showscale=True,
            colorbar=dict(
                title="Or (normalisé)",
                title_font=dict(color="#CCCCCC"),
                tickfont=dict(color="#CCCCCC"),
                len=0.5, x=1.02,
            ),
        ),
        text=[str(d.date()) for d in df.index],
        hovertemplate="Date: %{text}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="<b>Scatter Matrix des variables VAR</b><br>"
                 "<sup>Couleur = niveau du prix de l'or (bleu=bas, or=haut)</sup>",
            font=dict(size=16, color="#D4AF37"), x=0.5,
        ),
        height=750,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#111111",
        font=dict(color="#CCCCCC", family="Georgia", size=10),
        margin=dict(t=90, b=40, l=40, r=80),
    )

    if save:
        path = os.path.join(OUTPUT_DIR, "scatter_matrix.html")
        fig.write_html(path)
        print(f"[Plotly] Scatter matrix → {path}")

    fig.show()


# ─────────────────────────────────────────────
# 6) CARTE CHOROPLÈTHE GPR PAR PAYS
# ─────────────────────────────────────────────

def plot_gpr_choropleth(
    gpr_country_path: str,
    year: int = None,
    animate: bool = True,
    save: bool = True,
):
    """
    Carte choroplèthe du GPR par pays.

    Paramètres
    ----------
    gpr_country_path : str
        Chemin vers le fichier Excel GPR par pays
        (ex: 'data/gpr_country_raw.xlsx' depuis matteoiacoviello.com/gpr.htm).
    year : int, optional
        Si fourni et animate=False, affiche uniquement cette année.
    animate : bool
        Si True, produit une carte animée (slider annuel). Plotly uniquement.
    save : bool
        Sauvegarde le HTML dans figures/.

    Notes
    -----
    Le fichier GPR pays de Caldara & Iacoviello contient généralement
    les colonnes : Country, ISO3, Year, GPRC (GPR par pays).
    Ajuste `country_col`, `iso_col`, `year_col`, `value_col` si besoin.
    """

    # ── Chargement ──────────────────────────────
    ext = os.path.splitext(gpr_country_path)[1].lower()
    try:
        df_raw = pd.read_excel(gpr_country_path, engine="openpyxl" if ext == ".xlsx" else "xlrd")
    except Exception:
        df_raw = pd.read_excel(gpr_country_path)

    # Détection automatique des colonnes
    cols_lower = {c: c.lower() for c in df_raw.columns}

    country_col = next((c for c, cl in cols_lower.items()
                        if "country" in cl or "nation" in cl), df_raw.columns[0])
    iso_col     = next((c for c, cl in cols_lower.items()
                        if "iso" in cl or "code" in cl), None)
    year_col    = next((c for c, cl in cols_lower.items()
                        if "year" in cl), None)
    value_col   = next((c for c, cl in cols_lower.items()
                        if "gprc" in cl or "gpr" in cl), df_raw.columns[-1])

    print(f"Colonnes détectées → pays: '{country_col}' | iso: '{iso_col}' "
          f"| année: '{year_col}' | valeur: '{value_col}'")

    df_map = df_raw[[c for c in [country_col, iso_col, year_col, value_col]
                     if c is not None]].copy()
    df_map.columns = (["country"] +
                      (["iso3"] if iso_col else []) +
                      (["year"] if year_col else []) +
                      ["gpr"])
    df_map["gpr"] = pd.to_numeric(df_map["gpr"], errors="coerce")
    df_map = df_map.dropna(subset=["gpr"])

    if year_col:
        df_map["year"] = df_map["year"].astype(int)

    location_col = "iso3" if iso_col else "country"
    location_type = "ISO-3" if iso_col else "country names"

    # ── Plotly choroplèthe ────────────────────
    if animate and year_col:
        # Carte animée avec slider annuel
        df_map["year_str"] = df_map["year"].astype(str)

        fig = px.choropleth(
            df_map.sort_values("year"),
            locations=location_col,
            locationmode=location_type,
            color="gpr",
            hover_name="country",
            animation_frame="year_str",
            color_continuous_scale=[
                [0.0,  "#0D0D2B"],
                [0.2,  "#1a3a6b"],
                [0.4,  "#4A90D9"],
                [0.6,  "#F39C12"],
                [0.8,  "#E74C3C"],
                [1.0,  "#7B241C"],
            ],
            range_color=[df_map["gpr"].quantile(0.05),
                         df_map["gpr"].quantile(0.95)],
            projection="natural earth",
            labels={"gpr": "GPR", "year_str": "Année"},
            title="<b>Indice GPR par pays (Caldara & Iacoviello)</b>",
        )
    else:
        # Carte statique (année choisie ou moyenne)
        if year and year_col:
            df_year = df_map[df_map["year"] == year]
            subtitle = f" — {year}"
        elif year_col:
            df_year = df_map.groupby(location_col, as_index=False)["gpr"].mean()
            subtitle = " — Moyenne toutes années"
        else:
            df_year = df_map
            subtitle = ""

        fig = px.choropleth(
            df_year,
            locations=location_col,
            locationmode=location_type,
            color="gpr",
            hover_name="country" if "country" in df_year.columns else location_col,
            color_continuous_scale=[
                [0.0,  "#0D0D2B"],
                [0.2,  "#1a3a6b"],
                [0.4,  "#4A90D9"],
                [0.6,  "#F39C12"],
                [0.8,  "#E74C3C"],
                [1.0,  "#7B241C"],
            ],
            projection="natural earth",
            labels={"gpr": "GPR"},
            title=f"<b>Indice GPR par pays (Caldara & Iacoviello){subtitle}</b>",
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#0D0D0D",
        geo=dict(
            bgcolor="#0D0D0D",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#2A2A2A",
            showland=True, landcolor="#1A1A1A",
            showocean=True, oceancolor="#0D0D0D",
            showlakes=False,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=dict(
            title="GPR",
            title_font=dict(color="#CCCCCC"),
            tickfont=dict(color="#CCCCCC"),
            bgcolor="rgba(0,0,0,0.5)",
        ),
        title_font=dict(size=17, color="#D4AF37"),
        title_x=0.5,
        height=560,
        margin=dict(t=70, b=20, l=20, r=20),
        font=dict(color="#CCCCCC", family="Georgia"),
    )

    if animate and year_col:
        # Style du slider
        fig.layout.updatemenus[0].bgcolor = "#1A1A1A"
        fig.layout.updatemenus[0].font = dict(color="#CCCCCC")

    if save:
        fname = "gpr_choropleth_animated.html" if (animate and year_col) else "gpr_choropleth.html"
        path  = os.path.join(OUTPUT_DIR, fname)
        fig.write_html(path)
        print(f"[Plotly] Carte GPR → {path}")

    fig.show()