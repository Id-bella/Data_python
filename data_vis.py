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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import os
import warnings
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import skew, kurtosis
import xlrd

warnings.filterwarnings("ignore")

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
#  Préparation des données pour des visualisations plotly
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
    return series.resample("ME").mean().rename(name)


def _load_cpi() -> pd.Series:
    """Charge le CPI depuis data/cpi.csv."""
    path = os.path.join(DATA_DIR, "cpi.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    series = df["cpi"].dropna()
    series.index = pd.to_datetime(series.index)
    return series.resample("ME").mean().rename("cpi")


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


def plot_timeseries_multi(df: pd.DataFrame, save: bool = True):
    """
    Graphique 6 panneaux avec les séries normalisées + panneau or brut.
    Version Plotly (interactive).
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


def plot_geopolitical_timeline(df: pd.DataFrame, save: bool = True):
    """
    Cours de l'or + GPR en fond + annotations des grands événements.
    Versions Plotly et Matplotlib.
    """

    gold_series = df["gold"]
    gpr_series  = df["gpr"]

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


def plot_return_distributions(var_df, save=True):

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    variables = {
        'gold_ret': ("Or (GC=F)", "goldenrod"),
        'sp500_ret': ("S&P 500", "steelblue"),
        'dxy_ret': ("Dollar (DXY)", "seagreen"),
        'vix_ret': ("VIX", "firebrick"),
        'cpi_ret': ("CPI", "mediumpurple"),
        'gpr_ret': ("GPR (Géopolitique)", "darkorange"),
    }

    for ax, (col, (label, color)) in zip(axes, variables.items()):
        if col not in var_df.columns:
            ax.set_visible(False)
            continue

        data = var_df[col].dropna()

        # Histogramme
        ax.hist(data, bins=80, density=True, color=color, alpha=0.4, edgecolor='white')

        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)

        # Normale théorique
        mu, sigma = data.mean(), data.std()
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k--', linewidth=1.2, label='Normale')

        # KDE empirique
        kde = stats.gaussian_kde(data)
        ax.plot(x, kde(x), color=color, linewidth=1.8, label='KDE empirique')

        # Moments
        sk = skew(data)
        ku = kurtosis(data)

        ax.set_title(f"{label}\nskew={sk:.2f} | kurt={ku:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("Rendement log")
        ax.set_ylabel("Densité")

        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(data)
        ax.text(0.97, 0.95, f"JB p={jb_p:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                ha='right',
                va='top',
                color='red' if jb_p < 0.05 else 'green')

    plt.suptitle(
        "Distribution des rendements logarithmiques\n"
        "(courbe noire = normale | JB = test de normalité)",
        fontsize=12,
        y=1.01
    )

    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, "return_distributions.png"),
                    dpi=150, bbox_inches='tight')
    plt.show()

 
def plot_gold_daily(save: bool = True):
    """
    Trace le prix de l'or en données quotidiennes brutes (avant rééchantillonnage).
    Candlestick OHLC + moyennes mobiles 50j / 200j + volume en sous-graphique.
    """
 
    path = os.path.join(DATA_DIR, "gold.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}. Lance download_all() d'abord.")
 
    # ── Chargement du CSV brut Yahoo Finance ───
    df_raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df_raw.columns = ["_".join(col).strip() for col in df_raw.columns]
    df_raw.index   = pd.to_datetime(df_raw.index)
 
    open_col   = next(c for c in df_raw.columns if "Open"   in c)
    high_col   = next(c for c in df_raw.columns if "High"   in c)
    low_col    = next(c for c in df_raw.columns if "Low"    in c)
    close_col  = next(c for c in df_raw.columns if "Close"  in c and "Adj" not in c)
    volume_col = next((c for c in df_raw.columns if "Volume" in c), None)
 
    df_raw = df_raw.dropna(subset=[close_col])
 
    # Moyennes mobiles
    df_raw["ma50"]  = df_raw[close_col].rolling(50).mean()
    df_raw["ma200"] = df_raw[close_col].rolling(200).mean()
 
    # ── Subplots : prix + volume ────────────────
    rows        = 2 if volume_col else 1
    row_heights = [0.75, 0.25] if volume_col else [1.0]
 
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )
 
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_raw.index,
        open=df_raw[open_col], high=df_raw[high_col],
        low=df_raw[low_col],   close=df_raw[close_col],
        name="Or (OHLC)",
        increasing=dict(line=dict(color="#D4AF37"), fillcolor="#D4AF37"),
        decreasing=dict(line=dict(color="#6B4E0A"), fillcolor="#6B4E0A"),
    ), row=1, col=1)
 
    # MM 50j
    fig.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["ma50"],
        name="MM 50j",
        line=dict(color="#4A90D9", width=1.3, dash="dot"),
        hovertemplate="MM50: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
 
    # MM 200j
    fig.add_trace(go.Scatter(
        x=df_raw.index, y=df_raw["ma200"],
        name="MM 200j",
        line=dict(color="#E74C3C", width=1.3, dash="dash"),
        hovertemplate="MM200: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
 
    # Volume
    if volume_col:
        colors_vol = [
            "#D4AF37" if c >= o else "#6B4E0A"
            for c, o in zip(df_raw[close_col], df_raw[open_col])
        ]
        fig.add_trace(go.Bar(
            x=df_raw.index, y=df_raw[volume_col],
            name="Volume",
            marker_color=colors_vol,
            opacity=0.55,
            hovertemplate="%{x|%d %b %Y}<br>Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)
 
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(
            text="<b>Prix de l'Or — Données quotidiennes (2005–présent)</b>",
            font=dict(size=17, color="#D4AF37"), x=0.5,
        ),
        height=650,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#111111",
        font=dict(color="#CCCCCC", family="Georgia"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.02,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(t=80, b=40, l=60, r=30),
    )
 
    fig.update_xaxes(gridcolor="#1E1E1E", tickformat="%Y")
    fig.update_yaxes(gridcolor="#1E1E1E", tickprefix="$", row=1, col=1)
    if volume_col:
        fig.update_yaxes(gridcolor="#1E1E1E", row=2, col=1)
 
    if save:
        out = os.path.join(OUTPUT_DIR, "gold_daily.html")
        fig.write_html(out)
        print(f"[Plotly] Or quotidien → {out}")
 
    fig.show()

def check_monte_carlo_price_summary_dataframe(
    simulated_price_summary_df,
    require_median=True
):
    """
    Vérifie que le DataFrame résumé Monte Carlo des prix contient bien
    les colonnes attendues.

    Paramètres
    ----------
    simulated_price_summary_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - gold_price_actual
        - gold_price_sim_mean
        - gold_price_sim_q05
        - gold_price_sim_q95
        - éventuellement gold_price_sim_median
    require_median : bool
        Si True, impose la présence de la colonne `gold_price_sim_median`.
    """
    required_columns = [
        "date",
        "gold_price_actual",
        "gold_price_sim_mean",
        "gold_price_sim_q05",
        "gold_price_sim_q95"
    ]

    if require_median:
        required_columns.append("gold_price_sim_median")

    missing_cols = [col for col in required_columns if col not in simulated_price_summary_df.columns]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans simulated_price_summary_df : {missing_cols}. "
            f"Colonnes disponibles : {list(simulated_price_summary_df.columns)}"
        )


def plot_gold_price_monte_carlo_test(
    simulated_price_summary_df,
    figsize=(14, 6),
    show_median=True,
    show_band=True,
    major="2year",
    date_format="%Y"
):
    """
    Trace sur la période de test :
    - le prix observé de l'or
    - la moyenne des prix simulés
    - éventuellement la médiane des prix simulés
    - éventuellement une bande d'incertitude entre q05 et q95

    Paramètres
    ----------
    simulated_price_summary_df : pd.DataFrame
        Doit contenir au minimum :
        - date
        - gold_price_actual
        - gold_price_sim_mean
        - gold_price_sim_q05
        - gold_price_sim_q95
        - éventuellement gold_price_sim_median
    figsize : tuple
        Taille de la figure.
    show_median : bool
        Si True, trace la médiane des prix simulés.
    show_band : bool
        Si True, trace la bande d'incertitude [q05, q95].
    major : str
        Fréquence des ticks majeurs de l'axe des dates.
    date_format : str
        Format d'affichage des dates.
    """
    check_monte_carlo_price_summary_dataframe(
        simulated_price_summary_df,
        require_median=show_median
    )

    df = simulated_price_summary_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [
        "gold_price_actual",
        "gold_price_sim_mean",
        "gold_price_sim_q05",
        "gold_price_sim_q95"
    ]

    if show_median:
        numeric_columns.append("gold_price_sim_median")

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Des valeurs manquantes ont été détectées dans les colonnes "
            "utilisées pour le graphique Monte Carlo."
        )

    if show_band and (df["gold_price_sim_q05"] > df["gold_price_sim_q95"]).any():
        raise ValueError(
            "Certaines valeurs de gold_price_sim_q05 sont supérieures à gold_price_sim_q95."
        )

    df = df.sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------------
    # Bande d'incertitude Monte Carlo
    # --------------------------------------------------------
    if show_band:
        ax.fill_between(
            df["date"],
            df["gold_price_sim_q05"],
            df["gold_price_sim_q95"],
            color="#D4AF37",
            alpha=0.20,
            label="Bande Monte Carlo 5% - 95%"
        )

    # --------------------------------------------------------
    # Séries principales
    # --------------------------------------------------------
    ax.plot(
        df["date"],
        df["gold_price_actual"],
        label="Prix observé de l'or",
        color="black",
        linewidth=2
    )

    ax.plot(
        df["date"],
        df["gold_price_sim_mean"],
        label="Moyenne des prix simulés",
        color="#D4AF37",
        linestyle="--",
        linewidth=2
    )

    if show_median:
        ax.plot(
            df["date"],
            df["gold_price_sim_median"],
            label="Médiane des prix simulés",
            color="#4A90D9",
            linestyle=":",
            linewidth=2
        )

    ax.set_title("Prix observé vs simulation Monte Carlo sur le test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    ax.legend()
    ax.grid(True)

    format_date_axis(ax, major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()

