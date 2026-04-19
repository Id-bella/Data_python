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