import os
from glob import glob
import pandas as pd
import numpy as np
# ============================================================
# 1) Chargement des fichiers déjà téléchargés
# ============================================================
def load_market_data(data_dir="data"):
    """
    Charge les fichiers CSV déjà téléchargés pour les séries de marché.

    Ici on lit les CSV Yahoo sans supposer qu'ils sont déjà propres,
    car certains exports contiennent plusieurs lignes d'en-tête.
    """
    data = {
        "gold": pd.read_csv(f"{data_dir}/gold.csv", header=None),
        "dxy": pd.read_csv(f"{data_dir}/dxy.csv", header=None),
        "sp500": pd.read_csv(f"{data_dir}/sp500.csv", header=None),
        "vix": pd.read_csv(f"{data_dir}/vix.csv", header=None),
    }
    return data


# ============================================================
# 2) Nettoyage d'un DataFrame Yahoo Finance
# ============================================================
def clean_yahoo_data(df, series_name):
    """
    Nettoie un DataFrame Yahoo Finance exporté en CSV avec en-têtes parasites.

    Format observé :
    ligne 0 : Price, Adj Close, Close, ...
    ligne 1 : Ticker, DX-Y.NYB, DX-Y.NYB, ...
    ligne 2 : Date, ...
    ligne 3+ : données

    Retour :
    - date
    - <series_name>_price
    """
    df = df.copy()

    # --------------------------------------------------------
    # La première ligne contient les vrais noms de colonnes utiles
    # --------------------------------------------------------
    header = df.iloc[0].tolist()
    df.columns = header

    # --------------------------------------------------------
    # On enlève les 3 premières lignes parasites :
    # 0 = noms
    # 1 = Ticker
    # 2 = Date
    # --------------------------------------------------------
    df = df.iloc[3:].copy()

    # --------------------------------------------------------
    # La première colonne doit devenir la date
    # --------------------------------------------------------
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})

    # --------------------------------------------------------
    # Conversion de la date
    # --------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --------------------------------------------------------
    # On supprime les lignes invalides éventuelles
    # --------------------------------------------------------
    df = df.dropna(subset=["date"]).copy()

    # --------------------------------------------------------
    # On choisit Adj Close si disponible, sinon Close
    # --------------------------------------------------------
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"Aucune colonne de prix trouvée pour {series_name}.")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df[["date", price_col]].copy()
    df = df.rename(columns={price_col: f"{series_name}_price"})

    df = df.sort_values("date").reset_index(drop=True)

    return df

# ============================================================
# 3) Application du nettoyage à toutes les séries
# ============================================================
def clean_all_market_data(raw_data):
    """
    Nettoie toutes les séries de marché.

    Paramètres
    ----------
    raw_data : dict
        Dictionnaire de DataFrames bruts.

    Retour
    ------
    dict
        Dictionnaire de DataFrames nettoyés.
    """
    cleaned = {}
    for name, df in raw_data.items():
        cleaned[name] = clean_yahoo_data(df, name)
    return cleaned


# ============================================================
# 4) Calcul des log-rendements
# ============================================================
def add_log_returns(df, series_name):
    """
    Ajoute une colonne de log-rendement à partir d'une colonne de prix.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant :
        - date
        - <series_name>_price
    series_name : str
        Nom de la série

    Retour
    ------
    pd.DataFrame
        DataFrame avec une colonne supplémentaire :
        - <series_name>_ret
    """
    df = df.copy()

    price_col = f"{series_name}_price"
    ret_col = f"{series_name}_ret"

    df[ret_col] = np.log(df[price_col] / df[price_col].shift(1))

    return df


# ============================================================
# 5) Construction des séries transformées pour le VAR
# ============================================================
def build_var_series(cleaned_data):
    """
    Construit les séries transformées pour le VAR.

    Convention retenue :
    - gold   -> log-rendement
    - dxy    -> log-rendement
    - sp500  -> log-rendement
    - vix    -> log-différence

    Paramètres
    ----------
    cleaned_data : dict
        Dictionnaire des DataFrames nettoyés.

    Retour
    ------
    dict
        Dictionnaire contenant les DataFrames transformés.
    """
    transformed = {}

    for name, df in cleaned_data.items():
        transformed[name] = add_log_returns(df, name)

    return transformed


# ============================================================
# 6) Fusion de toutes les séries sur la date
# ============================================================
def merge_var_series(transformed_data):
    """
    Fusionne les séries transformées sur la date.

    Paramètres
    ----------
    transformed_data : dict
        Dictionnaire de DataFrames transformés.

    Retour
    ------
    pd.DataFrame
        DataFrame fusionné avec :
        - date
        - gold_ret
        - dxy_ret
        - sp500_ret
        - vix_ret
    """
    gold = transformed_data["gold"][["date", "gold_ret"]].copy()
    dxy = transformed_data["dxy"][["date", "dxy_ret"]].copy()
    sp500 = transformed_data["sp500"][["date", "sp500_ret"]].copy()
    vix = transformed_data["vix"][["date", "vix_ret"]].copy()

    df = gold.merge(dxy, on="date", how="inner")
    df = df.merge(sp500, on="date", how="inner")
    df = df.merge(vix, on="date", how="inner")

    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# 7) Suppression des NA créés par les rendements
# ============================================================
def drop_missing_var_rows(df):
    """
    Supprime les lignes incomplètes du dataset VAR.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame fusionné.

    Retour
    ------
    pd.DataFrame
        DataFrame sans NA.
    """
    return df.dropna().reset_index(drop=True)


# ============================================================
# 8) Pipeline complet de preprocessing
# ============================================================
def prepare_var_dataset(data_dir="data"):
    """
    Pipeline complet de preprocessing pour le VAR.

    Étapes :
    1. chargement des CSV
    2. nettoyage
    3. calcul des log-rendements / log-différences
    4. fusion
    5. suppression des NA

    Paramètres
    ----------
    data_dir : str
        Dossier contenant les fichiers téléchargés.

    Retour
    ------
    pd.DataFrame
        Dataset final prêt pour le VAR.
    """
    raw_data = load_market_data(data_dir=data_dir)
    cleaned_data = clean_all_market_data(raw_data)
    transformed_data = build_var_series(cleaned_data)
    var_df = merge_var_series(transformed_data)
    var_df = drop_missing_var_rows(var_df)

    return var_df

def prepare_daily_macro_exog(data_dir="data", daily_calendar_df=None):
    """
    Prépare les exogènes macro journalières :
    - gpr_level
    - cpi_mom

    Hypothèses :
    - cpi.csv contient : date, cpi
    - gpr_raw.xls(x) contient : month, GPR
    - daily_calendar_df contient une colonne date journalière
    """
    if daily_calendar_df is None:
        raise ValueError("Il faut fournir daily_calendar_df avec une colonne 'date'.")

    # --------------------------------------------------------
    # 1) Calendrier journalier
    # --------------------------------------------------------
    daily_df = daily_calendar_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)

    # Clé mensuelle = début de mois
    daily_df["month"] = daily_df["date"].dt.to_period("M").dt.to_timestamp()

    # --------------------------------------------------------
    # 2) CPI
    # --------------------------------------------------------
    cpi_path = os.path.join(data_dir, "cpi.csv")
    cpi_df = pd.read_csv(cpi_path)

    cpi_df = cpi_df[["date", "cpi"]].copy()
    cpi_df["date"] = pd.to_datetime(cpi_df["date"])
    cpi_df["cpi"] = pd.to_numeric(cpi_df["cpi"], errors="coerce")

    cpi_df = cpi_df.dropna(subset=["date", "cpi"]).sort_values("date").reset_index(drop=True)

    # Variation mensuelle du CPI
    cpi_df["cpi_mom"] = cpi_df["cpi"].pct_change()

    # Clé mensuelle
    cpi_df["month"] = cpi_df["date"].dt.to_period("M").dt.to_timestamp()

    cpi_df = cpi_df[["month", "cpi_mom"]].copy()

    # --------------------------------------------------------
    # 3) GPR
    # --------------------------------------------------------
    gpr_candidates = sorted(glob(os.path.join(data_dir, "gpr_raw.xls*")))
    if not gpr_candidates:
        raise FileNotFoundError("Aucun fichier GPR trouvé dans le dossier data.")

    gpr_df = pd.read_excel(gpr_candidates[0])

    gpr_df = gpr_df[["month", "GPR"]].copy()
    gpr_df["month"] = pd.to_datetime(gpr_df["month"])
    gpr_df["GPR"] = pd.to_numeric(gpr_df["GPR"], errors="coerce")

    gpr_df = gpr_df.dropna(subset=["month", "GPR"]).sort_values("month").reset_index(drop=True)
    gpr_df = gpr_df.rename(columns={"GPR": "gpr_level"})

    gpr_df["month"] = gpr_df["month"].dt.to_period("M").dt.to_timestamp()

    gpr_df = gpr_df[["month", "gpr_level"]].copy()

    # --------------------------------------------------------
    # 4) Merge journalier
    # --------------------------------------------------------
    exog_df = daily_df.merge(gpr_df, on="month", how="left")
    exog_df = exog_df.merge(cpi_df, on="month", how="left")

    exog_df = exog_df[["date", "gpr_level", "cpi_mom"]].copy()
    exog_df = exog_df.sort_values("date").reset_index(drop=True)

    return exog_df


# ============================================================
# 15) Split train / test des exogènes macro par date
# ============================================================
def split_macro_exog_train_test_by_date(
    macro_exog_df,
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2025-12-31"
):
    """
    Sépare explicitement les exogènes macro journalières en train et test.

    Paramètres
    ----------
    macro_exog_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - gpr_level
        - cpi_mom
    train_start : str
    train_end : str
    test_start : str
    test_end : str

    Retour
    ------
    train_exog_df : pd.DataFrame
    test_exog_df : pd.DataFrame
    """
    required_columns = ["date", "gpr_level", "cpi_mom"]
    missing_cols = [col for col in required_columns if col not in macro_exog_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans macro_exog_df : {missing_cols}. "
            f"Colonnes disponibles : {list(macro_exog_df.columns)}"
        )

    df = macro_exog_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    if test_start <= train_end:
        raise ValueError("La date de début du test doit être strictement postérieure à la fin du train.")
    if train_start > train_end:
        raise ValueError("train_start doit être antérieure ou égale à train_end.")
    if test_start > test_end:
        raise ValueError("test_start doit être antérieure ou égale à test_end.")

    train_exog_df = df.loc[
        (df["date"] >= train_start) & (df["date"] <= train_end)
    ].copy()

    test_exog_df = df.loc[
        (df["date"] >= test_start) & (df["date"] <= test_end)
    ].copy()

    if train_exog_df.empty:
        raise ValueError("Le sous-échantillon train des exogènes est vide.")
    if test_exog_df.empty:
        raise ValueError("Le sous-échantillon test des exogènes est vide.")

    return train_exog_df.reset_index(drop=True), test_exog_df.reset_index(drop=True)