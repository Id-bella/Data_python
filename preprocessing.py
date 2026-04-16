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