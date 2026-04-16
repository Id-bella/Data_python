import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


# ============================================================
# 1) Vérification minimale du DataFrame VAR
# ============================================================
def check_var_dataframe(var_df):
    """
    Vérifie que le DataFrame contient bien au moins les colonnes utiles.

    Paramètres
    ----------
    var_df : pd.DataFrame
        DataFrame contenant les séries du VAR.

    Retour
    ------
    None
    """
    required_columns = ["date", "gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]

    missing_cols = [col for col in required_columns if col not in var_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans var_df : {missing_cols}. "
            f"Colonnes disponibles : {list(var_df.columns)}"
        )


# ============================================================
# 2) Préparation des données pour le VAR
# ============================================================
def prepare_var_model_data(var_df, variables=None):
    """
    Prépare le DataFrame pour l'estimation du VAR.

    Étapes :
    - vérification des colonnes
    - conversion de la date
    - tri chronologique
    - mise en index de la date
    - conservation des seules variables choisies

    Paramètres
    ----------
    var_df : pd.DataFrame
    variables : list or None
        Liste des variables à garder dans le VAR.
        Si None, on prend le système complet :
        ["gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]

    Retour
    ------
    pd.DataFrame
        DataFrame indexé par date, prêt pour VAR.
    """
    check_var_dataframe(var_df)

    if variables is None:
        variables = ["gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]

    missing_vars = [col for col in variables if col not in var_df.columns]
    if missing_vars:
        raise ValueError(
            f"Variables demandées absentes du DataFrame : {missing_vars}. "
            f"Colonnes disponibles : {list(var_df.columns)}"
        )

    df = var_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    df = df[variables].copy()
    df = df.dropna()

    return df


# ============================================================
# 3) Construction des fenêtres glissantes
# ============================================================
def build_rolling_windows(model_df, window_years=7, step_years=2):
    """
    Construit les fenêtres glissantes en années.
    """
    if not isinstance(model_df.index, pd.DatetimeIndex):
        raise ValueError("Le DataFrame doit avoir un DatetimeIndex.")

    windows = []

    global_start = model_df.index.min()
    global_end = model_df.index.max()

    current_start = global_start

    while current_start + pd.DateOffset(years=window_years) <= global_end:
        current_end = current_start + pd.DateOffset(years=window_years)

        window_df = model_df.loc[
            (model_df.index >= current_start) & (model_df.index < current_end)
        ].copy()

        windows.append((current_start, current_end, window_df))

        current_start = current_start + pd.DateOffset(years=step_years)

    return windows


# ============================================================
# 4) Estimation des VAR(p) sur une fenêtre donnée
# ============================================================
def estimate_var_candidates_on_window(window_df, max_lag=10):
    """
    Estime VAR(1) à VAR(max_lag) sur une fenêtre donnée
    et récupère AIC et BIC.
    """
    results = []

    if window_df.shape[0] == 0:
        return pd.DataFrame(columns=["lag", "aic", "bic", "n_obs"])

    model = VAR(window_df)

    for lag in range(1, max_lag + 1):
        try:
            fitted = model.fit(lag)

            results.append({
                "lag": lag,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "n_obs": fitted.nobs
            })

        except Exception:
            results.append({
                "lag": lag,
                "aic": np.nan,
                "bic": np.nan,
                "n_obs": np.nan
            })

    return pd.DataFrame(results)


# ============================================================
# 5) Meilleurs lags AIC et BIC sur une fenêtre
# ============================================================
def get_best_lags_for_window(criteria_df):
    """
    Identifie, sur une fenêtre donnée, le meilleur lag selon AIC et BIC.
    """
    clean_df = criteria_df.dropna(subset=["aic", "bic"]).copy()

    if clean_df.empty:
        return {
            "best_lag_aic": np.nan,
            "best_aic": np.nan,
            "best_lag_bic": np.nan,
            "best_bic": np.nan
        }

    best_aic_row = clean_df.loc[clean_df["aic"].idxmin()]
    best_bic_row = clean_df.loc[clean_df["bic"].idxmin()]

    return {
        "best_lag_aic": int(best_aic_row["lag"]),
        "best_aic": float(best_aic_row["aic"]),
        "best_lag_bic": int(best_bic_row["lag"]),
        "best_bic": float(best_bic_row["bic"])
    }


# ============================================================
# 6) Procédure complète de sélection du lag en rolling window
# ============================================================
def rolling_var_lag_selection(var_df, window_years=7, step_years=2, max_lag=10, variables=None):
    """
    Applique la procédure complète de sélection du lag du VAR
    sur fenêtres glissantes.

    Paramètres
    ----------
    var_df : pd.DataFrame
    window_years : int
    step_years : int
    max_lag : int
    variables : list or None
        Variables retenues dans le VAR.
    """
    model_df = prepare_var_model_data(var_df, variables=variables)
    windows = build_rolling_windows(model_df, window_years=window_years, step_years=step_years)

    summary_rows = []
    criteria_dict = {}

    for i, (window_start, window_end, window_data) in enumerate(windows, start=1):
        criteria_df = estimate_var_candidates_on_window(window_data, max_lag=max_lag)
        best_lags = get_best_lags_for_window(criteria_df)

        window_name = f"window_{i}"
        criteria_dict[window_name] = criteria_df.copy()

        summary_rows.append({
            "window": window_name,
            "start_date": window_start,
            "end_date": window_end,
            "n_obs": window_data.shape[0],
            "variables": ", ".join(window_data.columns),
            "best_lag_aic": best_lags["best_lag_aic"],
            "best_aic": best_lags["best_aic"],
            "best_lag_bic": best_lags["best_lag_bic"],
            "best_bic": best_lags["best_bic"]
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, criteria_dict


# ============================================================
# 7) Choix final du lag par moyenne des meilleurs lags
# ============================================================
def choose_final_lag_from_rolling(summary_df):
    """
    Choisit le lag final selon la règle de moyenne des meilleurs lags.
    """
    aic_lags = summary_df["best_lag_aic"].dropna().tolist()
    bic_lags = summary_df["best_lag_bic"].dropna().tolist()

    selected_lags = aic_lags + bic_lags

    if len(selected_lags) == 0:
        raise ValueError("Aucun lag sélectionné n'a pu être récupéré.")

    mean_selected_lag = float(np.mean(selected_lags))
    final_lag = int(np.round(mean_selected_lag))

    return {
        "selected_lags": selected_lags,
        "mean_selected_lag": mean_selected_lag,
        "final_lag": final_lag
    }


# ============================================================
# 8) Tableau de fréquence des lags sélectionnés
# ============================================================
def lag_selection_frequency_table(summary_df):
    """
    Construit un tableau de fréquence des lags sélectionnés
    par AIC et BIC.
    """
    aic_lags = summary_df["best_lag_aic"].dropna().astype(int).tolist()
    bic_lags = summary_df["best_lag_bic"].dropna().astype(int).tolist()

    all_lags = pd.Series(aic_lags + bic_lags, name="selected_lag")

    freq_table = all_lags.value_counts().sort_index().reset_index()
    freq_table.columns = ["lag", "frequency"]

    return freq_table


# ============================================================
# 9) Impression simple du choix final
# ============================================================
def print_final_lag_selection(final_lag_dict):
    """
    Affiche un résumé lisible du lag final retenu.
    """
    print("Lags sélectionnés sur l'ensemble des fenêtres (AIC + BIC) :")
    print(final_lag_dict["selected_lags"])
    print()
    print(f"Moyenne des lags sélectionnés : {final_lag_dict['mean_selected_lag']:.4f}")
    print(f"Lag final retenu (arrondi) : {final_lag_dict['final_lag']}")


# ============================================================
# 10) Estimation d'un VAR avec un lag fixé
# ============================================================
def fit_var_model(var_df, lag_order, variables=None):
    """
    Estime un VAR avec un nombre de retards fixé.

    Paramètres
    ----------
    var_df : pd.DataFrame
    lag_order : int
        Nombre de retards du VAR.
    variables : list or None
        Variables retenues dans le VAR.

    Retour
    ------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.
    """
    model_df = prepare_var_model_data(var_df, variables=variables)

    model = VAR(model_df)
    fitted_model = model.fit(lag_order)

    return fitted_model


# ============================================================
# 11) Résumé textuel du modèle VAR
# ============================================================
def get_var_summary(fitted_model):
    """
    Renvoie le résumé textuel complet du VAR estimé.
    """
    return fitted_model.summary()


# ============================================================
# 12) Tableau récapitulatif simple du VAR estimé
# ============================================================
def get_var_model_info(fitted_model):
    """
    Renvoie un petit tableau récapitulatif du modèle estimé.
    """
    info = {
        "lag_order": fitted_model.k_ar,
        "n_obs": fitted_model.nobs,
        "n_equations": fitted_model.neqs,
        "aic": fitted_model.aic,
        "bic": fitted_model.bic,
        "hqic": fitted_model.hqic,
        "fpe": fitted_model.fpe,
        "variables": fitted_model.names
    }

    return info


# ============================================================
# 13) Valeurs ajustées du VAR
# ============================================================
def get_var_fitted_values(fitted_model):
    """
    Renvoie les valeurs ajustées du modèle VAR.
    """
    return fitted_model.fittedvalues.copy()


# ============================================================
# 14) Résidus du VAR
# ============================================================
def get_var_residuals(fitted_model):
    """
    Renvoie les résidus du modèle VAR.
    """
    return fitted_model.resid.copy()


# ============================================================
# 15) Coefficients du VAR par équation
# ============================================================
def get_var_parameters(fitted_model):
    """
    Renvoie les coefficients estimés du modèle VAR.
    """
    return fitted_model.params.copy()


# ============================================================
# 16) Fonction pratique pour estimer directement un VAR(3)
# ============================================================
def fit_var3(var_df):
    """
    Estime directement un VAR(3) complet :
    gold, dxy, sp500, vix
    """
    return fit_var_model(
        var_df,
        lag_order=3,
        variables=["gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]
    )


# ============================================================
# 17) Fonction pratique pour estimer un VAR(3) sans le S&P 500
# ============================================================
def fit_var3_without_sp500(var_df):
    """
    Estime un VAR(3) sur :
    gold, dxy, vix
    """
    return fit_var_model(
        var_df,
        lag_order=3,
        variables=["gold_ret", "dxy_ret", "vix_ret"]
    )