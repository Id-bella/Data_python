import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.optimize import minimize
from scipy.special import gammaln

FULL_VAR_VARIABLES = ["gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]
REDUCED_VAR_VARIABLES = ["gold_ret", "dxy_ret", "vix_ret"]


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
        variables = FULL_VAR_VARIABLES

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
    """
    model_df = prepare_var_model_data(var_df, variables=variables)
    windows = build_rolling_windows(
        model_df,
        window_years=window_years,
        step_years=step_years
    )

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
# 8) Impression simple du choix final
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
# 9) Résumé textuel du modèle VAR
# ============================================================
def get_var_summary(fitted_model):
    """
    Renvoie le résumé textuel complet du VAR estimé.
    """
    return fitted_model.summary()


# ============================================================
# 10) Tableau récapitulatif simple du VAR estimé
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
# 11) Split train / test par date
# ============================================================
def split_var_train_test_by_date(
    var_df,
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2025-12-31"
):
    """
    Sépare explicitement le dataset VAR en train et test selon la date.

    Convention retenue :
    - train : de train_start à train_end inclus
    - test  : de test_start à test_end inclus
    """
    check_var_dataframe(var_df)

    df = var_df.copy()
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

    train_df = df.loc[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    test_df = df.loc[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    if train_df.empty:
        raise ValueError("Le sous-échantillon train est vide.")
    if test_df.empty:
        raise ValueError("Le sous-échantillon test est vide.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# 12) Sélection du lag uniquement sur le train
# ============================================================
def select_var_lag_on_train(
    train_df,
    variables,
    window_years=7,
    step_years=2,
    max_lag=10
):
    """
    Applique la procédure de sélection du lag uniquement sur le train.
    """
    summary_df, criteria_dict = rolling_var_lag_selection(
        train_df,
        window_years=window_years,
        step_years=step_years,
        max_lag=max_lag,
        variables=variables
    )

    if summary_df.empty:
        raise ValueError("La sélection de lag sur le train n'a produit aucune fenêtre exploitable.")

    final_lag_dict = choose_final_lag_from_rolling(summary_df)

    return summary_df, criteria_dict, final_lag_dict


# ============================================================
# 13) Estimation finale du VAR sur le train
# ============================================================
def fit_var_on_train(train_df, lag_order, variables):
    """
    Estime le VAR final uniquement sur le train.
    """
    if lag_order is None or int(lag_order) < 1:
        raise ValueError("lag_order doit être un entier >= 1.")

    train_model_df = prepare_var_model_data(train_df, variables=variables)

    if train_model_df.shape[0] <= int(lag_order):
        raise ValueError("Le train ne contient pas assez d'observations pour estimer ce VAR.")

    model = VAR(train_model_df)
    fitted_model = model.fit(int(lag_order))

    return fitted_model


# ============================================================
# 14) Prévisions one-step-ahead sur le test
# ============================================================
def forecast_var_one_step_ahead(fitted_model, train_df, test_df, variables):
    """
    Produit des prévisions one-step-ahead sur le test.

    Logique :
    - le VAR est estimé une seule fois sur le train ;
    - à chaque date du test, on prédit t avec les infos disponibles jusqu'à t-1 ;
    - après la prévision, on ajoute l'observation réelle à l'historique.

    Cette sortie est utile pour juger la qualité de prévision des rendements.
    """
    if fitted_model is None:
        raise ValueError("fitted_model ne peut pas être None.")

    model_variables = list(fitted_model.names)
    if model_variables != variables:
        raise ValueError(
            f"Variables du modèle : {model_variables}. Variables attendues : {variables}."
        )

    lag_order = int(fitted_model.k_ar)
    if lag_order < 1:
        raise ValueError("Le modèle VAR doit avoir au moins un retard.")

    train_model_df = prepare_var_model_data(train_df, variables=variables)
    test_model_df = prepare_var_model_data(test_df, variables=variables)

    if train_model_df.shape[0] < lag_order:
        raise ValueError("Le train préparé ne contient pas assez d'observations pour lancer les prévisions.")

    history_df = train_model_df.copy()
    forecast_rows = []

    for current_date, actual_row in test_model_df.iterrows():
        lagged_values = history_df.values[-lag_order:]
        predicted_values = fitted_model.forecast(y=lagged_values, steps=1)[0]

        row_dict = {"date": current_date}

        for idx, var_name in enumerate(variables):
            row_dict[f"{var_name}_actual"] = float(actual_row[var_name])
            row_dict[f"{var_name}_pred"] = float(predicted_values[idx])

        forecast_rows.append(row_dict)

        history_df.loc[current_date, variables] = actual_row[variables].values

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return forecast_df


# ============================================================
# 15) Prévisions dynamiques sur le test
# ============================================================
def forecast_var_dynamically_on_test(fitted_model, train_df, test_df, variables):
    """
    Produit des prévisions dynamiques sur le test.

    Logique :
    - le VAR est estimé sur le train ;
    - on part des dernières observations du train ;
    - à chaque date du test, on prédit t ;
    - puis on réinjecte la prédiction dans l'historique,
      et non la vraie observation.

    Cela donne une vraie trajectoire prédite autonome sur le test.
    """
    if fitted_model is None:
        raise ValueError("fitted_model ne peut pas être None.")

    model_variables = list(fitted_model.names)
    if model_variables != variables:
        raise ValueError(
            f"Variables du modèle : {model_variables}. Variables attendues : {variables}."
        )

    lag_order = int(fitted_model.k_ar)
    if lag_order < 1:
        raise ValueError("Le modèle VAR doit avoir au moins un retard.")

    train_model_df = prepare_var_model_data(train_df, variables=variables)
    test_model_df = prepare_var_model_data(test_df, variables=variables)

    if train_model_df.shape[0] < lag_order:
        raise ValueError("Le train préparé ne contient pas assez d'observations pour lancer les prévisions.")

    history_df = train_model_df.copy()
    forecast_rows = []

    for current_date, actual_row in test_model_df.iterrows():
        lagged_values = history_df.values[-lag_order:]
        predicted_values = fitted_model.forecast(y=lagged_values, steps=1)[0]

        row_dict = {"date": current_date}

        for idx, var_name in enumerate(variables):
            row_dict[f"{var_name}_actual"] = float(actual_row[var_name])
            row_dict[f"{var_name}_pred"] = float(predicted_values[idx])

        forecast_rows.append(row_dict)

        history_df.loc[current_date, variables] = predicted_values

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    return forecast_df


# ============================================================
# 16) Outils génériques de reconstruction de prix
# ============================================================
def reconstruct_price_path_from_returns(initial_price, predicted_returns, return_type="log"):
    """
    Reconstructs a dynamic price path from a sequence of predicted returns.

    Parameters
    ----------
    initial_price : float
        Last observed price before the forecast period.
    predicted_returns : array-like
        Sequence of predicted returns.
    return_type : str
        - "log"    : P_t = P_{t-1} * exp(r_t)
        - "simple" : P_t = P_{t-1} * (1 + r_t)

    Returns
    -------
    np.ndarray
        Dynamic predicted price path.
    """
    if initial_price is None or not np.isfinite(initial_price) or float(initial_price) <= 0:
        raise ValueError("initial_price doit être un nombre fini strictement positif.")

    predicted_returns = pd.Series(predicted_returns).astype(float)

    if predicted_returns.isna().any():
        raise ValueError("predicted_returns contient des valeurs manquantes.")

    if return_type not in ["log", "simple"]:
        raise ValueError("return_type doit valoir 'log' ou 'simple'.")

    price_path = []
    current_price = float(initial_price)

    for r_t in predicted_returns:
        if return_type == "log":
            current_price = current_price * np.exp(r_t)
        else:
            current_price = current_price * (1 + r_t)

        price_path.append(float(current_price))

    return np.array(price_path, dtype=float)


def build_dynamic_price_forecast(
    price_df,
    forecast_df,
    price_col="gold_price",
    pred_return_col="gold_ret_pred",
    actual_return_col=None,
    pred_vol_col=None,
    date_col="date",
    return_type="log"
):
    """
    Builds a clean dynamic forecast DataFrame from predicted returns.

    The function is model-agnostic: any forecast DataFrame can be used as
    long as it contains a date column and a predicted return column.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame containing the observed prices.
    forecast_df : pd.DataFrame
        DataFrame containing predicted returns and optional extra columns.
    price_col : str
        Name of the observed price column in `price_df`.
    pred_return_col : str
        Name of the predicted return column in `forecast_df`.
    actual_return_col : str or None
        Optional name of the actual return column in `forecast_df`.
    pred_vol_col : str or None
        Optional name of the predicted volatility column in `forecast_df`.
    date_col : str
        Name of the date column in both DataFrames.
    return_type : str
        - "log"
        - "simple"

    Returns
    -------
    pd.DataFrame
        Clean forecast DataFrame with actual and predicted prices.
    """
    required_price_cols = [date_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in price_df.columns]
    if missing_price_cols:
        raise ValueError(
            f"Colonnes manquantes dans price_df : {missing_price_cols}. "
            f"Colonnes disponibles : {list(price_df.columns)}"
        )

    required_forecast_cols = [date_col, pred_return_col]
    missing_forecast_cols = [col for col in required_forecast_cols if col not in forecast_df.columns]
    if missing_forecast_cols:
        raise ValueError(
            f"Colonnes manquantes dans forecast_df : {missing_forecast_cols}. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )

    if actual_return_col is not None and actual_return_col not in forecast_df.columns:
        raise ValueError(
            f"La colonne '{actual_return_col}' est absente de forecast_df. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )

    if pred_vol_col is not None and pred_vol_col not in forecast_df.columns:
        raise ValueError(
            f"La colonne '{pred_vol_col}' est absente de forecast_df. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )

    price_df = price_df.copy()
    forecast_df = forecast_df.copy()

    price_df[date_col] = pd.to_datetime(price_df[date_col])
    forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])

    price_df = price_df.sort_values(date_col).reset_index(drop=True)
    forecast_df = forecast_df.sort_values(date_col).reset_index(drop=True)

    merge_columns = [date_col, pred_return_col]

    if actual_return_col is not None:
        merge_columns.append(actual_return_col)

    if pred_vol_col is not None:
        merge_columns.append(pred_vol_col)

    merged_df = forecast_df[merge_columns].merge(
        price_df[[date_col, price_col]],
        on=date_col,
        how="inner"
    ).sort_values(date_col).reset_index(drop=True)

    if merged_df.empty:
        raise ValueError("Aucune date commune entre price_df et forecast_df pour reconstruire les prix.")

    first_forecast_date = merged_df.loc[0, date_col]

    initial_price_df = price_df.loc[
        price_df[date_col] < first_forecast_date,
        [date_col, price_col]
    ]

    if initial_price_df.empty:
        raise ValueError("Impossible de trouver le dernier prix observé avant le début de la prévision.")

    initial_price = float(initial_price_df.iloc[-1][price_col])

    merged_df["predicted_price"] = reconstruct_price_path_from_returns(
        initial_price=initial_price,
        predicted_returns=merged_df[pred_return_col],
        return_type=return_type
    )

    merged_df = merged_df.rename(columns={price_col: "actual_price"})

    output_columns = [date_col, pred_return_col]

    if actual_return_col is not None:
        output_columns.append(actual_return_col)

    if pred_vol_col is not None:
        output_columns.append(pred_vol_col)

    output_columns.extend(["actual_price", "predicted_price"])

    return merged_df[output_columns].copy()


# ============================================================
# 17) Reconstruction one-step-ahead des prix de l'or sur le test
# ============================================================
def reconstruct_gold_test_prices_one_step(price_df, forecast_df):
    """
    Reconstruit le prix prédit de l'or sur le test dans un cadre one-step-ahead.

    À la date t :
        P_t_pred = P_{t-1, actual} * exp(gold_ret_pred_t)
    """
    required_price_cols = ["date", "gold_price"]
    missing_price_cols = [col for col in required_price_cols if col not in price_df.columns]
    if missing_price_cols:
        raise ValueError(
            f"Colonnes manquantes dans price_df : {missing_price_cols}. "
            f"Colonnes disponibles : {list(price_df.columns)}"
        )

    required_forecast_cols = ["date", "gold_ret_pred"]
    missing_forecast_cols = [col for col in required_forecast_cols if col not in forecast_df.columns]
    if missing_forecast_cols:
        raise ValueError(
            f"Colonnes manquantes dans forecast_df : {missing_forecast_cols}. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )

    price_df = price_df.copy()
    forecast_df = forecast_df.copy()

    price_df["date"] = pd.to_datetime(price_df["date"])
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    price_df = price_df.sort_values("date").reset_index(drop=True)
    forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

    price_df["gold_price_lag1_actual"] = price_df["gold_price"].shift(1)

    merged_df = forecast_df.merge(
        price_df[["date", "gold_price", "gold_price_lag1_actual"]],
        on="date",
        how="inner"
    ).sort_values("date").reset_index(drop=True)

    if merged_df.empty:
        raise ValueError("Aucune date commune entre price_df et forecast_df pour reconstruire les prix.")

    if merged_df["gold_price_lag1_actual"].isna().any():
        raise ValueError(
            "Certaines dates du test n'ont pas de prix observé à t-1. "
            "Vérifie l'alignement des dates dans price_df."
        )

    merged_df["gold_price_pred"] = (
        merged_df["gold_price_lag1_actual"] * np.exp(merged_df["gold_ret_pred"])
    )

    merged_df = merged_df.rename(columns={"gold_price": "gold_price_actual"})

    output_columns = ["date", "gold_ret_pred", "gold_price_actual", "gold_price_pred"]
    if "gold_ret_actual" in merged_df.columns:
        output_columns.insert(2, "gold_ret_actual")

    return merged_df[output_columns].copy()


# ============================================================
# 18) Reconstruction dynamique des prix de l'or sur le test
# ============================================================
def reconstruct_gold_test_prices_dynamic(price_df, forecast_df):
    """
    Reconstruit le prix prédit de l'or sur le test en suivant uniquement
    les rendements prédits depuis le début du test.

    Idée retenue, identique à ton code ARMA :
    - on récupère le dernier prix observé juste avant le test ;
    - on initialise une liste avec ce dernier prix ;
    - pour chaque rendement prédit, on calcule :
          nouveau_prix = dernier_prix_simulé * exp(rendement_prédit)
    - on enlève ensuite le prix initial pour ne garder que les prix simulés
      sur la période de test.
    """
    dynamic_price_df = build_dynamic_price_forecast(
        price_df=price_df,
        forecast_df=forecast_df,
        price_col="gold_price",
        pred_return_col="gold_ret_pred",
        actual_return_col="gold_ret_actual" if "gold_ret_actual" in forecast_df.columns else None,
        pred_vol_col=None,
        date_col="date",
        return_type="log"
    )

    dynamic_price_df = dynamic_price_df.rename(columns={
        "actual_price": "gold_price_actual",
        "predicted_price": "gold_price_pred"
    })

    return dynamic_price_df


# ============================================================
# 19) Pipeline complète VAR train/test
# ============================================================
def run_var_train_test_pipeline(
    var_df,
    price_df,
    variables,
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2025-12-31",
    window_years=7,
    step_years=2,
    max_lag=10,
    price_reconstruction_mode="dynamic"
):
    """
    Pipeline complète :
    - split train/test
    - sélection du lag sur le train
    - estimation du VAR final sur le train
    - prévisions one-step-ahead sur le test
    - prévisions dynamiques sur le test
    - reconstruction du prix prédit de l'or sur le test

    Paramètres
    ----------
    price_reconstruction_mode : str
        - "dynamic"  : vraie trajectoire autonome sur le test
        - "one_step" : prix prédit jour par jour à partir du vrai prix de la veille
    """
    train_df, test_df = split_var_train_test_by_date(
        var_df=var_df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    lag_summary_df, lag_criteria_dict, final_lag_dict = select_var_lag_on_train(
        train_df=train_df,
        variables=variables,
        window_years=window_years,
        step_years=step_years,
        max_lag=max_lag
    )

    final_lag = int(final_lag_dict["final_lag"])

    fitted_model = fit_var_on_train(
        train_df=train_df,
        lag_order=final_lag,
        variables=variables
    )

    forecast_df_one_step = forecast_var_one_step_ahead(
        fitted_model=fitted_model,
        train_df=train_df,
        test_df=test_df,
        variables=variables
    )

    forecast_df_dynamic = forecast_var_dynamically_on_test(
        fitted_model=fitted_model,
        train_df=train_df,
        test_df=test_df,
        variables=variables
    )

    if price_reconstruction_mode == "one_step":
        gold_price_forecast_df = reconstruct_gold_test_prices_one_step(
            price_df=price_df,
            forecast_df=forecast_df_one_step
        )
    elif price_reconstruction_mode == "dynamic":
        gold_price_forecast_df = reconstruct_gold_test_prices_dynamic(
            price_df=price_df,
            forecast_df=forecast_df_dynamic
        )
    else:
        raise ValueError("price_reconstruction_mode doit valoir 'dynamic' ou 'one_step'.")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "lag_summary_df": lag_summary_df,
        "lag_criteria_dict": lag_criteria_dict,
        "final_lag_dict": final_lag_dict,
        "final_lag": final_lag,
        "fitted_model": fitted_model,
        "forecast_df_one_step": forecast_df_one_step,
        "forecast_df_dynamic": forecast_df_dynamic,
        "gold_price_forecast_df": gold_price_forecast_df
    }


# ============================================================
# 20) Pipelines pratiques
# ============================================================
def run_full_var_train_test_pipeline(
    var_df,
    price_df,
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2025-12-31",
    window_years=7,
    step_years=2,
    max_lag=10,
    price_reconstruction_mode="dynamic"
):
    """
    Pipeline VAR train/test avec S&P 500.
    """
    return run_var_train_test_pipeline(
        var_df=var_df,
        price_df=price_df,
        variables=FULL_VAR_VARIABLES,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        window_years=window_years,
        step_years=step_years,
        max_lag=max_lag,
        price_reconstruction_mode=price_reconstruction_mode
    )


def run_reduced_var_train_test_pipeline(
    var_df,
    price_df,
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2025-12-31",
    window_years=7,
    step_years=2,
    max_lag=10,
    price_reconstruction_mode="dynamic"
):
    """
    Pipeline VAR train/test sans S&P 500.
    """
    return run_var_train_test_pipeline(
        var_df=var_df,
        price_df=price_df,
        variables=REDUCED_VAR_VARIABLES,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        window_years=window_years,
        step_years=step_years,
        max_lag=max_lag,
        price_reconstruction_mode=price_reconstruction_mode
    )


# ============================================================
# 21) Extraction des résidus de l'équation de l'or du VAR
# ============================================================
def extract_gold_var_residuals(fitted_model):
    """
    Extrait les résidus de l'équation de l'or d'un VAR estimé.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant :
        - date
        - gold_var_resid
    """
    residuals_df = fitted_model.resid.copy()

    if "gold_ret" not in residuals_df.columns:
        raise ValueError(
            f"La colonne 'gold_ret' est absente des résidus du modèle. "
            f"Colonnes disponibles : {list(residuals_df.columns)}"
        )

    gold_resid_df = residuals_df[["gold_ret"]].copy()
    gold_resid_df = gold_resid_df.rename(columns={"gold_ret": "gold_var_resid"})
    gold_resid_df = gold_resid_df.reset_index()

    if "index" in gold_resid_df.columns:
        gold_resid_df = gold_resid_df.rename(columns={"index": "date"})

    gold_resid_df["date"] = pd.to_datetime(gold_resid_df["date"])
    gold_resid_df = gold_resid_df.sort_values("date").reset_index(drop=True)

    return gold_resid_df




# ============================================================
# 22) Construction du DataFrame train pour le GARCH-X
# ============================================================
def build_garchx_train_dataset(gold_resid_train_df, macro_train_df):
    """
    Aligne parfaitement les résidus de l'or du VAR et les exogènes macro.

    Sortie :
    - date
    - gold_var_resid
    - gpr_daily
    - cpi_daily_transformed
    """
    resid_df = gold_resid_train_df.copy()
    macro_df = macro_train_df.copy()

    required_resid_cols = ["date", "gold_var_resid"]
    required_macro_cols = ["date", "gpr_level", "cpi_mom"]

    missing_resid = [col for col in required_resid_cols if col not in resid_df.columns]
    missing_macro = [col for col in required_macro_cols if col not in macro_df.columns]

    if missing_resid:
        raise ValueError(
            f"Colonnes manquantes dans gold_resid_train_df : {missing_resid}. "
            f"Colonnes disponibles : {list(resid_df.columns)}"
        )

    if missing_macro:
        raise ValueError(
            f"Colonnes manquantes dans macro_train_df : {missing_macro}. "
            f"Colonnes disponibles : {list(macro_df.columns)}"
        )

    resid_df["date"] = pd.to_datetime(resid_df["date"])
    macro_df["date"] = pd.to_datetime(macro_df["date"])

    resid_df = resid_df.sort_values("date").reset_index(drop=True)
    macro_df = macro_df.sort_values("date").reset_index(drop=True)

    df = resid_df.merge(
        macro_df[["date", "gpr_level", "cpi_mom"]],
        on="date",
        how="inner"
    ).sort_values("date").reset_index(drop=True)

    # On garde des colonnes transformées positives pour entrer dans la variance
    df["gpr_daily"] = pd.to_numeric(df["gpr_level"], errors="coerce")
    df["cpi_daily_transformed"] = pd.to_numeric(df["cpi_mom"], errors="coerce")

    df = df.dropna(subset=["gold_var_resid", "gpr_daily", "cpi_daily_transformed"]).copy()

    # Transformation simple pour garantir des exogènes positives dans la variance
    df["gpr_daily"] = df["gpr_daily"] - df["gpr_daily"].min() + 1e-8
    df["cpi_daily_transformed"] = df["cpi_daily_transformed"] - df["cpi_daily_transformed"].min() + 1e-8

    return df[["date", "gold_var_resid", "gpr_daily", "cpi_daily_transformed"]].copy()


# ============================================================
# 23) Récursion de variance GARCH(1,1)-Student
# ============================================================
def _compute_garch11_variance(eps, omega, alpha, beta):
    """
    Calcule la variance conditionnelle d'un GARCH(1,1) simple.
    """
    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(eps, ddof=1)

    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]

        if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
            return None

    return sigma2


# ============================================================
# 24) Récursion de variance GARCH(1,1)-X-Student
# ============================================================
def _compute_garch11x_variance(eps, x1, x2, omega, alpha, beta, gamma1, gamma2):
    """
    Calcule la variance conditionnelle d'un GARCH(1,1)-X.
    """
    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(eps, ddof=1)

    for t in range(1, n):
        sigma2[t] = (
            omega
            + alpha * (eps[t - 1] ** 2)
            + beta * sigma2[t - 1]
            + gamma1 * x1[t]
            + gamma2 * x2[t]
        )

        if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
            return None

    return sigma2


# ============================================================
# 25) Log-vraisemblance Student-t standardisée
# ============================================================
def _student_t_negloglik(eps, sigma2, nu):
    """
    Renvoie l'opposé de la log-vraisemblance Student-t standardisée.
    """
    if sigma2 is None:
        return np.inf

    if nu <= 2:
        return np.inf

    const = (
        gammaln((nu + 1) / 2)
        - gammaln(nu / 2)
        - 0.5 * np.log((nu - 2) * np.pi)
    )

    ll = const - 0.5 * np.log(sigma2) - ((nu + 1) / 2) * np.log(
        1 + (eps ** 2) / ((nu - 2) * sigma2)
    )

    if np.any(~np.isfinite(ll)):
        return np.inf

    return -np.sum(ll)


# ============================================================
# 26) Estimation GARCH(1,1)-Student sur les résidus du VAR
# ============================================================
def fit_garch11_student(gold_resid_train_df):
    """
    Estime un GARCH(1,1) simple avec innovations Student-t
    sur les résidus de l'or du VAR.
    """
    required_cols = ["date", "gold_var_resid"]
    missing_cols = [col for col in required_cols if col not in gold_resid_train_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans gold_resid_train_df : {missing_cols}. "
            f"Colonnes disponibles : {list(gold_resid_train_df.columns)}"
        )

    df = gold_resid_train_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["gold_var_resid"]).reset_index(drop=True)

    eps = df["gold_var_resid"].to_numpy(dtype=float)
    var_eps = np.var(eps, ddof=1)

    def objective(params):
        omega, alpha, beta, nu = params

        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999 or nu <= 2.01:
            return np.inf

        sigma2 = _compute_garch11_variance(eps, omega, alpha, beta)
        return _student_t_negloglik(eps, sigma2, nu)

    x0 = np.array([0.01 * var_eps, 0.05, 0.90, 8.0])

    bounds = [
        (1e-12, None),
        (0.0, 0.999),
        (0.0, 0.999),
        (2.01, 200.0),
    ]

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds
    )

    if not result.success:
        raise RuntimeError(f"Échec estimation GARCH(1,1)-Student : {result.message}")

    omega, alpha, beta, nu = result.x
    sigma2 = _compute_garch11_variance(eps, omega, alpha, beta)

    output_df = df.copy()
    output_df["garch_cond_var"] = sigma2
    output_df["garch_cond_vol"] = np.sqrt(sigma2)
    output_df["garch_std_resid"] = output_df["gold_var_resid"] / output_df["garch_cond_vol"]

    return {
        "model_name": "GARCH(1,1)-Student",
        "params": {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "nu": nu
        },
        "loglik": -result.fun,
        "success": result.success,
        "optimizer_message": result.message,
        "output_df": output_df
    }


# ============================================================
# 27) Estimation GARCH(1,1)-X-Student
# ============================================================
def fit_garch11x_student(garchx_train_df):
    """
    Estime un GARCH(1,1)-X avec innovations Student-t
    sur les résidus de l'or du VAR.

    Exogènes dans la variance :
    - gpr_daily
    - cpi_daily_transformed
    """
    required_cols = ["date", "gold_var_resid", "gpr_daily", "cpi_daily_transformed"]
    missing_cols = [col for col in required_cols if col not in garchx_train_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans garchx_train_df : {missing_cols}. "
            f"Colonnes disponibles : {list(garchx_train_df.columns)}"
        )

    df = garchx_train_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=required_cols[1:]).reset_index(drop=True)

    eps = df["gold_var_resid"].to_numpy(dtype=float)
    x1 = df["gpr_daily"].to_numpy(dtype=float)
    x2 = df["cpi_daily_transformed"].to_numpy(dtype=float)

    # mise à l'échelle simple pour aider l'optimisation
    x1 = x1 / np.std(x1) if np.std(x1) > 0 else x1
    x2 = x2 / np.std(x2) if np.std(x2) > 0 else x2

    var_eps = np.var(eps, ddof=1)

    def objective(params):
        omega, alpha, beta, gamma1, gamma2, nu = params

        if (
            omega <= 0
            or alpha < 0
            or beta < 0
            or gamma1 < 0
            or gamma2 < 0
            or (alpha + beta) >= 0.999
            or nu <= 2.01
        ):
            return np.inf

        sigma2 = _compute_garch11x_variance(
            eps, x1, x2, omega, alpha, beta, gamma1, gamma2
        )
        return _student_t_negloglik(eps, sigma2, nu)

    x0 = np.array([0.01 * var_eps, 0.05, 0.85, 0.01 * var_eps, 0.01 * var_eps, 8.0])

    bounds = [
        (1e-12, None),
        (0.0, 0.999),
        (0.0, 0.999),
        (0.0, None),
        (0.0, None),
        (2.01, 200.0),
    ]

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds
    )

    if not result.success:
        raise RuntimeError(f"Échec estimation GARCH(1,1)-X-Student : {result.message}")

    omega, alpha, beta, gamma1, gamma2, nu = result.x
    sigma2 = _compute_garch11x_variance(
        eps, x1, x2, omega, alpha, beta, gamma1, gamma2
    )

    output_df = df.copy()
    output_df["garch_cond_var"] = sigma2
    output_df["garch_cond_vol"] = np.sqrt(sigma2)
    output_df["garch_std_resid"] = output_df["gold_var_resid"] / output_df["garch_cond_vol"]

    return {
        "model_name": "GARCH(1,1)-X-Student",
        "params": {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "gamma_gpr": gamma1,
            "gamma_cpi": gamma2,
            "nu": nu
        },
        "loglik": -result.fun,
        "success": result.success,
        "optimizer_message": result.message,
        "output_df": output_df
    }


# ============================================================
# 28) Tableau résumé simple du GARCH / GARCH-X
# ============================================================
def get_garch_model_info(garch_result):
    """
    Renvoie un tableau simple des paramètres estimés.
    """
    info = {
        "model_name": garch_result["model_name"],
        "loglik": garch_result["loglik"],
        "success": garch_result["success"],
        "optimizer_message": garch_result["optimizer_message"],
    }

    info.update(garch_result["params"])

    return info


# ============================================================
# 29) Helpers for out-of-sample GARCH / GARCH-X forecast tables
# ============================================================
def _check_return_forecast_dataframe(
    forecast_df,
    date_col="date",
    pred_return_col="gold_ret_pred",
    actual_return_col=None
):
    """
    Validates a forecast DataFrame containing predicted returns.
    """
    required_cols = [date_col, pred_return_col]
    missing_cols = [col for col in required_cols if col not in forecast_df.columns]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans forecast_df : {missing_cols}. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )

    if actual_return_col is not None and actual_return_col not in forecast_df.columns:
        raise ValueError(
            f"La colonne '{actual_return_col}' est absente de forecast_df. "
            f"Colonnes disponibles : {list(forecast_df.columns)}"
        )


def build_garch_forecast_df(
    garch_result,
    gold_resid_train_df,
    forecast_df,
    date_col="date",
    pred_return_col="gold_ret_pred",
    actual_return_col=None
):
    """
    Builds an out-of-sample forecast DataFrame for plain GARCH.

    The mean forecast comes from `forecast_df`.
    The volatility forecast is generated recursively from the fitted GARCH.
    """
    _check_return_forecast_dataframe(
        forecast_df=forecast_df,
        date_col=date_col,
        pred_return_col=pred_return_col,
        actual_return_col=actual_return_col
    )

    required_resid_cols = [date_col, "gold_var_resid"]
    missing_resid_cols = [col for col in required_resid_cols if col not in gold_resid_train_df.columns]

    if missing_resid_cols:
        raise ValueError(
            f"Colonnes manquantes dans gold_resid_train_df : {missing_resid_cols}. "
            f"Colonnes disponibles : {list(gold_resid_train_df.columns)}"
        )

    if "params" not in garch_result or "output_df" not in garch_result:
        raise ValueError("garch_result doit contenir au minimum 'params' et 'output_df'.")

    params = garch_result["params"]
    output_df = garch_result["output_df"].copy()

    for key in ["omega", "alpha", "beta"]:
        if key not in params:
            raise ValueError(f"Paramètre manquant dans garch_result['params'] : {key}")

    if "garch_cond_var" not in output_df.columns:
        raise ValueError("La colonne 'garch_cond_var' est absente de garch_result['output_df'].")

    forecast_out = forecast_df.copy()
    train_resid = gold_resid_train_df.copy()

    forecast_out[date_col] = pd.to_datetime(forecast_out[date_col])
    train_resid[date_col] = pd.to_datetime(train_resid[date_col])
    output_df[date_col] = pd.to_datetime(output_df[date_col])

    forecast_out = forecast_out.sort_values(date_col).reset_index(drop=True)
    train_resid = train_resid.sort_values(date_col).reset_index(drop=True)
    output_df = output_df.sort_values(date_col).reset_index(drop=True)

    last_eps2 = float(train_resid.iloc[-1]["gold_var_resid"] ** 2)
    last_sigma2 = float(output_df.iloc[-1]["garch_cond_var"])

    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])

    sigma2_forecast = []

    for _ in range(len(forecast_out)):
        sigma2_next = omega + alpha * last_eps2 + beta * last_sigma2

        if sigma2_next <= 0 or not np.isfinite(sigma2_next):
            raise ValueError("La variance prédite GARCH est invalide.")

        sigma2_forecast.append(float(sigma2_next))

        # Dynamic multi-step recursion
        last_sigma2 = sigma2_next
        last_eps2 = sigma2_next

    result_df = forecast_out[[date_col, pred_return_col]].copy()

    if actual_return_col is not None and actual_return_col in forecast_out.columns:
        result_df[actual_return_col] = forecast_out[actual_return_col]

    result_df["garch_cond_var_pred"] = sigma2_forecast
    result_df["garch_cond_vol_pred"] = np.sqrt(result_df["garch_cond_var_pred"])

    return result_df


def build_garchx_forecast_df(
    garchx_result,
    gold_resid_train_df,
    macro_train_df,
    macro_test_df,
    forecast_df,
    date_col="date",
    pred_return_col="gold_ret_pred",
    actual_return_col=None
):
    """
    Builds an out-of-sample forecast DataFrame for GARCH-X.

    The mean forecast comes from `forecast_df`.
    The volatility forecast is generated recursively from the fitted GARCH-X.
    Exogenous variables are transformed on the test set using the train-set
    transformation and scaling rules.
    """
    _check_return_forecast_dataframe(
        forecast_df=forecast_df,
        date_col=date_col,
        pred_return_col=pred_return_col,
        actual_return_col=actual_return_col
    )

    required_resid_cols = [date_col, "gold_var_resid"]
    missing_resid_cols = [col for col in required_resid_cols if col not in gold_resid_train_df.columns]

    if missing_resid_cols:
        raise ValueError(
            f"Colonnes manquantes dans gold_resid_train_df : {missing_resid_cols}. "
            f"Colonnes disponibles : {list(gold_resid_train_df.columns)}"
        )

    required_macro_cols = [date_col, "gpr_level", "cpi_mom"]

    missing_train_macro = [col for col in required_macro_cols if col not in macro_train_df.columns]
    if missing_train_macro:
        raise ValueError(
            f"Colonnes manquantes dans macro_train_df : {missing_train_macro}. "
            f"Colonnes disponibles : {list(macro_train_df.columns)}"
        )

    missing_test_macro = [col for col in required_macro_cols if col not in macro_test_df.columns]
    if missing_test_macro:
        raise ValueError(
            f"Colonnes manquantes dans macro_test_df : {missing_test_macro}. "
            f"Colonnes disponibles : {list(macro_test_df.columns)}"
        )

    if "params" not in garchx_result or "output_df" not in garchx_result:
        raise ValueError("garchx_result doit contenir au minimum 'params' et 'output_df'.")

    params = garchx_result["params"]
    output_df = garchx_result["output_df"].copy()

    for key in ["omega", "alpha", "beta", "gamma_gpr", "gamma_cpi"]:
        if key not in params:
            raise ValueError(f"Paramètre manquant dans garchx_result['params'] : {key}")

    if "garch_cond_var" not in output_df.columns:
        raise ValueError("La colonne 'garch_cond_var' est absente de garchx_result['output_df'].")

    forecast_out = forecast_df.copy()
    train_resid = gold_resid_train_df.copy()
    macro_train = macro_train_df.copy()
    macro_test = macro_test_df.copy()

    forecast_out[date_col] = pd.to_datetime(forecast_out[date_col])
    train_resid[date_col] = pd.to_datetime(train_resid[date_col])
    output_df[date_col] = pd.to_datetime(output_df[date_col])
    macro_train[date_col] = pd.to_datetime(macro_train[date_col])
    macro_test[date_col] = pd.to_datetime(macro_test[date_col])

    forecast_out = forecast_out.sort_values(date_col).reset_index(drop=True)
    train_resid = train_resid.sort_values(date_col).reset_index(drop=True)
    output_df = output_df.sort_values(date_col).reset_index(drop=True)
    macro_train = macro_train.sort_values(date_col).reset_index(drop=True)
    macro_test = macro_test.sort_values(date_col).reset_index(drop=True)

    macro_train["gpr_level"] = pd.to_numeric(macro_train["gpr_level"], errors="coerce")
    macro_train["cpi_mom"] = pd.to_numeric(macro_train["cpi_mom"], errors="coerce")
    macro_test["gpr_level"] = pd.to_numeric(macro_test["gpr_level"], errors="coerce")
    macro_test["cpi_mom"] = pd.to_numeric(macro_test["cpi_mom"], errors="coerce")

    macro_train = macro_train.dropna(subset=["gpr_level", "cpi_mom"]).copy()
    macro_test = macro_test.dropna(subset=["gpr_level", "cpi_mom"]).copy()

    gpr_min_train = float(macro_train["gpr_level"].min())
    cpi_min_train = float(macro_train["cpi_mom"].min())

    macro_train["gpr_daily"] = macro_train["gpr_level"] - gpr_min_train + 1e-8
    macro_train["cpi_daily_transformed"] = macro_train["cpi_mom"] - cpi_min_train + 1e-8

    macro_test["gpr_daily"] = macro_test["gpr_level"] - gpr_min_train + 1e-8
    macro_test["cpi_daily_transformed"] = macro_test["cpi_mom"] - cpi_min_train + 1e-8

    gpr_std_train = float(np.std(macro_train["gpr_daily"].to_numpy(dtype=float)))
    cpi_std_train = float(np.std(macro_train["cpi_daily_transformed"].to_numpy(dtype=float)))

    if gpr_std_train > 0:
        macro_test["gpr_daily_scaled"] = macro_test["gpr_daily"] / gpr_std_train
    else:
        macro_test["gpr_daily_scaled"] = macro_test["gpr_daily"]

    if cpi_std_train > 0:
        macro_test["cpi_daily_scaled"] = macro_test["cpi_daily_transformed"] / cpi_std_train
    else:
        macro_test["cpi_daily_scaled"] = macro_test["cpi_daily_transformed"]

    merged_df = forecast_out.merge(
        macro_test[
            [date_col, "gpr_level", "cpi_mom", "gpr_daily_scaled", "cpi_daily_scaled"]
        ],
        on=date_col,
        how="inner"
    ).sort_values(date_col).reset_index(drop=True)

    if merged_df.empty:
        raise ValueError("Aucune date commune entre forecast_df et macro_test_df.")

    last_eps2 = float(train_resid.iloc[-1]["gold_var_resid"] ** 2)
    last_sigma2 = float(output_df.iloc[-1]["garch_cond_var"])

    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma_gpr = float(params["gamma_gpr"])
    gamma_cpi = float(params["gamma_cpi"])

    sigma2_forecast = []

    for _, row in merged_df.iterrows():
        sigma2_next = (
            omega
            + alpha * last_eps2
            + beta * last_sigma2
            + gamma_gpr * float(row["gpr_daily_scaled"])
            + gamma_cpi * float(row["cpi_daily_scaled"])
        )

        if sigma2_next <= 0 or not np.isfinite(sigma2_next):
            raise ValueError("La variance prédite GARCH-X est invalide.")

        sigma2_forecast.append(float(sigma2_next))

        # Dynamic multi-step recursion
        last_sigma2 = sigma2_next
        last_eps2 = sigma2_next

    result_cols = [date_col, pred_return_col]

    if actual_return_col is not None and actual_return_col in merged_df.columns:
        result_cols.append(actual_return_col)

    result_df = merged_df[result_cols + ["gpr_level", "cpi_mom"]].copy()
    result_df["garch_cond_var_pred"] = sigma2_forecast
    result_df["garch_cond_vol_pred"] = np.sqrt(result_df["garch_cond_var_pred"])

    return result_df

# ============================================================
# 30) Construction du DataFrame d'entrée pour la simulation
# ============================================================
def build_simulation_input_df(
    return_forecast_df,
    vol_forecast_df,
    date_col="date",
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred",
    actual_return_col="gold_ret_actual"
):
    """
    Construit le DataFrame d'entrée de la simulation Monte Carlo
    en alignant parfaitement la moyenne prédite et la volatilité prédite.

    Paramètres
    ----------
    return_forecast_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - gold_ret_pred
        - éventuellement gold_ret_actual
    vol_forecast_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - garch_cond_vol_pred
        - éventuellement gold_ret_actual
    date_col : str
        Nom de la colonne de date.
    return_col : str
        Nom de la colonne de rendement moyen prédit.
    vol_col : str
        Nom de la colonne de volatilité prédite.
    actual_return_col : str
        Nom de la colonne de rendement observé, si disponible.

    Retour
    ------
    pd.DataFrame
        DataFrame aligné contenant au minimum :
        - date
        - gold_ret_pred
        - garch_cond_vol_pred
        - éventuellement gold_ret_actual
    """
    return_has_actual = actual_return_col in return_forecast_df.columns
    vol_has_actual = actual_return_col in vol_forecast_df.columns

    _check_return_forecast_dataframe(
        forecast_df=return_forecast_df,
        date_col=date_col,
        pred_return_col=return_col,
        actual_return_col=actual_return_col if return_has_actual else None
    )

    required_vol_cols = [date_col, vol_col]
    missing_vol_cols = [col for col in required_vol_cols if col not in vol_forecast_df.columns]

    if missing_vol_cols:
        raise ValueError(
            f"Colonnes manquantes dans vol_forecast_df : {missing_vol_cols}. "
            f"Colonnes disponibles : {list(vol_forecast_df.columns)}"
        )

    return_df = return_forecast_df.copy()
    vol_df = vol_forecast_df.copy()

    return_df[date_col] = pd.to_datetime(return_df[date_col])
    vol_df[date_col] = pd.to_datetime(vol_df[date_col])

    if return_df[date_col].duplicated().any():
        raise ValueError("Des dates dupliquées ont été détectées dans return_forecast_df.")

    if vol_df[date_col].duplicated().any():
        raise ValueError("Des dates dupliquées ont été détectées dans vol_forecast_df.")

    return_df = return_df.sort_values(date_col).reset_index(drop=True)
    vol_df = vol_df.sort_values(date_col).reset_index(drop=True)

    return_dates = return_df[date_col].reset_index(drop=True)
    vol_dates = vol_df[date_col].reset_index(drop=True)

    # On impose un alignement parfait des dates
    if not return_dates.equals(vol_dates):
        missing_in_vol = return_dates[~return_dates.isin(vol_dates)].dt.strftime("%Y-%m-%d").tolist()
        missing_in_return = vol_dates[~vol_dates.isin(return_dates)].dt.strftime("%Y-%m-%d").tolist()

        raise ValueError(
            "Les dates de return_forecast_df et vol_forecast_df ne sont pas parfaitement alignées. "
            f"Dates absentes de vol_forecast_df : {missing_in_vol[:5]}. "
            f"Dates absentes de return_forecast_df : {missing_in_return[:5]}."
        )

    # Si les deux DataFrames contiennent le rendement observé,
    # on vérifie qu'il est cohérent.
    if return_has_actual and vol_has_actual:
        actual_compare_df = return_df[[date_col, actual_return_col]].merge(
            vol_df[[date_col, actual_return_col]],
            on=date_col,
            how="inner",
            suffixes=("_return", "_vol")
        )

        actual_return = pd.to_numeric(
            actual_compare_df[f"{actual_return_col}_return"],
            errors="coerce"
        ).to_numpy(dtype=float)

        actual_vol = pd.to_numeric(
            actual_compare_df[f"{actual_return_col}_vol"],
            errors="coerce"
        ).to_numpy(dtype=float)

        if not np.allclose(actual_return, actual_vol, equal_nan=True):
            raise ValueError(
                f"La colonne '{actual_return_col}' diffère entre return_forecast_df et vol_forecast_df."
            )

    simulation_input_df = return_df[[date_col, return_col]].merge(
        vol_df[[date_col, vol_col]],
        on=date_col,
        how="inner"
    )

    # On conserve gold_ret_actual si disponible dans l'un des deux DataFrames
    if return_has_actual:
        simulation_input_df = simulation_input_df.merge(
            return_df[[date_col, actual_return_col]],
            on=date_col,
            how="left"
        )
    elif vol_has_actual:
        simulation_input_df = simulation_input_df.merge(
            vol_df[[date_col, actual_return_col]],
            on=date_col,
            how="left"
        )

    ordered_cols = [date_col, return_col, vol_col]
    if actual_return_col in simulation_input_df.columns:
        ordered_cols.append(actual_return_col)

    simulation_input_df = simulation_input_df[ordered_cols].copy()
    simulation_input_df = simulation_input_df.sort_values(date_col).reset_index(drop=True)

    if simulation_input_df.empty:
        raise ValueError("Le DataFrame final de simulation est vide.")

    return simulation_input_df


# ============================================================
# 31) Simulation des chocs Student-t standardisés
# ============================================================
def simulate_standardized_student_shocks(
    n_periods,
    n_simulations,
    nu,
    random_state=None
):
    """
    Simule des innovations Student-t standardisées de variance 1.

    Si u ~ t_nu, on utilise :
        z = u * sqrt((nu - 2) / nu)

    Paramètres
    ----------
    n_periods : int
        Nombre de dates à simuler.
    n_simulations : int
        Nombre de trajectoires Monte Carlo.
    nu : float
        Nombre de degrés de liberté de la Student-t.
    random_state : int, np.random.Generator ou None
        Graine ou générateur aléatoire.

    Retour
    ------
    np.ndarray
        Matrice de taille (n_periods, n_simulations).
    """
    n_periods = int(n_periods)
    n_simulations = int(n_simulations)
    nu = float(nu)

    if n_periods <= 0:
        raise ValueError("n_periods doit être un entier strictement positif.")

    if n_simulations <= 0:
        raise ValueError("n_simulations doit être un entier strictement positif.")

    if nu <= 2:
        raise ValueError("nu doit être strictement supérieur à 2 pour standardiser la Student-t.")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    student_draws = rng.standard_t(df=nu, size=(n_periods, n_simulations))
    standardized_draws = student_draws * np.sqrt((nu - 2.0) / nu)

    return standardized_draws


# ============================================================
# 32) Simulation de la distribution des rendements
# ============================================================
def simulate_return_distribution(
    simulation_input_df,
    nu,
    n_simulations=1000,
    random_state=None,
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred"
):
    """
    Simule la distribution des rendements conditionnels :

        r_t^(b) = mu_t + sigma_t * z_t^(b)

    avec z_t^(b) Student-t standardisée de variance 1.

    Paramètres
    ----------
    simulation_input_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - gold_ret_pred
        - garch_cond_vol_pred
    nu : float
        Paramètre de degrés de liberté issu du GARCH / GARCH-X.
    n_simulations : int
        Nombre de trajectoires Monte Carlo.
    random_state : int, np.random.Generator ou None
        Graine ou générateur aléatoire.
    return_col : str
        Nom de la colonne de moyenne prédite.
    vol_col : str
        Nom de la colonne de volatilité prédite.

    Retour
    ------
    np.ndarray
        Matrice de rendements simulés de taille (n_periods, n_simulations).
    """
    required_cols = ["date", return_col, vol_col]
    missing_cols = [col for col in required_cols if col not in simulation_input_df.columns]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans simulation_input_df : {missing_cols}. "
            f"Colonnes disponibles : {list(simulation_input_df.columns)}"
        )

    df = simulation_input_df.copy()
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    if df[[return_col, vol_col]].isna().any().any():
        raise ValueError(
            f"Des valeurs manquantes ont été détectées dans '{return_col}' ou '{vol_col}'."
        )

    if (df[vol_col] < 0).any():
        raise ValueError(f"La colonne '{vol_col}' contient des valeurs négatives.")

    mu = df[return_col].to_numpy(dtype=float).reshape(-1, 1)
    sigma = df[vol_col].to_numpy(dtype=float).reshape(-1, 1)

    standardized_shocks = simulate_standardized_student_shocks(
        n_periods=len(df),
        n_simulations=n_simulations,
        nu=nu,
        random_state=random_state
    )

    simulated_returns = mu + sigma * standardized_shocks

    return simulated_returns


# ============================================================
# 33) Résumé des rendements simulés
# ============================================================
def summarize_simulated_returns(
    simulation_input_df,
    simulated_returns_matrix,
    date_col="date",
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred",
    actual_return_col="gold_ret_actual"
):
    """
    Résume, date par date, la distribution simulée des rendements.

    Statistiques calculées :
    - moyenne
    - médiane
    - quantile 5%
    - quantile 95%

    Paramètres
    ----------
    simulation_input_df : pd.DataFrame
        DataFrame d'entrée de la simulation.
    simulated_returns_matrix : np.ndarray
        Matrice des rendements simulés.
    date_col : str
        Nom de la colonne de date.
    return_col : str
        Nom de la colonne de rendement moyen prédit.
    vol_col : str
        Nom de la colonne de volatilité prédite.
    actual_return_col : str
        Nom de la colonne de rendement observé, si disponible.

    Retour
    ------
    pd.DataFrame
        DataFrame résumé des rendements simulés.
    """
    required_cols = [date_col, return_col, vol_col]
    missing_cols = [col for col in required_cols if col not in simulation_input_df.columns]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans simulation_input_df : {missing_cols}. "
            f"Colonnes disponibles : {list(simulation_input_df.columns)}"
        )

    n_periods = simulation_input_df.shape[0]

    if simulated_returns_matrix.shape[0] != n_periods:
        raise ValueError(
            "Le nombre de lignes de simulated_returns_matrix ne correspond pas "
            "au nombre de dates de simulation_input_df."
        )

    summary_df = simulation_input_df[[date_col, return_col, vol_col]].copy()

    if actual_return_col in simulation_input_df.columns:
        summary_df[actual_return_col] = simulation_input_df[actual_return_col].to_numpy()

    summary_df["gold_ret_sim_mean"] = np.mean(simulated_returns_matrix, axis=1)
    summary_df["gold_ret_sim_median"] = np.median(simulated_returns_matrix, axis=1)
    summary_df["gold_ret_sim_q05"] = np.quantile(simulated_returns_matrix, 0.05, axis=1)
    summary_df["gold_ret_sim_q95"] = np.quantile(simulated_returns_matrix, 0.95, axis=1)

    ordered_cols = [date_col, return_col, vol_col]
    if actual_return_col in summary_df.columns:
        ordered_cols.append(actual_return_col)

    ordered_cols.extend([
        "gold_ret_sim_mean",
        "gold_ret_sim_median",
        "gold_ret_sim_q05",
        "gold_ret_sim_q95"
    ])

    return summary_df[ordered_cols].copy()


# ============================================================
# 34) Reconstruction des trajectoires simulées de prix
# ============================================================
def simulate_price_paths_from_returns(
    price_df,
    simulation_input_df,
    simulated_returns_matrix,
    date_col="date",
    price_col="gold_price"
):
    """
    Reconstruit les trajectoires simulées de prix à partir des log-rendements
    simulés selon :

        P_t^(b) = P_(t-1)^(b) * exp(r_t^(b))

    Paramètres
    ----------
    price_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - gold_price
    simulation_input_df : pd.DataFrame
        DataFrame d'entrée de la simulation.
    simulated_returns_matrix : np.ndarray
        Matrice des rendements simulés.
    date_col : str
        Nom de la colonne de date.
    price_col : str
        Nom de la colonne de prix observé.

    Retour
    ------
    np.ndarray
        Matrice des prix simulés de taille (n_periods, n_simulations).
    """
    required_price_cols = [date_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in price_df.columns]

    if missing_price_cols:
        raise ValueError(
            f"Colonnes manquantes dans price_df : {missing_price_cols}. "
            f"Colonnes disponibles : {list(price_df.columns)}"
        )

    sim_df = simulation_input_df.copy()
    prices_df = price_df.copy()

    sim_df[date_col] = pd.to_datetime(sim_df[date_col])
    prices_df[date_col] = pd.to_datetime(prices_df[date_col])

    if sim_df[date_col].duplicated().any():
        raise ValueError("Des dates dupliquées ont été détectées dans simulation_input_df.")

    if prices_df[date_col].duplicated().any():
        raise ValueError("Des dates dupliquées ont été détectées dans price_df.")

    sim_df = sim_df.sort_values(date_col).reset_index(drop=True)
    prices_df = prices_df.sort_values(date_col).reset_index(drop=True)

    n_periods = sim_df.shape[0]

    if simulated_returns_matrix.shape[0] != n_periods:
        raise ValueError(
            "Le nombre de lignes de simulated_returns_matrix ne correspond pas "
            "au nombre de dates de simulation_input_df."
        )

    first_simulation_date = sim_df.loc[0, date_col]

    # On récupère le dernier prix observé juste avant le début du test
    initial_price_df = prices_df.loc[
        prices_df[date_col] < first_simulation_date,
        [date_col, price_col]
    ]

    if initial_price_df.empty:
        raise ValueError(
            "Impossible de trouver le dernier prix observé avant le début de la simulation."
        )

    initial_price = float(initial_price_df.iloc[-1][price_col])
    n_simulations = simulated_returns_matrix.shape[1]

    simulated_price_matrix = np.zeros((n_periods, n_simulations), dtype=float)
    current_prices = np.full(n_simulations, initial_price, dtype=float)

    for t in range(n_periods):
        current_prices = current_prices * np.exp(simulated_returns_matrix[t, :])
        simulated_price_matrix[t, :] = current_prices

    return simulated_price_matrix


# ============================================================
# 35) Résumé des prix simulés
# ============================================================
def summarize_simulated_prices(
    price_df,
    simulation_input_df,
    simulated_price_matrix,
    date_col="date",
    price_col="gold_price"
):
    """
    Résume, date par date, la distribution simulée des prix.

    Statistiques calculées :
    - moyenne
    - médiane
    - quantile 5%
    - quantile 95%

    Paramètres
    ----------
    price_df : pd.DataFrame
        DataFrame contenant les prix observés.
    simulation_input_df : pd.DataFrame
        DataFrame d'entrée de la simulation.
    simulated_price_matrix : np.ndarray
        Matrice des prix simulés.
    date_col : str
        Nom de la colonne de date.
    price_col : str
        Nom de la colonne de prix observé.

    Retour
    ------
    pd.DataFrame
        DataFrame résumé des prix simulés avec les prix observés.
    """
    required_price_cols = [date_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in price_df.columns]

    if missing_price_cols:
        raise ValueError(
            f"Colonnes manquantes dans price_df : {missing_price_cols}. "
            f"Colonnes disponibles : {list(price_df.columns)}"
        )

    sim_df = simulation_input_df.copy()
    prices_df = price_df.copy()

    sim_df[date_col] = pd.to_datetime(sim_df[date_col])
    prices_df[date_col] = pd.to_datetime(prices_df[date_col])

    sim_df = sim_df.sort_values(date_col).reset_index(drop=True)
    prices_df = prices_df.sort_values(date_col).reset_index(drop=True)

    n_periods = sim_df.shape[0]

    if simulated_price_matrix.shape[0] != n_periods:
        raise ValueError(
            "Le nombre de lignes de simulated_price_matrix ne correspond pas "
            "au nombre de dates de simulation_input_df."
        )

    summary_df = sim_df[[date_col]].merge(
        prices_df[[date_col, price_col]],
        on=date_col,
        how="left"
    )

    if summary_df[price_col].isna().any():
        raise ValueError(
            "Certaines dates simulées n'ont pas de prix observé correspondant dans price_df."
        )

    summary_df = summary_df.rename(columns={price_col: "gold_price_actual"})
    summary_df["gold_price_sim_mean"] = np.mean(simulated_price_matrix, axis=1)
    summary_df["gold_price_sim_median"] = np.median(simulated_price_matrix, axis=1)
    summary_df["gold_price_sim_q05"] = np.quantile(simulated_price_matrix, 0.05, axis=1)
    summary_df["gold_price_sim_q95"] = np.quantile(simulated_price_matrix, 0.95, axis=1)

    ordered_cols = [
        date_col,
        "gold_price_actual",
        "gold_price_sim_mean",
        "gold_price_sim_median",
        "gold_price_sim_q05",
        "gold_price_sim_q95"
    ]

    return summary_df[ordered_cols].copy()


# ============================================================
# 36) Pipeline Monte Carlo VAR-GARCH-X
# ============================================================
def run_var_garchx_monte_carlo_pipeline(
    price_df,
    return_forecast_df,
    vol_forecast_df,
    nu,
    n_simulations=1000,
    random_state=None,
    date_col="date",
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred",
    actual_return_col="gold_ret_actual",
    price_col="gold_price"
):
    """
    Exécute le workflow complet de simulation Monte Carlo VAR-GARCH-X.

    Étapes :
    1. alignement de la moyenne et de la volatilité prédites
    2. simulation des rendements conditionnels
    3. résumé des rendements simulés
    4. reconstruction des trajectoires simulées de prix
    5. résumé des prix simulés

    Paramètres
    ----------
    price_df : pd.DataFrame
        DataFrame des prix observés.
    return_forecast_df : pd.DataFrame
        DataFrame des rendements moyens prédits.
    vol_forecast_df : pd.DataFrame
        DataFrame des volatilités prédites.
    nu : float
        Paramètre de degrés de liberté Student-t issu du GARCH / GARCH-X.
    n_simulations : int
        Nombre de trajectoires Monte Carlo.
    random_state : int, np.random.Generator ou None
        Graine ou générateur aléatoire.
    date_col : str
        Nom de la colonne de date.
    return_col : str
        Nom de la colonne de rendement moyen prédit.
    vol_col : str
        Nom de la colonne de volatilité prédite.
    actual_return_col : str
        Nom de la colonne de rendement observé, si disponible.
    price_col : str
        Nom de la colonne de prix observé.

    Retour
    ------
    dict
        Dictionnaire contenant :
        - simulation_input_df
        - simulated_returns_matrix
        - simulated_returns_summary_df
        - simulated_price_matrix
        - simulated_price_summary_df
    """
    simulation_input_df = build_simulation_input_df(
        return_forecast_df=return_forecast_df,
        vol_forecast_df=vol_forecast_df,
        date_col=date_col,
        return_col=return_col,
        vol_col=vol_col,
        actual_return_col=actual_return_col
    )

    simulated_returns_matrix = simulate_return_distribution(
        simulation_input_df=simulation_input_df,
        nu=nu,
        n_simulations=n_simulations,
        random_state=random_state,
        return_col=return_col,
        vol_col=vol_col
    )

    simulated_returns_summary_df = summarize_simulated_returns(
        simulation_input_df=simulation_input_df,
        simulated_returns_matrix=simulated_returns_matrix,
        date_col=date_col,
        return_col=return_col,
        vol_col=vol_col,
        actual_return_col=actual_return_col
    )

    simulated_price_matrix = simulate_price_paths_from_returns(
        price_df=price_df,
        simulation_input_df=simulation_input_df,
        simulated_returns_matrix=simulated_returns_matrix,
        date_col=date_col,
        price_col=price_col
    )

    simulated_price_summary_df = summarize_simulated_prices(
        price_df=price_df,
        simulation_input_df=simulation_input_df,
        simulated_price_matrix=simulated_price_matrix,
        date_col=date_col,
        price_col=price_col
    )

    return {
        "simulation_input_df": simulation_input_df,
        "simulated_returns_matrix": simulated_returns_matrix,
        "simulated_returns_summary_df": simulated_returns_summary_df,
        "simulated_price_matrix": simulated_price_matrix,
        "simulated_price_summary_df": simulated_price_summary_df
    }
