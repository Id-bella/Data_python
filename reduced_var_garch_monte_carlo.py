import pandas as pd
import numpy as np

from preprocessing import (
    prepare_var_dataset,
    load_market_data,
    clean_all_market_data,
)

from modele import (
    split_var_train_test_by_date,
    select_var_lag_on_train,
    fit_var_on_train,
    forecast_var_dynamically_on_test,
    extract_gold_var_residuals,
    fit_garch11_student,
    build_garch_forecast_df,
    build_simulation_input_df,
    simulate_standardized_student_shocks,
    simulate_return_distribution,
    summarize_simulated_returns,
    simulate_price_paths_from_returns,
    summarize_simulated_prices,
    build_dynamic_price_forecast,
)


REDUCED_VAR_GARCH_VARIABLES = ["gold_ret", "dxy_ret", "vix_ret"]


# ============================================================
# 1) Construction du DataFrame de prix bruts
# ============================================================
def build_reduced_var_garch_price_df(data_dir="data"):
    """
    Construit le DataFrame des prix bruts utiles au pipeline.

    Paramètres
    ----------
    data_dir : str
        Dossier contenant les fichiers de marché.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant :
        - date
        - gold_price
        - dxy_price
        - sp500_price
        - vix_price
    """
    raw_data = load_market_data(data_dir=data_dir)
    cleaned_data = clean_all_market_data(raw_data)

    gold = cleaned_data["gold"][["date", "gold_price"]].copy()
    dxy = cleaned_data["dxy"][["date", "dxy_price"]].copy()
    sp500 = cleaned_data["sp500"][["date", "sp500_price"]].copy()
    vix = cleaned_data["vix"][["date", "vix_price"]].copy()

    price_df = gold.merge(dxy, on="date", how="inner")
    price_df = price_df.merge(sp500, on="date", how="inner")
    price_df = price_df.merge(vix, on="date", how="inner")

    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date").reset_index(drop=True)

    return price_df


# ============================================================
# 2) Préparation des données train/test du VAR réduit
# ============================================================
def prepare_reduced_var_train_test_data(
    var_df=None,
    data_dir="data",
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2021-12-31"
):
    """
    Prépare les sous-échantillons train/test du VAR réduit.

    Paramètres
    ----------
    var_df : pd.DataFrame ou None
        DataFrame journalier du VAR. Si None, il est reconstruit via
        `prepare_var_dataset`.
    data_dir : str
        Dossier contenant les données.
    train_start : str
    train_end : str
    test_start : str
    test_end : str

    Retour
    ------
    tuple
        (var_df, train_df, test_df)
    """
    if var_df is None:
        var_df = prepare_var_dataset(data_dir=data_dir)

    train_df, test_df = split_var_train_test_by_date(
        var_df=var_df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    return var_df, train_df, test_df


# ============================================================
# 3) Sélection du lag sur le train pour le VAR réduit
# ============================================================
def select_reduced_var_lag_on_train(
    train_df,
    window_years=7,
    step_years=2,
    max_lag=10
):
    """
    Applique la sélection du lag uniquement sur le train
    pour le VAR réduit :
    - gold_ret
    - dxy_ret
    - vix_ret
    """
    return select_var_lag_on_train(
        train_df=train_df,
        variables=REDUCED_VAR_GARCH_VARIABLES,
        window_years=window_years,
        step_years=step_years,
        max_lag=max_lag
    )


# ============================================================
# 4) Estimation finale du VAR réduit sur le train
# ============================================================
def fit_reduced_var_on_train(train_df, lag_order):
    """
    Estime le VAR réduit final sur le train.
    """
    return fit_var_on_train(
        train_df=train_df,
        lag_order=lag_order,
        variables=REDUCED_VAR_GARCH_VARIABLES
    )


# ============================================================
# 5) Prévisions dynamiques du VAR réduit sur le test
# ============================================================
def forecast_reduced_var_dynamically_on_test(fitted_var_model, train_df, test_df):
    """
    Produit les prévisions dynamiques du VAR réduit sur le test.
    """
    return forecast_var_dynamically_on_test(
        fitted_model=fitted_var_model,
        train_df=train_df,
        test_df=test_df,
        variables=REDUCED_VAR_GARCH_VARIABLES
    )


# ============================================================
# 6) Extraction des résidus de l'or sur le train
# ============================================================
def extract_reduced_var_gold_residuals(fitted_var_model):
    """
    Extrait les résidus de l'équation de l'or du VAR réduit.
    """
    return extract_gold_var_residuals(fitted_var_model)


# ============================================================
# 7) Estimation du GARCH(1,1)-Student
# ============================================================
def fit_reduced_var_garch_student(gold_resid_train_df):
    """
    Estime le GARCH(1,1)-Student sur les résidus de l'or.
    """
    return fit_garch11_student(gold_resid_train_df)


# ============================================================
# 8) Prévision de la volatilité conditionnelle sur le test
# ============================================================
def forecast_reduced_var_garch_volatility(
    garch_result,
    gold_resid_train_df,
    return_forecast_df,
    date_col="date",
    pred_return_col="gold_ret_pred",
    actual_return_col="gold_ret_actual"
):
    """
    Produit les prévisions de volatilité conditionnelle sur le test
    à partir du GARCH(1,1)-Student estimé.
    """
    actual_col = actual_return_col if actual_return_col in return_forecast_df.columns else None

    return build_garch_forecast_df(
        garch_result=garch_result,
        gold_resid_train_df=gold_resid_train_df,
        forecast_df=return_forecast_df,
        date_col=date_col,
        pred_return_col=pred_return_col,
        actual_return_col=actual_col
    )


# ============================================================
# 9) Construction du DataFrame d'entrée de simulation
# ============================================================
def build_reduced_var_garch_simulation_input(
    return_forecast_df,
    vol_forecast_df,
    date_col="date",
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred",
    actual_return_col="gold_ret_actual"
):
    """
    Construit le DataFrame d'entrée de la simulation Monte Carlo
    en alignant la moyenne et la volatilité prédites.
    """
    actual_col = actual_return_col if actual_return_col in return_forecast_df.columns else None

    return build_simulation_input_df(
        return_forecast_df=return_forecast_df,
        vol_forecast_df=vol_forecast_df,
        date_col=date_col,
        return_col=return_col,
        vol_col=vol_col,
        actual_return_col=actual_col
    )


# ============================================================
# 10) Simulation des chocs Student standardisés
# ============================================================
def simulate_reduced_var_garch_student_shocks(
    n_periods,
    n_simulations=5000,
    nu=8.0,
    random_state=None
):
    """
    Simule des chocs Student standardisés de variance 1.
    """
    return simulate_standardized_student_shocks(
        n_periods=n_periods,
        n_simulations=n_simulations,
        nu=nu,
        random_state=random_state
    )


# ============================================================
# 11) Simulation stochastique des rendements
# ============================================================
def simulate_reduced_var_garch_returns(
    simulation_input_df,
    nu,
    n_simulations=5000,
    random_state=None,
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred"
):
    """
    Simule les rendements sur le test selon :

        r_t^(b) = mu_t + sigma_t * z_t^(b)

    où z_t^(b) suit une Student-t standardisée.
    """
    return simulate_return_distribution(
        simulation_input_df=simulation_input_df,
        nu=nu,
        n_simulations=n_simulations,
        random_state=random_state,
        return_col=return_col,
        vol_col=vol_col
    )


# ============================================================
# 12) Résumé des rendements simulés
# ============================================================
def summarize_reduced_var_garch_returns(
    simulation_input_df,
    simulated_returns_matrix,
    date_col="date",
    return_col="gold_ret_pred",
    vol_col="garch_cond_vol_pred",
    actual_return_col="gold_ret_actual"
):
    """
    Résume la distribution simulée des rendements sur le test.
    """
    actual_col = actual_return_col if actual_return_col in simulation_input_df.columns else None

    return summarize_simulated_returns(
        simulation_input_df=simulation_input_df,
        simulated_returns_matrix=simulated_returns_matrix,
        date_col=date_col,
        return_col=return_col,
        vol_col=vol_col,
        actual_return_col=actual_col
    )


# ============================================================
# 13) Simulation des trajectoires de prix
# ============================================================
def simulate_reduced_var_garch_price_paths(
    price_df,
    simulation_input_df,
    simulated_returns_matrix,
    date_col="date",
    price_col="gold_price"
):
    """
    Reconstruit les trajectoires simulées de prix à partir des
    log-rendements simulés.
    """
    return simulate_price_paths_from_returns(
        price_df=price_df,
        simulation_input_df=simulation_input_df,
        simulated_returns_matrix=simulated_returns_matrix,
        date_col=date_col,
        price_col=price_col
    )


# ============================================================
# 14) Résumé des trajectoires simulées de prix
# ============================================================
def summarize_reduced_var_garch_prices(
    price_df,
    simulation_input_df,
    simulated_price_matrix,
    date_col="date",
    price_col="gold_price"
):
    """
    Résume les trajectoires simulées de prix :
    moyenne, médiane et quantiles.
    """
    return summarize_simulated_prices(
        price_df=price_df,
        simulation_input_df=simulation_input_df,
        simulated_price_matrix=simulated_price_matrix,
        date_col=date_col,
        price_col=price_col
    )


# ============================================================
# 15) Construction du DataFrame final de prédiction
# ============================================================
def build_reduced_var_garch_predicted_price_df(
    simulation_input_df,
    simulated_price_summary_df,
    simulated_price_matrix=None,
    prediction_mode="single_path",
    selected_path_index=0,
    date_col="date"
):
    """
    Construit le DataFrame final de prédiction des prix.

    Paramètres
    ----------
    simulation_input_df : pd.DataFrame
        DataFrame d'entrée de simulation contenant au minimum `date`.
    simulated_price_summary_df : pd.DataFrame
        Résumé des prix simulés contenant au minimum :
        - date
        - gold_price_actual
        - gold_price_sim_mean
        - gold_price_sim_median
    simulated_price_matrix : np.ndarray ou None
        Matrice des trajectoires simulées de prix. Obligatoire si
        `prediction_mode="single_path"`.
    prediction_mode : str
        Mode de construction de la prévision finale :
        - "single_path"
        - "mean"
        - "median"
    selected_path_index : int
        Index de la trajectoire retenue si `prediction_mode="single_path"`.
    date_col : str
        Nom de la colonne de date.

    Retour
    ------
    tuple
        (predicted_price_df, selected_simulation_index)
    """
    required_summary_cols = [
        date_col,
        "gold_price_actual",
        "gold_price_sim_mean",
        "gold_price_sim_median"
    ]
    missing_cols = [
        col for col in required_summary_cols
        if col not in simulated_price_summary_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans simulated_price_summary_df : {missing_cols}. "
            f"Colonnes disponibles : {list(simulated_price_summary_df.columns)}"
        )

    if date_col not in simulation_input_df.columns:
        raise ValueError(
            f"La colonne '{date_col}' est absente de simulation_input_df."
        )

    prediction_mode = str(prediction_mode).lower()

    summary_df = simulated_price_summary_df.copy()
    summary_df[date_col] = pd.to_datetime(summary_df[date_col])
    summary_df = summary_df.sort_values(date_col).reset_index(drop=True)

    selected_simulation_index = None

    if prediction_mode == "mean":
        predicted_price_df = summary_df[
            [date_col, "gold_price_actual", "gold_price_sim_mean"]
        ].copy()
        predicted_price_df = predicted_price_df.rename(
            columns={"gold_price_sim_mean": "gold_price_pred"}
        )

    elif prediction_mode == "median":
        predicted_price_df = summary_df[
            [date_col, "gold_price_actual", "gold_price_sim_median"]
        ].copy()
        predicted_price_df = predicted_price_df.rename(
            columns={"gold_price_sim_median": "gold_price_pred"}
        )

    elif prediction_mode == "single_path":
        if simulated_price_matrix is None:
            raise ValueError(
                "simulated_price_matrix est requis si prediction_mode='single_path'."
            )

        n_simulations = simulated_price_matrix.shape[1]
        selected_path_index = int(selected_path_index)

        if selected_path_index < 0 or selected_path_index >= n_simulations:
            raise ValueError(
                f"selected_path_index doit être compris entre 0 et {n_simulations - 1}."
            )

        selected_simulation_index = selected_path_index

        predicted_price_df = summary_df[[date_col, "gold_price_actual"]].copy()
        predicted_price_df["gold_price_pred"] = simulated_price_matrix[:, selected_path_index]

    else:
        raise ValueError(
            "prediction_mode doit valoir 'single_path', 'mean' ou 'median'."
        )

    predicted_price_df = predicted_price_df.sort_values(date_col).reset_index(drop=True)

    return predicted_price_df, selected_simulation_index


# ============================================================
# 16) Pipeline global réduit VAR + GARCH + Monte Carlo
# ============================================================
def run_reduced_var_garch_monte_carlo_pipeline(
    var_df=None,
    price_df=None,
    data_dir="data",
    train_start="2005-01-01",
    train_end="2020-12-31",
    test_start="2021-01-01",
    test_end="2021-12-31",
    window_years=7,
    step_years=2,
    max_lag=10,
    n_simulations=5000,
    random_state=42,
    prediction_mode="single_path",
    selected_path_index=0
):
    """
    Exécute un pipeline parallèle complet :
    - VAR réduit pour la moyenne
    - GARCH(1,1)-Student pour la variance
    - simulation Monte Carlo sur le test
    - reconstruction des prix prédits à partir :
      - d'une trajectoire simulée unique
      - ou d'un résumé de la distribution simulée

    Paramètres
    ----------
    var_df : pd.DataFrame ou None
        Dataset journalier du VAR. Si None, il est reconstruit.
    price_df : pd.DataFrame ou None
        Dataset des prix. Si None, il est reconstruit.
    data_dir : str
        Dossier des données.
    train_start : str
    train_end : str
    test_start : str
    test_end : str
    window_years : int
    step_years : int
    max_lag : int
    n_simulations : int
        Nombre de trajectoires Monte Carlo.
    random_state : int ou None
        Graine aléatoire.
    prediction_mode : str
        Mode de prédiction finale :
        - "single_path"
        - "mean"
        - "median"
    selected_path_index : int
        Index de la trajectoire retenue si `prediction_mode="single_path"`.

    Retour
    ------
    dict
        Dictionnaire contenant les objets principaux du pipeline.
    """
    if var_df is None:
        var_df = prepare_var_dataset(data_dir=data_dir)

    if price_df is None:
        price_df = build_reduced_var_garch_price_df(data_dir=data_dir)

    var_df, train_df, test_df = prepare_reduced_var_train_test_data(
        var_df=var_df,
        data_dir=data_dir,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    lag_summary_df, lag_criteria_dict, final_lag_dict = select_reduced_var_lag_on_train(
        train_df=train_df,
        window_years=window_years,
        step_years=step_years,
        max_lag=max_lag
    )

    final_lag = int(final_lag_dict["final_lag"])

    fitted_var_model = fit_reduced_var_on_train(
        train_df=train_df,
        lag_order=final_lag
    )

    return_forecast_df = forecast_reduced_var_dynamically_on_test(
        fitted_var_model=fitted_var_model,
        train_df=train_df,
        test_df=test_df
    )

    gold_resid_train_df = extract_reduced_var_gold_residuals(
        fitted_var_model=fitted_var_model
    )

    garch_result = fit_reduced_var_garch_student(
        gold_resid_train_df=gold_resid_train_df
    )

    vol_forecast_df = forecast_reduced_var_garch_volatility(
        garch_result=garch_result,
        gold_resid_train_df=gold_resid_train_df,
        return_forecast_df=return_forecast_df
    )

    simulation_input_df = build_reduced_var_garch_simulation_input(
        return_forecast_df=return_forecast_df,
        vol_forecast_df=vol_forecast_df
    )

    simulated_returns_matrix = simulate_reduced_var_garch_returns(
        simulation_input_df=simulation_input_df,
        nu=garch_result["params"]["nu"],
        n_simulations=n_simulations,
        random_state=random_state
    )

    simulated_returns_summary_df = summarize_reduced_var_garch_returns(
        simulation_input_df=simulation_input_df,
        simulated_returns_matrix=simulated_returns_matrix
    )

    simulated_price_matrix = simulate_reduced_var_garch_price_paths(
        price_df=price_df,
        simulation_input_df=simulation_input_df,
        simulated_returns_matrix=simulated_returns_matrix
    )

    simulated_price_summary_df = summarize_reduced_var_garch_prices(
        price_df=price_df,
        simulation_input_df=simulation_input_df,
        simulated_price_matrix=simulated_price_matrix
    )

    predicted_price_df, selected_simulation_index = build_reduced_var_garch_predicted_price_df(
        simulation_input_df=simulation_input_df,
        simulated_price_summary_df=simulated_price_summary_df,
        simulated_price_matrix=simulated_price_matrix,
        prediction_mode=prediction_mode,
        selected_path_index=selected_path_index
    )

    return {
        "var_df": var_df,
        "price_df": price_df,
        "train_df": train_df,
        "test_df": test_df,
        "lag_summary_df": lag_summary_df,
        "lag_criteria_dict": lag_criteria_dict,
        "final_lag_dict": final_lag_dict,
        "final_lag": final_lag,
        "fitted_var_model": fitted_var_model,
        "gold_resid_train_df": gold_resid_train_df,
        "garch_result": garch_result,
        "return_forecast_df": return_forecast_df,
        "vol_forecast_df": vol_forecast_df,
        "simulation_input_df": simulation_input_df,
        "simulated_returns_matrix": simulated_returns_matrix,
        "simulated_returns_summary_df": simulated_returns_summary_df,
        "simulated_price_matrix": simulated_price_matrix,
        "simulated_price_summary_df": simulated_price_summary_df,
        "selected_simulation_index": selected_simulation_index,
        "predicted_price_df": predicted_price_df
    }
