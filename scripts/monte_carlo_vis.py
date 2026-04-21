import pandas as pd
import matplotlib.pyplot as plt


from .data_vis import format_date_axis


# ============================================================
# 1) Vérification du DataFrame final de prédiction
# ============================================================
def check_reduced_var_garch_predicted_price_dataframe(predicted_price_df):
    """
    Vérifie que le DataFrame final de prédiction contient bien
    les colonnes attendues.

    Paramètres
    ----------
    predicted_price_df : pd.DataFrame
        DataFrame contenant au minimum :
        - date
        - gold_price_actual
        - gold_price_pred
    """
    required_columns = ["date", "gold_price_actual", "gold_price_pred"]

    missing_cols = [col for col in required_columns if col not in predicted_price_df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans predicted_price_df : {missing_cols}. "
            f"Colonnes disponibles : {list(predicted_price_df.columns)}"
        )


# ============================================================
# 2) Vérification du DataFrame résumé Monte Carlo
# ============================================================
def check_reduced_var_garch_monte_carlo_price_dataframe(
    simulated_price_summary_df,
    require_median=True
):
    """
    Vérifie que le DataFrame résumé Monte Carlo contient bien
    les colonnes attendues.
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


# ============================================================
# 3) Tracé final prix observé vs prix prédit
# ============================================================
def plot_gold_price_reduced_var_garch_prediction_test(
    predicted_price_df,
    figsize=(14, 6),
    major="year",
    date_format="%Y"
):
    """
    Trace le prix observé et le prix prédit final du nouveau pipeline.
    """
    check_reduced_var_garch_predicted_price_dataframe(predicted_price_df)

    df = predicted_price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["gold_price_actual"] = pd.to_numeric(df["gold_price_actual"], errors="coerce")
    df["gold_price_pred"] = pd.to_numeric(df["gold_price_pred"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    if df[["gold_price_actual", "gold_price_pred"]].isna().any().any():
        raise ValueError("Des valeurs manquantes ont été détectées dans predicted_price_df.")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        df["date"],
        df["gold_price_actual"],
        label="Prix observé de l'or",
        color="black",
        linewidth=2
    )

    ax.plot(
        df["date"],
        df["gold_price_pred"],
        label="Prix prédit de l'or",
        color="#D35400",
        linestyle="--",
        linewidth=2
    )

    ax.set_title("Prix observé vs prix prédit - VAR réduit + GARCH gaussien")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    ax.legend()
    ax.grid(True)

    format_date_axis(ax, major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Tracé Monte Carlo complet des prix sur le test
# ============================================================
def plot_gold_price_reduced_var_garch_monte_carlo_test(
    simulated_price_summary_df,
    predicted_price_df=None,
    figsize=(14, 6),
    show_median=True,
    show_band=True,
    major="year",
    date_format="%Y"
):
    """
    Trace sur la période de test :
    - le prix observé de l'or
    - la moyenne des prix simulés
    - éventuellement la médiane des prix simulés
    - éventuellement la courbe finale de prévision retenue
    - une bande d'incertitude entre q05 et q95
    """
    check_reduced_var_garch_monte_carlo_price_dataframe(
        simulated_price_summary_df=simulated_price_summary_df,
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

    if show_band:
        ax.fill_between(
            df["date"],
            df["gold_price_sim_q05"],
            df["gold_price_sim_q95"],
            color="#D4AF37",
            alpha=0.15,
            label="Bande Monte Carlo 5% - 95%"
        )

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
        color="#D35400",
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

    if predicted_price_df is not None:
        check_reduced_var_garch_predicted_price_dataframe(predicted_price_df)

        pred_df = predicted_price_df.copy()
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df["gold_price_pred"] = pd.to_numeric(pred_df["gold_price_pred"], errors="coerce")
        pred_df = pred_df.sort_values("date").reset_index(drop=True)

        ax.plot(
            pred_df["date"],
            pred_df["gold_price_pred"],
            label="Trajectoire simulée retenue",
            color="#8E44AD",
            linewidth=2
        )

    ax.set_title("Prix observé vs simulation stochastique VAR réduit + GARCH gaussien")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    ax.legend()
    ax.grid(True)

    format_date_axis(ax, major=major, date_format=date_format)

    plt.tight_layout()
    plt.show()
