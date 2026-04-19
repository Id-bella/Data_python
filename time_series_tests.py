import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch


# ============================================================
# 1) Vérification minimale du DataFrame VAR
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
# 2) Test ADF sur une seule série
# ============================================================
def adf_test_single_series(df, series_col, regression="c", autolag="AIC"):
    """
    Effectue le test ADF sur une seule série.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant la série.
    series_col : str
        Nom de la colonne à tester.
    regression : str
        Type de régression dans le test ADF :
        - "c"  : constante
        - "ct" : constante + tendance
        - "ctt": constante + tendance + tendance quadratique
        - "n"  : sans constante ni tendance
    autolag : str
        Méthode de sélection automatique du nombre de retards.
        En général : "AIC" ou "BIC".

    Retour
    ------
    dict
        Dictionnaire contenant les résultats du test ADF.
    """
    if series_col not in df.columns:
        raise ValueError(f"La colonne '{series_col}' est absente du DataFrame.")

    series = df[series_col].dropna()

    result = adfuller(series, regression=regression, autolag=autolag)

    output = {
        "series": series_col,
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lags": result[2],
        "n_obs": result[3],
        "critical_value_1pct": result[4]["1%"],
        "critical_value_5pct": result[4]["5%"],
        "critical_value_10pct": result[4]["10%"],
        "stationary_5pct": result[1] < 0.05
    }

    return output


# ============================================================
# 3) Test ADF sur toutes les séries du VAR
# ============================================================
def adf_test_all_var_series(var_df, regression="c", autolag="AIC"):
    """
    Effectue le test ADF sur toutes les séries du VAR.

    Paramètres
    ----------
    var_df : pd.DataFrame
        DataFrame contenant :
        - date
        - gold_ret
        - dxy_ret
        - sp500_ret
        - vix_ret
    regression : str
        Type de régression utilisé dans le test ADF.
    autolag : str
        Critère utilisé pour choisir automatiquement le nombre de lags.

    Retour
    ------
    pd.DataFrame
        Tableau de résultats du test ADF pour chaque série.
    """
    check_var_dataframe(var_df)

    series_list = ["gold_ret", "dxy_ret", "sp500_ret", "vix_ret"]

    results = []
    for col in series_list:
        test_result = adf_test_single_series(
            df=var_df,
            series_col=col,
            regression=regression,
            autolag=autolag
        )
        results.append(test_result)

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# 4) Interprétation textuelle simple du test ADF
# ============================================================
def print_adf_conclusion(adf_result):
    """
    Affiche une conclusion simple à partir d'un résultat ADF
    obtenu avec adf_test_single_series.

    Paramètres
    ----------
    adf_result : dict
        Résultat renvoyé par adf_test_single_series.

    Retour
    ------
    None
    """
    print(f"Série testée : {adf_result['series']}")
    print(f"Statistique ADF : {adf_result['adf_statistic']:.6f}")
    print(f"p-value : {adf_result['p_value']:.6f}")
    print(f"Lags utilisés : {adf_result['used_lags']}")
    print(f"Nombre d'observations : {adf_result['n_obs']}")
    print("Valeurs critiques :")
    print(f"  1%  : {adf_result['critical_value_1pct']:.6f}")
    print(f"  5%  : {adf_result['critical_value_5pct']:.6f}")
    print(f"  10% : {adf_result['critical_value_10pct']:.6f}")

    if adf_result["stationary_5pct"]:
        print("Conclusion : on rejette l'hypothèse de racine unitaire au seuil de 5%.")
        print("La série peut être considérée comme stationnaire.")
    else:
        print("Conclusion : on ne rejette pas l'hypothèse de racine unitaire au seuil de 5%.")
        print("La série ne peut pas être considérée comme stationnaire.")

import numpy as np
import pandas as pd


# ============================================================
# 1) Vérification de la stabilité du VAR
# ============================================================
def get_var_roots(fitted_model):
    """
    Récupère les racines du VAR.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.

    Retour
    ------
    np.ndarray
        Tableau des racines du modèle.
    """
    return fitted_model.roots


def build_var_stability_table(fitted_model):
    """
    Construit un tableau résumant les racines du VAR
    et leur module.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.

    Retour
    ------
    pd.DataFrame
        Tableau avec :
        - root
        - modulus
        - outside_unit_circle
    """
    roots = fitted_model.roots

    stability_df = pd.DataFrame({
        "root": roots,
        "modulus": np.abs(roots),
        "outside_unit_circle": np.abs(roots) > 1
    })

    return stability_df


def is_var_stable(fitted_model):
    """
    Vérifie si le VAR est stable.

    Condition utilisée par statsmodels :
    un VAR est stable si toutes les racines sont à l'extérieur
    du cercle unité.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.

    Retour
    ------
    bool
        True si le modèle est stable, False sinon.
    """
    return fitted_model.is_stable()


def print_var_stability_conclusion(fitted_model):
    """
    Affiche une conclusion simple sur la stabilité du VAR.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults

    Retour
    ------
    None
    """
    stable = is_var_stable(fitted_model)

    print("Test de stabilité du VAR")
    print("------------------------")
    print(f"Le modèle est-il stable ? {stable}")

    if stable:
        print("Conclusion : toutes les racines sont à l'extérieur du cercle unité.")
        print("Le VAR peut être considéré comme stable.")
    else:
        print("Conclusion : au moins une racine ne satisfait pas la condition de stabilité.")
        print("Le VAR ne peut pas être considéré comme stable.")


# ============================================================
# 2) Test d'autocorrélation des résidus
# ============================================================
def residual_serial_correlation_test(fitted_model, nlags=12):
    """
    Effectue le test de Portmanteau sur l'autocorrélation des résidus.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.
    nlags : int
        Nombre de retards utilisés pour le test.

    Retour
    ------
    results
        Objet statsmodels contenant les résultats du test.
    """
    return fitted_model.test_whiteness(nlags=nlags)


def get_residual_serial_correlation_summary(fitted_model, nlags=12):
    """
    Résume les résultats du test d'autocorrélation des résidus.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.
    nlags : int
        Nombre de lags du test.

    Retour
    ------
    dict
        Résumé du test.
    """
    test_result = residual_serial_correlation_test(fitted_model, nlags=nlags)

    summary = {
        "test_statistic": test_result.test_statistic,
        "critical_value": test_result.crit_value,
        "p_value": test_result.pvalue,
        "df": test_result.df,
        "nlags": nlags,
        "reject_h0_5pct": test_result.pvalue < 0.05,
    }

    return summary


def print_residual_serial_correlation_conclusion(fitted_model, nlags=12):
    """
    Affiche une conclusion simple sur l'autocorrélation des résidus.

    Hypothèse nulle :
    absence d'autocorrélation des résidus jusqu'au lag choisi.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.
    nlags : int
        Nombre de lags du test.

    Retour
    ------
    None
    """
    res = get_residual_serial_correlation_summary(fitted_model, nlags=nlags)

    print("Test d'autocorrélation des résidus (Portmanteau)")
    print("------------------------------------------------")
    print(f"Nombre de lags testés : {res['nlags']}")
    print(f"Statistique de test : {res['test_statistic']:.6f}")
    print(f"Valeur critique : {res['critical_value']:.6f}")
    print(f"p-value : {res['p_value']:.6f}")
    print(f"Degrés de liberté : {res['df']}")

    if res["reject_h0_5pct"]:
        print("Conclusion : on rejette l'hypothèse nulle au seuil de 5%.")
        print("Les résidus présentent encore de l'autocorrélation.")
    else:
        print("Conclusion : on ne rejette pas l'hypothèse nulle au seuil de 5%.")
        print("Les résidus peuvent être considérés comme non autocorrélés.")


# ============================================================
# 3) Tests de causalité de Granger
# ============================================================
def granger_causality_test_single(fitted_model, caused, causing):
    """
    Effectue un test de causalité de Granger dans le VAR estimé.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.
    caused : str
        Variable expliquée.
    causing : str or list
        Variable(s) testée(s) comme cause(s) au sens de Granger.

    Retour
    ------
    results
        Objet statsmodels du test.
    """
    return fitted_model.test_causality(caused=caused, causing=causing, kind="f")


def get_granger_test_summary(fitted_model, caused, causing):
    """
    Résume les résultats d'un test de causalité de Granger.

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.
    caused : str
        Variable expliquée.
    causing : str or list
        Variable(s) explicative(s) testée(s).

    Retour
    ------
    dict
        Résumé du test.
    """
    test_result = granger_causality_test_single(fitted_model, caused=caused, causing=causing)

    summary = {
        "caused": caused,
        "causing": causing if isinstance(causing, str) else ", ".join(causing),
        "test_statistic": test_result.test_statistic,
        "p_value": test_result.pvalue,
        "df": test_result.df,
        "reject_h0_5pct": test_result.pvalue < 0.05
    }

    return summary


def granger_tests_for_gold(fitted_model):
    """
    Effectue les trois tests principaux pour le projet :

    - DXY -> gold
    - S&P 500 -> gold
    - VIX -> gold

    Paramètres
    ----------
    fitted_model : statsmodels VARResults
        Modèle VAR estimé.

    Retour
    ------
    pd.DataFrame
        Tableau récapitulatif des tests de causalité de Granger
        vers l'or.
    """
    tests = [
        get_granger_test_summary(fitted_model, caused="gold_ret", causing="dxy_ret"),
        get_granger_test_summary(fitted_model, caused="gold_ret", causing="sp500_ret"),
        get_granger_test_summary(fitted_model, caused="gold_ret", causing="vix_ret"),
    ]

    return pd.DataFrame(tests)


def print_granger_conclusion(summary_dict):
    """
    Affiche une conclusion simple pour un test de causalité de Granger.

    Paramètres
    ----------
    summary_dict : dict
        Résultat produit par get_granger_test_summary.

    Retour
    ------
    None
    """
    print("Test de causalité de Granger")
    print("----------------------------")
    print(f"Variable expliquée : {summary_dict['caused']}")
    print(f"Variable testée : {summary_dict['causing']}")
    print(f"Statistique de test : {summary_dict['test_statistic']:.6f}")
    print(f"p-value : {summary_dict['p_value']:.6f}")
    print(f"Degrés de liberté : {summary_dict['df']}")

    if summary_dict["reject_h0_5pct"]:
        print("Conclusion : on rejette l'hypothèse nulle au seuil de 5%.")
        print("La variable testée Granger-cause la variable expliquée.")
    else:
        print("Conclusion : on ne rejette pas l'hypothèse nulle au seuil de 5%.")
        print("Pas d'évidence de causalité de Granger au sens prédictif.")


# ============================================================
# 4) Vérification du DataFrame des résidus VAR de l'or
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
# 5) Test ARCH-LM sur les résidus de l'or
# ============================================================
def arch_lm_test_gold_residuals(gold_resid_df, nlags=12):
    """
    Effectue un test ARCH-LM sur les résidus de l'équation de l'or.

    Paramètres
    ----------
    gold_resid_df : pd.DataFrame
        DataFrame contenant :
        - date
        - gold_var_resid
    nlags : int
        Nombre de retards utilisés dans le test.

    Retour
    ------
    dict
        Résultats du test ARCH-LM.
    """
    check_gold_residuals_dataframe(gold_resid_df)

    df = gold_resid_df.copy()
    resid = df["gold_var_resid"].dropna()

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=nlags)

    return {
        "nlags": nlags,
        "lm_statistic": lm_stat,
        "lm_pvalue": lm_pvalue,
        "f_statistic": f_stat,
        "f_pvalue": f_pvalue,
        "reject_h0_5pct": lm_pvalue < 0.05
    }


# ============================================================
# 6) Conclusion textuelle du test ARCH-LM
# ============================================================
def print_arch_lm_conclusion(gold_resid_df, nlags=12):
    """
    Affiche une conclusion simple du test ARCH-LM.

    Hypothèse nulle :
    absence d'effet ARCH.
    """
    res = arch_lm_test_gold_residuals(gold_resid_df, nlags=nlags)

    print("Test ARCH-LM sur les résidus de l'or")
    print("------------------------------------")
    print(f"Nombre de lags testés : {res['nlags']}")
    print(f"LM statistic : {res['lm_statistic']:.6f}")
    print(f"LM p-value : {res['lm_pvalue']:.6f}")
    print(f"F statistic : {res['f_statistic']:.6f}")
    print(f"F p-value : {res['f_pvalue']:.6f}")

    if res["reject_h0_5pct"]:
        print("Conclusion : on rejette l'hypothèse nulle au seuil de 5%.")
        print("Il existe un effet ARCH dans les résidus.")
    else:
        print("Conclusion : on ne rejette pas l'hypothèse nulle au seuil de 5%.")
        print("Pas d'évidence claire d'effet ARCH dans les résidus.")
