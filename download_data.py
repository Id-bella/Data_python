"""
download_data.py
================
Téléchargement et sauvegarde des données brutes pour le projet
"Prédiction du prix de l'or via un modèle VAR".

Sources :
    - Yahoo Finance  : Or (GC=F), DXY, S&P 500, VIX
    - FRED (St. Louis Fed) : CPI (CPIAUCSL)
    - Matteo Iacoviello    : Geopolitical Risk Index (GPR)

Chaque fonction retourne un DataFrame propre et sauvegarde un CSV/Excel
dans DATA_DIR. Un fallback local est prévu pour chaque source en cas
d'indisponibilité de l'API ou du site.
"""

import os
import requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from datetime import datetime

# ============================================================
# Paramètres globaux
# ============================================================

START_DATE = "2005-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
DATA_DIR   = "data"

FRED_API_KEY = "31da204ceab6c0f0deeb22ba69f9c488"

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Adresses & chemins de sauvegarde
# ============================================================

# Yahoo Finance — tickers
YAHOO_TICKERS = {
    "gold":  "GC=F",       # Or (contrat futures front-month)
    "dxy":   "DX-Y.NYB",   # US Dollar Index
    "sp500": "^GSPC",      # S&P 500
    "vix":   "^VIX",       # CBOE Volatility Index
}

# FRED — CPI
FRED_URL      = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES   = "CPIAUCSL"
CPI_SAVE_PATH = os.path.join(DATA_DIR, "cpi.csv")
CPI_FALLBACK  = os.path.join(DATA_DIR, "cpi_backup.csv")   # sauvegarde locale

# GPR — Caldara & Iacoviello
GPR_PAGE_URL  = "https://www.matteoiacoviello.com/gpr.htm"
GPR_SAVE_PATH = os.path.join(DATA_DIR, "gpr_raw")          # ext ajoutée dynamiquement
GPR_FALLBACK  = os.path.join(DATA_DIR, "gpr_backup.xls")   # sauvegarde locale


# ============================================================
# 1. Yahoo Finance — Or, DXY, S&P 500, VIX
# ============================================================

def download_yahoo_series(
    tickers: dict = YAHOO_TICKERS,
    start: str = START_DATE,
    end:   str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Télécharge les séries de prix depuis Yahoo Finance.

    Paramètres
    ----------
    tickers : dict  {nom: ticker_yf}
    start   : str   date de début (YYYY-MM-DD)
    end     : str   date de fin   (YYYY-MM-DD)

    Retourne
    --------
    dict {nom: DataFrame} avec colonne 'Close' et index DatetimeIndex.
    Chaque DataFrame est aussi sauvegardé dans DATA_DIR/{nom}.csv.
    """
    results = {}

    for name, ticker in tickers.items():
        save_path = os.path.join(DATA_DIR, f"{name}.csv")
        print(f"  [{name}] Téléchargement Yahoo Finance ({ticker})...")

        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                raise ValueError(f"Aucune donnée retournée pour {ticker}.")

            df.to_csv(save_path)
            print(f"  [{name}] ✓ {len(df)} observations → {save_path}")
            results[name] = df

        except Exception as e:
            print(f"  [{name}] ✗ Échec ({e})")
            # Fallback local
            if os.path.exists(save_path):
                print(f"  [{name}] → Utilisation du fichier local : {save_path}")
                results[name] = pd.read_csv(save_path, index_col=0, parse_dates=True)
            else:
                print(f"  [{name}] → Aucun fichier local disponible. Ignoré.")

    return results


# ============================================================
# 2. FRED — CPI (Consumer Price Index, CPIAUCSL)
# ============================================================

def download_cpi(
    api_key:    str = FRED_API_KEY,
    start_date: str = START_DATE,
    end_date:   str = END_DATE,
    save_path:  str = CPI_SAVE_PATH,
    fallback:   str = CPI_FALLBACK,
) -> pd.DataFrame:
    """
    Télécharge l'indice des prix à la consommation (CPI) depuis l'API FRED
    de la Réserve Fédérale de St. Louis (série CPIAUCSL, mensuelle).

    Paramètres
    ----------
    api_key    : str  clé API FRED (gratuite sur fred.stlouisfed.org)
    start_date : str  date de début (YYYY-MM-DD)
    end_date   : str  date de fin   (YYYY-MM-DD)
    save_path  : str  chemin de sauvegarde CSV
    fallback   : str  chemin du fichier de secours local

    Retourne
    --------
    DataFrame avec colonnes ['date', 'cpi'], trié par date.
    """
    print(f"  [cpi] Téléchargement FRED (série {FRED_SERIES})...")

    try:
        if not api_key:
            raise ValueError("Clé API FRED manquante.")

        params = {
            "series_id":         FRED_SERIES,
            "api_key":           api_key,
            "file_type":         "json",
            "observation_start": start_date,
            "observation_end":   end_date,
        }

        r = requests.get(FRED_URL, params=params, timeout=30)
        r.raise_for_status()

        observations = r.json().get("observations", [])
        if not observations:
            raise ValueError("Aucune observation retournée par FRED.")

        cpi = (
            pd.DataFrame(observations)[["date", "value"]]
            .rename(columns={"value": "cpi"})
            .assign(
                date=lambda d: pd.to_datetime(d["date"]),
                cpi =lambda d: pd.to_numeric(d["cpi"], errors="coerce"),
            )
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )

        cpi.to_csv(save_path, index=False)
        print(f"  [cpi] ✓ {len(cpi)} observations → {save_path}")
        return cpi

    except Exception as e:
        print(f"  [cpi] ✗ Échec ({e})")
        # Fallback local
        for path in (save_path, fallback):
            if os.path.exists(path):
                print(f"  [cpi] → Utilisation du fichier local : {path}")
                return pd.read_csv(path, parse_dates=["date"])
        raise RuntimeError("CPI indisponible et aucun fichier local trouvé.")


# ============================================================
# 3. GPR — Geopolitical Risk Index (Caldara & Iacoviello)
# ============================================================

def download_gpr(
    page_url:  str = GPR_PAGE_URL,
    save_path: str = GPR_SAVE_PATH,
    fallback:  str = GPR_FALLBACK,
) -> str:
    """
    Télécharge le fichier Excel du Geopolitical Risk Index (GPR global)
    depuis le site de Matteo Iacoviello en scrapant la page pour trouver
    le lien de téléchargement.

    Paramètres
    ----------
    page_url  : str  URL de la page web contenant le lien Excel
    save_path : str  préfixe du fichier sauvegardé (sans extension)
    fallback  : str  chemin du fichier de secours local

    Retourne
    --------
    str : chemin vers le fichier Excel brut téléchargé (ou fallback).
    """
    print("  [gpr] Scraping de la page Iacoviello...")

    try:
        page  = requests.get(page_url, timeout=30)
        page.raise_for_status()
        soup  = BeautifulSoup(page.text, "html.parser")

        # Recherche des liens Excel sur la page
        excel_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.lower().endswith((".xls", ".xlsx")):
                if not href.startswith("http"):
                    href = "https://www.matteoiacoviello.com/" + href.lstrip("/")
                excel_links.append(href)

        if not excel_links:
            raise ValueError("Aucun lien Excel trouvé sur la page GPR.")

        gpr_url = excel_links[0]
        print(f"  [gpr] Fichier détecté : {gpr_url}")

        resp = requests.get(gpr_url, timeout=60)
        resp.raise_for_status()

        ext      = ".xlsx" if gpr_url.lower().endswith(".xlsx") else ".xls"
        out_path = save_path + ext

        with open(out_path, "wb") as f:
            f.write(resp.content)

        print(f"  [gpr] ✓ Fichier brut sauvegardé → {out_path}")
        return out_path

    except Exception as e:
        print(f"  [gpr] ✗ Échec ({e})")
        # Fallback local
        for path in (save_path + ".xls", save_path + ".xlsx", fallback):
            if os.path.exists(path):
                print(f"  [gpr] → Utilisation du fichier local : {path}")
                return path
        raise RuntimeError("GPR indisponible et aucun fichier local trouvé.")


# ============================================================
# 4. Téléchargement global
# ============================================================

def download_all() -> None:
    """
    Lance le téléchargement de toutes les sources dans l'ordre :
        1. Yahoo Finance (Or, DXY, S&P 500, VIX)
        2. FRED          (CPI)
        3. Iacoviello    (GPR)
    """
    print("=" * 55)
    print("  Téléchargement des données")
    print(f"  Période : {START_DATE} → {END_DATE}")
    print("=" * 55)

    print("\n── 1/3  Yahoo Finance ──────────────────────────────")
    download_yahoo_series()

    print("\n── 2/3  FRED — CPI ─────────────────────────────────")
    download_cpi()

    print("\n── 3/3  GPR — Iacoviello ───────────────────────────")
    download_gpr()

    print("\n" + "=" * 55)
    print("  ✓ Tous les fichiers sont dans le dossier data/")
    print("=" * 55)


# ============================================================
# 5. Lancement direct
# ============================================================

if __name__ == "__main__":
    download_all()