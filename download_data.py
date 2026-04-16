import os
import io
import requests
import pandas as pd
import yfinance as yf
import xlrd
from bs4 import BeautifulSoup
from datetime import datetime
# ============================================================
# Paramètres globaux
# ============================================================
FRED_API_KEY = "31da204ceab6c0f0deeb22ba69f9c488"

START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1) Téléchargement Yahoo Finance
# ============================================================
def download_yahoo_series():
    tickers = {
        "gold": "GC=F",
        "dxy": "DX-Y.NYB",
        "sp500": "^GSPC",
        "vix": "^VIX"
    }

    for name, ticker in tickers.items():
        print(f"Téléchargement Yahoo Finance : {name} ({ticker})")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            print(f"Attention : aucune donnée récupérée pour {ticker}")
            continue

        output_path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(output_path)
        print(f"Enregistré : {output_path}")

# ============================================================
# 2) CPI via FRED
# ============================================================
def download_cpi(api_key, start_date=START_DATE, end_date=END_DATE):
    """
    Télécharge le CPI depuis FRED et renvoie un DataFrame propre.
    """
    if not api_key:
        raise ValueError("La clé API FRED est vide ou manquante.")

    print("Téléchargement CPI via FRED")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }

    r = requests.get(url, params=params, timeout=30)
    print("URL CPI appelée :", r.url)
    r.raise_for_status()

    data = r.json()
    observations = data.get("observations", [])

    if not observations:
        raise ValueError("Aucune observation CPI récupérée depuis FRED.")

    cpi = pd.DataFrame(observations)[["date", "value"]]
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi["value"] = pd.to_numeric(cpi["value"], errors="coerce")
    cpi = cpi.rename(columns={"value": "cpi"})

    output_path = os.path.join(DATA_DIR, "cpi.csv")
    cpi.to_csv(output_path, index=False)
    print(f"Enregistré : {output_path}")

    return cpi

# ============================================================
# 3) GPR depuis le site de Matteo Iacoviello
# ============================================================
def download_gpr():
    print("Téléchargement GPR")

    page_url = "https://www.matteoiacoviello.com/gpr.htm"
    page = requests.get(page_url, timeout=30)
    page.raise_for_status()

    soup = BeautifulSoup(page.text, "html.parser")

    excel_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        href_lower = href.lower()
        if href_lower.endswith(".xls") or href_lower.endswith(".xlsx"):
            if href.startswith("http"):
                excel_links.append(href)
            else:
                base = "https://www.matteoiacoviello.com/"
                excel_links.append(base + href.lstrip("/"))

    if not excel_links:
        raise ValueError("Aucun lien Excel GPR trouvé sur la page.")

    gpr_file_url = excel_links[0]
    print(f"Fichier GPR détecté : {gpr_file_url}")

    file_resp = requests.get(gpr_file_url, timeout=60)
    file_resp.raise_for_status()

    ext = ".xlsx" if gpr_file_url.lower().endswith(".xlsx") else ".xls"
    raw_path = os.path.join(DATA_DIR, f"gpr_raw{ext}")

    with open(raw_path, "wb") as f:
        f.write(file_resp.content)

    print(f"Fichier brut enregistré : {raw_path}")
    return raw_path

# ============================================================
# 4) Fonction globale
# ============================================================
def download_all():
    download_yahoo_series()
    download_cpi(FRED_API_KEY)
    download_gpr()
    print("Téléchargement terminé.")

# ============================================================
# 5) Lancement direct
# ============================================================
if __name__ == "__main__":
    download_all()