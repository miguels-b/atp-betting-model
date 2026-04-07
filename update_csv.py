"""
=============================================================================
 UPDATE_CSV.PY — Construye/actualiza Tennis.csv
 
 Modos:
   --build          Construye Tennis.csv desde 0 descargando todos los años
   --update         Actualiza con partidos nuevos (default si CSV existe)
   --dry-run        Muestra qué haría sin modificar
   --from-year=2013 Año de inicio para --build (default: 2013)
   --to-year=2026   Año final (default: año actual)
   --only-tennis-data  No usar OddsPortal como complemento
 
 Fuentes:
   1. tennis-data.co.uk — Datos completos (rankings, sets, odds)
   2. OddsPortal — Complemento para torneos en curso
 
 Uso:
   python update_csv.py --build                    # construir desde 0
   python update_csv.py --build --from-year=2015   # desde 2015
   python update_csv.py                            # actualizar existente
   python update_csv.py --dry-run                  # preview sin guardar
 
 Requisitos:
   pip install pandas requests openpyxl
=============================================================================
"""
import os
import sys
import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
PROJECT_DIR = os.path.abspath(".")
MASTER_CSV = os.path.join(PROJECT_DIR, "Tennis.csv")
DATE_FORMAT = "%Y-%m-%d"

TENNIS_DATA_BASE = "http://www.tennis-data.co.uk"
DEFAULT_FROM_YEAR = 2013
DEFAULT_TO_YEAR = datetime.now().year


# =====================================================================
# DESCARGA DE TENNIS-DATA.CO.UK
# =====================================================================

def download_year(year):
    """Descarga archivo consolidado anual. Intenta xlsx, xls."""
    urls = [
        (f"{TENNIS_DATA_BASE}/{year}/{year}.xlsx", "xlsx"),
        (f"{TENNIS_DATA_BASE}/{year}/{year}.xls",  "xls"),
    ]
    print(f"    {year}: descargando...", end=" ", flush=True)
    
    for url, fmt in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200 or len(resp.content) < 500:
                continue
            
            engine = "openpyxl" if fmt == "xlsx" else None
            df = pd.read_excel(BytesIO(resp.content), engine=engine, header=None)
            
            # Buscar la fila que contiene "Winner" (por si hay filas vacías encima)
            header_row = None
            for i, row in df.iterrows():
                if any(str(v).strip() == "Winner" for v in row.values):
                    header_row = i
                    break
            
            if header_row is None:
                print(f"[{fmt}: sin columna Winner, cols={list(df.iloc[0])}]", end=" ")
                continue
            
            # Re-leer con el header correcto
            df = pd.read_excel(BytesIO(resp.content), engine=engine, header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.dropna(how="all")
            
            if len(df) > 0:
                print(f"OK ({len(df)} partidos, {fmt})")
                return df
        except Exception as e:
            print(f"[error {fmt}: {e}]", end=" ")
            continue
    
    print("no disponible")
    return None


def normalize_dataframe(df):
    """Normaliza fechas a yyyy-mm-dd, limpia columnas."""
    # Encontrar columna de fecha
    date_col = None
    for col in df.columns:
        if col.strip().lower() == "date":
            date_col = col
            break
    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    
    # Convertir fecha y formatear como string uniforme
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Winner"]).copy()
    df["Date"] = df["Date"].dt.strftime(DATE_FORMAT)
    
    for col in ["WRank", "LRank", "WPts", "LPts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


# =====================================================================
# BUILD: Construir CSV desde cero
# =====================================================================

def build_csv(from_year, to_year, dry_run=False):
    print(f"\n  Descargando {from_year}-{to_year} de tennis-data.co.uk...\n")
    
    frames = []
    for year in range(from_year, to_year + 1):
        df = download_year(year)
        if df is not None:
            df = normalize_dataframe(df)
            frames.append(df)
        time.sleep(0.5)
    
    if not frames:
        print("\n  ERROR: No se pudo descargar nada")
        return
    
    combined = pd.concat(frames, ignore_index=True)
    combined["_sort"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    
    n_before = len(combined)
    combined = combined.drop_duplicates(
        subset=["Date", "Winner", "Loser", "Tournament"], keep="first"
    ).reset_index(drop=True)
    
    print(f"\n  {'─'*50}")
    print(f"  Total: {len(combined):,} partidos")
    print(f"  Duplicados eliminados: {n_before - len(combined)}")
    print(f"  Rango: {combined['Date'].iloc[0]} → {combined['Date'].iloc[-1]}")
    
    if dry_run:
        print(f"\n  [DRY RUN] No guardado.")
        print(combined[["Date","Tournament","Winner","Loser"]].tail(10).to_string(index=False))
    else:
        combined.to_csv(MASTER_CSV, index=False, encoding="utf-8")
        print(f"\n  ✓ Guardado: {MASTER_CSV} ({len(combined):,} partidos)")


# =====================================================================
# UPDATE: Añadir partidos nuevos al CSV existente
# =====================================================================

def load_master():
    if not os.path.exists(MASTER_CSV):
        return None, None
    df = pd.read_csv(MASTER_CSV, encoding="utf-8", low_memory=False)
    df["_dp"] = pd.to_datetime(df["Date"], errors="coerce")
    last = df["_dp"].max()
    bad = df["_dp"].isna().sum()
    print(f"  CSV: {len(df):,} partidos, último: {last.strftime('%Y-%m-%d') if pd.notna(last) else '???'}")
    if bad > 0:
        print(f"  ⚠ {bad} filas con fecha inválida")
    return df, last


def update_csv(dry_run=False, only_td=False):
    df_ex, last_date = load_master()
    if df_ex is None:
        print("  CSV no encontrado. Usa --build"); return
    if pd.isna(last_date):
        print("  Fechas corruptas. Usa --build para reconstruir."); return
    
    gap = (datetime.now() - last_date).days
    print(f"  Días sin actualizar: {gap}")
    if gap <= 0:
        print("\n  ✓ Ya al día."); return
    
    new_parts = []
    
    # ── tennis-data.co.uk ──
    print(f"\n{'─'*55}\n  tennis-data.co.uk\n{'─'*55}")
    for year in sorted(set([last_date.year, datetime.now().year])):
        df_y = download_year(year)
        if df_y is None: continue
        df_y = normalize_dataframe(df_y)
        df_y["_dp"] = pd.to_datetime(df_y["Date"], errors="coerce")
        new = df_y[df_y["_dp"] > last_date].drop(columns=["_dp"]).copy()
        if len(new) > 0:
            print(f"    → {len(new)} nuevos de {year}")
            new_parts.append(("tennis-data", new))
    
    # Calcular hasta dónde cubrió tennis-data
    td_last = last_date
    for _, df_s in new_parts:
        d = pd.to_datetime(df_s["Date"], errors="coerce").max()
        if pd.notna(d) and d > td_last: td_last = d
    td_gap = (datetime.now() - td_last).days
    
    # ── OddsPortal ──
    if not only_td and td_gap > 2:
        print(f"\n{'─'*55}\n  OddsPortal (últimos {td_gap} días)\n{'─'*55}")
        df_op = _scrape_oddsportal_results(td_last)
        if not df_op.empty:
            new_parts.append(("oddsportal", df_op))
    
    if not new_parts:
        print("\n  Sin partidos nuevos. tennis-data se actualiza al final de cada torneo.")
        return
    
    # ── Deduplicar y guardar ──
    print(f"\n{'─'*55}\n  Resultado\n{'─'*55}")
    
    ex_keys = set()
    for _, r in df_ex.iterrows():
        ex_keys.add(f"{r['Date']}|{r.get('Winner','')}|{r.get('Loser','')}")
    
    to_add = []
    for src, df_s in new_parts:
        mask = [f"{r['Date']}|{r.get('Winner','')}|{r.get('Loser','')}" not in ex_keys 
                for _, r in df_s.iterrows()]
        df_new = df_s[mask].copy()
        
        # Alinear columnas
        for c in df_ex.columns:
            if c not in df_new.columns: df_new[c] = np.nan
        extra = [c for c in df_new.columns if c not in df_ex.columns]
        df_new = df_new.drop(columns=extra, errors="ignore")
        df_new = df_new[df_ex.columns]
        
        print(f"  {src}: {len(df_new)} nuevos")
        if len(df_new) > 0:
            to_add.append(df_new)
    
    total = sum(len(d) for d in to_add)
    if total == 0:
        print("  Nada nuevo tras deduplicar."); return
    
    if dry_run:
        print(f"\n  [DRY RUN] Se añadirían {total} partidos:")
        for d in to_add:
            cols = [c for c in ["Date","Tournament","Winner","Loser"] if c in d.columns]
            print(d[cols].head(15).to_string(index=False))
        return
    
    df_ex = df_ex.drop(columns=["_dp"], errors="ignore")
    result = pd.concat([df_ex] + to_add, ignore_index=True)
    result["_s"] = pd.to_datetime(result["Date"], errors="coerce")
    result = result.sort_values("_s").drop(columns=["_s"]).reset_index(drop=True)
    result.to_csv(MASTER_CSV, index=False, encoding="utf-8")
    
    last = pd.to_datetime(result["Date"], errors="coerce").max()
    print(f"\n  ✓ {len(result):,} partidos | último: {last.strftime('%Y-%m-%d')} | +{total} nuevos")


# =====================================================================
# ODDSPORTAL RESULTS SCRAPER
# =====================================================================

def _scrape_oddsportal_results(since_date):
    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
        from odds_scraper import TOURNAMENT_CONFIG, normalize_player_name
    except ImportError as e:
        print(f"  ⚠ {e}"); return pd.DataFrame()
    
    slugs = [
        "atp-miami", "atp-indian-wells", "atp-monte-carlo", "atp-madrid",
        "atp-rome", "atp-shanghai", "atp-paris", "atp-canada", "atp-cincinnati",
        "australian-open", "french-open", "wimbledon", "us-open",
        "atp-barcelona", "atp-dubai", "atp-rotterdam", "atp-acapulco",
        "atp-halle", "queens-club", "atp-hamburg", "atp-vienna", "atp-basel",
        "atp-beijing", "atp-washington", "atp-rio", "atp-tokyo",
        "atp-brisbane", "atp-adelaide", "atp-buenos-aires", "atp-doha",
        "atp-marseille", "atp-montpellier", "atp-dallas", "atp-santiago",
        "atp-finals",
    ]
    
    all_m = []
    try:
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage",
                                                          "--disable-blink-features=AutomationControlled"])
            ctx = br.new_context(
                viewport={"width":1920,"height":1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
                locale="en-US", timezone_id="Europe/Madrid",
            )
            ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});window.chrome={runtime:{}};")
            pg = ctx.new_page()
            
            for slug in slugs:
                cfg = TOURNAMENT_CONFIG.get(slug)
                if not cfg or "oddsportal" not in cfg: continue
                url = cfg["oddsportal"].rstrip("/") + "/results/"
                
                try:
                    pg.goto(url, timeout=15000, wait_until="domcontentloaded")
                    pg.wait_for_timeout(3000)
                    soup = BeautifulSoup(pg.content(), "lxml")
                    
                    rows = soup.find_all("div", class_=re.compile(r'eventRow'))
                    cnt = 0
                    for row in rows:
                        st = row.get("style","")
                        if "position: absolute" in st or "left: -9999" in st: continue
                        
                        names = row.find_all("p", class_="participant-name")
                        if len(names) < 2: continue
                        p1 = names[0].get_text(strip=True)
                        p2 = names[1].get_text(strip=True)
                        if "/" in p1 or "/" in p2: continue
                        
                        odds_els = row.find_all("p", attrs={"data-testid":"odd-container-default"})
                        odds = []
                        for el in odds_els:
                            try:
                                v=float(el.get_text(strip=True))
                                if 1.01<=v<=50: odds.append(v)
                            except: pass
                        
                        rt = row.get_text(" ",strip=True)
                        ss = re.findall(r'(\d+)\s*:\s*(\d+)', rt)
                        w1=l1=w2=l2=w3=l3=w4=l4=w5=l5=np.nan
                        ws=ls=np.nan
                        if ss:
                            sw=sl=0
                            for i,(a,b) in enumerate(ss):
                                a,b=int(a),int(b)
                                if a>b: sw+=1
                                else: sl+=1
                                if i==0: w1,l1=a,b
                                elif i==1: w2,l2=a,b
                                elif i==2: w3,l3=a,b
                                elif i==3: w4,l4=a,b
                                elif i==4: w5,l5=a,b
                            ws,ls=sw,sl
                        
                        comment="Completed"
                        tl=rt.lower()
                        if "ret" in tl: comment="Retired"
                        elif "w/o" in tl or "walkover" in tl: comment="Walkover"
                        
                        all_m.append({
                            "Date": datetime.now().strftime(DATE_FORMAT),
                            "Location": cfg["name"], "Tournament": cfg["name"],
                            "Series": cfg["series"], "Court": cfg["court"],
                            "Surface": cfg["surface"], "Round": "",
                            "Best of": 5 if cfg["series"]=="Grand Slam" else 3,
                            "Winner": normalize_player_name(p1),
                            "Loser": normalize_player_name(p2),
                            "WRank":np.nan,"LRank":np.nan,"WPts":np.nan,"LPts":np.nan,
                            "W1":w1,"L1":l1,"W2":w2,"L2":l2,"W3":w3,"L3":l3,
                            "W4":w4,"L4":l4,"W5":w5,"L5":l5,"Wsets":ws,"Lsets":ls,
                            "Comment": comment,
                            "AvgW": odds[0] if len(odds)>=2 else np.nan,
                            "AvgL": odds[1] if len(odds)>=2 else np.nan,
                        })
                        cnt+=1
                    
                    if cnt>0: print(f"    ✓ {cfg['name']}: {cnt} resultados")
                    time.sleep(1.5)
                except: continue
                if len(all_m)>=150: break
            br.close()
    except Exception as e:
        print(f"  ✗ Error: {e}"); return pd.DataFrame()
    
    if not all_m:
        print("  Sin resultados en OddsPortal"); return pd.DataFrame()
    
    df=pd.DataFrame(all_m)
    print(f"  → Total OddsPortal: {len(df)} resultados")
    return df


# =====================================================================
def main():
    dry_run = "--dry-run" in sys.argv
    build = "--build" in sys.argv
    only_td = "--only-tennis-data" in sys.argv
    from_y, to_y = DEFAULT_FROM_YEAR, DEFAULT_TO_YEAR
    
    for a in sys.argv[1:]:
        if a.startswith("--from-year="): from_y=int(a.split("=")[1])
        elif a.startswith("--to-year="): to_y=int(a.split("=")[1])
    
    print("="*55)
    print("  TENNIS.CSV — Build / Update")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}")
    print("="*55)
    
    if build or not os.path.exists(MASTER_CSV):
        if not os.path.exists(MASTER_CSV):
            print(f"\n  Tennis.csv no encontrado → build")
        build_csv(from_y, to_y, dry_run=dry_run)
    else:
        update_csv(dry_run=dry_run, only_td=only_td)

if __name__=="__main__": main()