"""
=============================================================================
 SCRIPT 2: ESCÁNER DIARIO DE APUESTAS (v2 — Playwright)
 
 Usa Playwright + BeautifulSoup para scraping real de cuotas.
 Fuentes: OddsPortal (principal) + Oddschecker (fallback)
 
 Requisitos:
   pip install playwright beautifulsoup4 lxml catboost pandas numpy
   playwright install chromium
 
 Uso:
   python daily_scanner.py --tournament=atp-miami --round=R3 --bankroll=1000
   python daily_scanner.py --tournament=atp-miami --source=oddschecker
   python daily_scanner.py --manual --bankroll=500
   python daily_scanner.py --list    # ver torneos disponibles
=============================================================================
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from collections import defaultdict

# Importar el módulo de scraping
from odds_scraper import (
    fetch_matches, list_available_tournaments,
    TOURNAMENT_CONFIG, normalize_player_name
)

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
PROJECT_DIR = os.path.abspath("./")
MASTER_CSV = os.path.join(PROJECT_DIR, "Tennis.csv")
MODEL_FILE = os.path.join(PROJECT_DIR, "tennis_v4.cbm")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
DEFAULT_BANKROLL = 1000.0

SEGMENT_CONFIG = {
    "round": {
        "Semifinals":    {"min_edge": 0.04, "kelly_mult": 1.5, "enabled": True},
        "The Final":     {"min_edge": 0.04, "kelly_mult": 1.5, "enabled": True},
        "2nd Round":     {"min_edge": 0.04, "kelly_mult": 1.2, "enabled": True},
        "3rd Round":     {"min_edge": 0.05,"kelly_mult": 1.0, "enabled": True},
        "4th Round":     {"min_edge": 0.05,"kelly_mult": 1.0, "enabled": True},
        "Round Robin":   {"min_edge": 0.05,"kelly_mult": 1.0, "enabled": True},
        "Quarterfinals": {"min_edge": 0.06, "kelly_mult": 0.8, "enabled": True},
        "1st Round":     {"min_edge": 0.0,  "kelly_mult": 0.0, "enabled": False},
    },
    "surface": {
        "Clay": {"edge_bonus":-0.02,"kelly_mult":1.3},
        "Hard": {"edge_bonus":0.00,"kelly_mult":1.0},
        "Grass":{"edge_bonus":0.02,"kelly_mult":0.8},
    },
    "series": {
        "Grand Slam":  {"edge_bonus":-0.02,"kelly_mult":1.3},
        "ATP500":      {"edge_bonus":0.00,"kelly_mult":1.0},
        "Masters 1000":{"edge_bonus":0.03,"kelly_mult":0.7},
        "ATP250":      {"edge_bonus":0.02,"kelly_mult":0.8},
        "Masters Cup": {"edge_bonus":0.00,"kelly_mult":1.0},
    },
}
BASE_KELLY_FRACTION = 0.25
MAX_BET_PCT = 0.03
ROUND_SHORTCUTS = {"R1":"1st Round","R2":"2nd Round","R3":"3rd Round",
                   "R4":"4th Round","QF":"Quarterfinals","SF":"Semifinals",
                   "F":"The Final","RR":"Round Robin"}

def get_segment_config(round_name, surface, series):
    rnd = SEGMENT_CONFIG["round"].get(round_name,{"min_edge":0.075,"kelly_mult":1.0,"enabled":True})
    if not rnd["enabled"]: return None
    surf = SEGMENT_CONFIG["surface"].get(surface,{"edge_bonus":0,"kelly_mult":1.0})
    ser = SEGMENT_CONFIG["series"].get(series,{"edge_bonus":0,"kelly_mult":1.0})
    return {"threshold":max(0.02,rnd["min_edge"]+surf["edge_bonus"]+ser["edge_bonus"]),
            "kelly_mult":rnd["kelly_mult"]*surf["kelly_mult"]*ser["kelly_mult"]}

def manual_input_matches():
    print("\n  Introduce partidos manualmente (vacío para terminar):")
    matches = []
    while True:
        print(f"\n  Partido {len(matches)+1}:")
        p1 = input("    Jugador 1 (ej 'Sinner J.'): ").strip()
        if not p1: break
        p2 = input("    Jugador 2 (ej 'Alcaraz C.'): ").strip()
        try: p1_odds=float(input("    Cuota P1: ")); p2_odds=float(input("    Cuota P2: "))
        except ValueError: print("    Cuota inválida"); continue
        surface = input("    Superficie [Hard]: ").strip() or "Hard"
        court = input("    Court [Outdoor]: ").strip() or "Outdoor"
        rnd = input("    Ronda (R1/R2/R3/R4/QF/SF/F): ").strip()
        series = input("    Serie: ").strip()
        matches.append({"p1_name":p1,"p2_name":p2,"p1_odds":p1_odds,"p2_odds":p2_odds,
                        "surface":surface,"court":court,"round":ROUND_SHORTCUTS.get(rnd,rnd),
                        "series":series,"source":"manual"})
    return matches

# =====================================================================
# MOTOR DE PREDICCIÓN (sin cambios funcionales respecto a tu v1)
# =====================================================================
class TennisPredictor:
    def __init__(self, master_csv, model_file):
        from catboost import CatBoostClassifier
        print("Cargando modelo y datos...")
        self.model = CatBoostClassifier(); self.model.load_model(model_file)
        self.feature_cols = self.model.feature_names_
        df = pd.read_csv(master_csv, encoding="utf-8"); df["Date"]=pd.to_datetime(df["Date"])
        self.df = df[df["Comment"].isin(["Completed","Retired"])].copy()
        for c in ["AvgW","AvgL","WRank","LRank","WPts","LPts"]:
            if c in self.df.columns: self.df[c]=pd.to_numeric(self.df[c],errors="coerce")
        self.df=self.df.dropna(subset=["WRank","LRank","WPts","LPts"]).sort_values("Date").reset_index(drop=True)
        self._build_elo(); self._build_history(); self._build_player_features(); self._build_h2h()
        print(f"Listo: {len(self.df):,} partidos, {len(self.elo_global):,} jugadores")

    def _build_elo(self):
        K,S,D,RW=32,1500,180,0.5
        def r2e(r):
            if pd.isna(r) or r<=0: return S
            return max(1300,2100-150*np.log2(max(1,r)))
        self.elo_global,self.elo_surface={},defaultdict(dict); self.ELO_START=S; ld={}
        self.df["was_retirement"]=(self.df["Comment"]=="Retired").astype(int)
        for _,row in self.df.iterrows():
            w,l,sf,dt=row["Winner"],row["Loser"],row["Surface"],row["Date"]
            ir=row.get("was_retirement",0)
            for p,r in [(w,row["WRank"]),(l,row["LRank"])]:
                if p not in self.elo_global: self.elo_global[p]=r2e(r)
                if sf not in self.elo_surface[p]: self.elo_surface[p][sf]=self.elo_global[p]
                if p in ld:
                    d=(dt-ld[p]).days
                    if d>D:
                        dc=0.5**(d/365); self.elo_global[p]=S+(self.elo_global[p]-S)*dc
                        self.elo_surface[p][sf]=S+(self.elo_surface[p][sf]-S)*dc
            km={"Grand Slam":1.5,"Masters 1000":1.2,"ATP500":1.0}.get(row["Series"],0.8)
            if row["Round"]=="The Final": km*=1.3
            elif row["Round"] in ["Semifinals","Quarterfinals"]: km*=1.1
            k=K*km; aw=1.0-(RW*ir*0.5); al=1.0-aw
            ew,eg=self.elo_global[w],self.elo_global[l]; xw=1/(1+10**((eg-ew)/400))
            self.elo_global[w]+=k*(aw-xw); self.elo_global[l]+=k*(al-(1-xw))
            es1=self.elo_surface[w].get(sf,ew); es2=self.elo_surface[l].get(sf,eg)
            xs=1/(1+10**((es2-es1)/400))
            self.elo_surface[w][sf]=es1+k*(aw-xs); self.elo_surface[l][sf]=es2+k*(al-(1-xs))
            ld[w]=dt; ld[l]=dt

    def _build_history(self):
        recs=[]
        for i,row in self.df.iterrows():
            b={"match_idx":i,"date":row["Date"],"tournament":row["Tournament"],
               "location":row["Location"],"surface":row["Surface"],"round":row["Round"]}
            sp=(row.get("Wsets",0) or 0)+(row.get("Lsets",0) or 0)
            wr=row.get("was_retirement",0)
            recs.append({**b,"player":row["Winner"],"won":1,"player_rank":row["WRank"],
                "player_pts":row["WPts"],"opp_rank":row["LRank"],"opp_pts":row["LPts"],
                "sets_played":sp,"retired":0,"was_retirement":wr})
            recs.append({**b,"player":row["Loser"],"won":0,"player_rank":row["LRank"],
                "player_pts":row["LPts"],"opp_rank":row["WRank"],"opp_pts":row["WPts"],
                "sets_played":sp,"retired":wr,"was_retirement":wr})
        self.history=pd.DataFrame(recs).sort_values(["player","date"]).reset_index(drop=True)

    def _build_player_features(self):
        DC=60; self.player_latest={}
        for player,g in self.history.groupby("player"):
            g=g.sort_values("date").reset_index(drop=True)
            if len(g)<2: continue
            w=g["won"].values.astype(float); op=g["opp_pts"].values; sp=g["sets_played"].values
            rt=g["retired"].values; wr=g["was_retirement"].values
            now=np.datetime64('now','D'); dn=pd.to_datetime(g["date"].values).values.astype('datetime64[D]')
            da=(now-dn).astype(float); dc=np.exp(-np.log(2)*da/DC)
            aw=w.copy()
            for k in range(len(aw)):
                if aw[k]==1 and wr[k]==1 and rt[k]==0: aw[k]=0.75
            f={}
            for win,nm in [(30,"30d"),(60,"60d"),(90,"90d"),(180,"180d")]:
                m=da<=win; f[f"win_rate_{nm}"]=aw[m].mean() if m.sum()>0 else np.nan
            f["win_rate_decay"]=np.average(aw,weights=dc)
            qw=op*dc; f["weighted_wr_decay"]=np.average(aw,weights=qw) if qw.sum()>0 else np.nan
            f["matches_7d"]=int((da<=7).sum()); f["matches_30d"]=int((da<=30).sum())
            f["matches_60d"]=int((da<=60).sum())
            m14=da<=14; f["sets_14d"]=int(sp[m14].sum()) if m14.sum()>0 else 0
            f["days_since_last"]=float(da.min())
            s=0
            if w[-1]==1:
                for k in range(len(w)-1,-1,-1):
                    if w[k]==1: s+=1
                    else: break
            else:
                for k in range(len(w)-1,-1,-1):
                    if w[k]==0: s-=1
                    else: break
            f["streak"]=s; f["tourn_momentum"]=0
            wm=w==1; f["avg_opp_pts_won_decay"]=np.average(op[wm],weights=dc[wm]) if wm.sum()>0 else np.nan
            lm=w==0; f["avg_opp_pts_lost_decay"]=np.average(op[lm],weights=dc[lm]) if lm.sum()>0 else np.nan
            m90=da<=90; f["retirement_rate_90d"]=rt[m90].mean() if m90.sum()>0 else np.nan
            f["player_rank"]=g.iloc[-1]["player_rank"]; f["player_pts"]=g.iloc[-1]["player_pts"]
            self.player_latest[player]=f

    def _build_h2h(self):
        self.h2h_history=defaultdict(list)
        for _,row in self.df.iterrows():
            k=tuple(sorted([row["Winner"],row["Loser"]])); self.h2h_history[k].append((row["Date"],row["Winner"]))

    def find_player(self,name):
        if name in self.player_latest: return name
        sn=name.split()[0] if name else ""
        c=[p for p in self.player_latest if p.startswith(sn)]
        return c[0] if len(c)==1 else None

    def predict(self,p1,p2,surface,court,rnd,series):
        p1f=self.player_latest.get(p1); p2f=self.player_latest.get(p2)
        if not p1f: return {"error":f"No encontrado: {p1}"}
        if not p2f: return {"error":f"No encontrado: {p2}"}
        r={}
        r["p1_rank"]=p1f["player_rank"];r["p2_rank"]=p2f["player_rank"]
        r["p1_pts"]=p1f["player_pts"];r["p2_pts"]=p2f["player_pts"]
        r["rank_diff"]=r["p1_rank"]-r["p2_rank"]
        r["log_rank_ratio"]=np.log1p(r["p1_rank"])-np.log1p(r["p2_rank"])
        r["pts_diff"]=r["p1_pts"]-r["p2_pts"]
        r["pts_ratio"]=r["p1_pts"]/r["p2_pts"] if r["p2_pts"]>0 else np.nan
        r["p1_elo_global"]=self.elo_global.get(p1,self.ELO_START)
        r["p2_elo_global"]=self.elo_global.get(p2,self.ELO_START)
        r["elo_global_diff"]=r["p1_elo_global"]-r["p2_elo_global"]
        e1s=self.elo_surface.get(p1,{}).get(surface,r["p1_elo_global"])
        e2s=self.elo_surface.get(p2,{}).get(surface,r["p2_elo_global"])
        r["p1_elo_surface"]=e1s;r["p2_elo_surface"]=e2s;r["elo_surface_diff"]=e1s-e2s
        r["p1_elo_surf_bonus"]=e1s-r["p1_elo_global"];r["p2_elo_surf_bonus"]=e2s-r["p2_elo_global"]
        r["elo_surf_bonus_diff"]=r["p1_elo_surf_bonus"]-r["p2_elo_surf_bonus"]
        fm=["win_rate_30d","win_rate_60d","win_rate_90d","win_rate_180d","win_rate_decay",
            "weighted_wr_decay","matches_7d","matches_30d","matches_60d","sets_14d",
            "days_since_last","streak","tourn_momentum","avg_opp_pts_won_decay",
            "avg_opp_pts_lost_decay","retirement_rate_90d"]
        r["p1_surface_wr_decay"]=p1f.get("win_rate_decay",np.nan)
        r["p2_surface_wr_decay"]=p2f.get("win_rate_decay",np.nan)
        for f in fm: r[f"p1_{f}"]=p1f.get(f,np.nan);r[f"p2_{f}"]=p2f.get(f,np.nan)
        key=tuple(sorted([p1,p2]));past=self.h2h_history.get(key,[])
        if past:
            now=pd.Timestamp.now();s1=s2=0
            for d,w in past:
                dy=(now-d).days;wt=np.exp(-np.log(2)*dy/365)
                if w==p1:s1+=wt
                else:s2+=wt
            t=s1+s2;r.update({"h2h_p1_decay":s1,"h2h_p2_decay":s2,"h2h_total_decay":t,
                              "h2h_ratio_decay":s1/t if t>0 else 0.5})
        else: r.update({"h2h_p1_decay":0,"h2h_p2_decay":0,"h2h_total_decay":0,"h2h_ratio_decay":0.5})
        for w in ["win_rate_30d","win_rate_60d","win_rate_90d","win_rate_180d","win_rate_decay","weighted_wr_decay"]:
            v1=r.get(f"p1_{w}",np.nan);v2=r.get(f"p2_{w}",np.nan)
            r[f"diff_{w}"]=(v1 if v1 is not None else np.nan)-(v2 if v2 is not None else np.nan)
        r["diff_surface_wr"]=(r.get("p1_surface_wr_decay") or np.nan)-(r.get("p2_surface_wr_decay") or np.nan)
        for d in ["streak","tourn_momentum","matches_7d","matches_30d","sets_14d"]:
            r[f"diff_{d}"]=(r.get(f"p1_{d}",0) or 0)-(r.get(f"p2_{d}",0) or 0)
        r["diff_opp_pts_won"]=(r.get("p1_avg_opp_pts_won_decay") or np.nan)-(r.get("p2_avg_opp_pts_won_decay") or np.nan)
        for d in ["tourn_wins","tourn_apps","tourn_finals","loc_wins","loc_apps"]:
            r[f"p1_{d}"]=0;r[f"p2_{d}"]=0
        r["diff_tourn_wins"]=0;r["diff_loc_wins"]=0;r["p1_tourn_wr"]=np.nan;r["p2_tourn_wr"]=np.nan
        r["is_grand_slam"]=1 if series=="Grand Slam" else 0
        r["is_masters"]=1 if series=="Masters 1000" else 0
        r["surface"]=surface;r["court"]=court;r["round"]=rnd;r["series"]=series
        X=pd.DataFrame([r])
        for c in self.feature_cols:
            if c not in X.columns: X[c]=np.nan
        X=X[self.feature_cols]; pp=self.model.predict_proba(X)[0][1]
        hr=self.h2h_history.get(key,[])
        return {"p1_prob":pp,"p2_prob":1-pp,"p1_elo":r["p1_elo_global"],"p2_elo":r["p2_elo_global"],
                "p1_elo_surf":e1s,"p2_elo_surf":e2s,
                "h2h":f"{sum(1 for _,w in hr if w==p1)}-{sum(1 for _,w in hr if w==p2)}",
                "nan_count":int(X.isna().sum().sum())}

# =====================================================================
# INFORME
# =====================================================================
def generate_report(predictor, matches_data, bankroll):
    today=datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*70}\n  INFORME — {today} | Bankroll: {bankroll:,.0f}€\n{'='*70}")
    bets,skipped=[],[]
    for match in matches_data:
        p1,p2=match["p1_name"],match["p2_name"]
        sf,ct=match.get("surface","Hard"),match.get("court","Outdoor")
        rn,sr=match.get("round","3rd Round"),match.get("series","Masters 1000")
        o1,o2=match["p1_odds"],match["p2_odds"]
        for a in ["p1_name","p2_name"]:
            res=predictor.find_player(match[a])
            if res and res!=match[a]: print(f"  ({match[a]} → {res})"); match[a]=res
        p1,p2=match["p1_name"],match["p2_name"]
        seg=get_segment_config(rn,sf,sr)
        print(f"\n  {'─'*60}\n  {p1} vs {p2}\n  {sf}|{ct}|{rn}|{sr}")
        print(f"  Cuotas: {p1} @ {o1:.2f} | {p2} @ {o2:.2f} [{match.get('source','?')}]")
        if seg is None: print("  ⛔ DESACTIVADO"); continue
        res=predictor.predict(p1,p2,sf,ct,rn,sr)
        if "error" in res: print(f"  ⚠ {res['error']}"); skipped.append(res['error']); continue
        pp1,pp2=res["p1_prob"],res["p2_prob"]
        print(f"  Modelo: {p1} {pp1*100:.1f}% | {p2} {pp2*100:.1f}%")
        print(f"  Elo: {res['p1_elo']:.0f}({res['p1_elo_surf']:.0f}) vs {res['p2_elo']:.0f}({res['p2_elo_surf']:.0f})")
        print(f"  H2H: {res['h2h']} | Thr: {seg['threshold']*100:.1f}% | NaN: {res['nan_count']}")
        for nm,pr,od in [(p1,pp1,o1),(p2,pp2,o2)]:
            edge=pr-(1/od)
            if edge>seg["threshold"] and od>=1/pr:
                b=od-1;ky=max(0,(b*pr-(1-pr))/b)
                bp=min(ky*BASE_KELLY_FRACTION*seg["kelly_mult"],MAX_BET_PCT)
                st=bankroll*bp;ev=pr*(od-1)-(1-pr)
                print(f"\n  ✅ {nm} @ {od:.2f} | Edge:{edge*100:+.1f}% | {bp*100:.2f}% → {st:.2f}€")
                bets.append({"match":f"{p1} vs {p2}","bet_on":nm,"odds":od,"prob":pr,
                             "edge":edge,"stake":round(st,2),"ev":round(ev,4),"round":rn,"surface":sf,"series":sr})
            elif edge>0:
                print(f"  ⏸ {nm}: edge {edge*100:.1f}% < {seg['threshold']*100:.1f}%")
    print(f"\n{'='*70}\n  RESUMEN\n{'='*70}")
    if bets:
        ts=sum(b["stake"] for b in bets)
        print(f"  {len(bets)} apuestas | {ts:.2f}€ ({ts/bankroll*100:.1f}%)")
        for b in bets:
            print(f"    → {b['bet_on']:25s} @ {b['odds']:.2f} | {b['stake']:>7.2f}€ | Edge:{b['edge']*100:+.1f}%")
    else: print("  Sin apuestas. Paciencia.")
    if skipped: print(f"\n  ⚠ No encontrados: {', '.join(set(skipped))}")
    os.makedirs(REPORT_DIR,exist_ok=True)
    rf=os.path.join(REPORT_DIR,f"report_{today}.json")
    with open(rf,"w") as f: json.dump({"date":today,"bankroll":bankroll,"bets":bets,"skipped":skipped},f,indent=2,default=str)
    print(f"\n  Informe: {rf}"); return bets

# =====================================================================
def main():
    bankroll,tournament,manual,round_override,source=DEFAULT_BANKROLL,None,False,None,"auto"
    for a in sys.argv[1:]:
        if a.startswith("--bankroll="): bankroll=float(a.split("=")[1])
        elif a.startswith("--tournament="): tournament=a.split("=")[1]
        elif a.startswith("--round="): round_override=ROUND_SHORTCUTS.get(a.split("=")[1],a.split("=")[1])
        elif a.startswith("--source="): source=a.split("=")[1]
        elif a=="--manual": manual=True
        elif a=="--list": list_available_tournaments(); return
    print(f"{'='*70}\n  ESCÁNER v2 (Playwright) — {datetime.now():%Y-%m-%d %H:%M}\n{'='*70}")
    if not os.path.exists(MODEL_FILE): print(f"ERROR: {MODEL_FILE}"); return
    if not os.path.exists(MASTER_CSV): print(f"ERROR: {MASTER_CSV}"); return
    predictor=TennisPredictor(MASTER_CSV,MODEL_FILE)
    if manual:
        md=manual_input_matches()
    elif tournament:
        md,cfg=fetch_matches(tournament,source=source)
        if not md:
            print("¿Manual? (s/n)");
            md=manual_input_matches() if input().strip().lower()=='s' else None
            if not md: return
        for m in md:
            if round_override: m["round"]=round_override
            elif "round" not in m:
                r=input(f"  Ronda {m['p1_name']} vs {m['p2_name']}? ").strip()
                m["round"]=ROUND_SHORTCUTS.get(r,r)
    else:
        print("\nUso:\n  python daily_scanner.py --tournament=atp-miami --round=R3\n"
              "  python daily_scanner.py --manual\n  python daily_scanner.py --list"); return
    if md: generate_report(predictor,md,bankroll)

if __name__=="__main__": main()