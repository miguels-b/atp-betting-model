#!/usr/bin/env python3
"""
=============================================================================
 TENNIS BETTING ML MODEL v4 - ATP Value Bet Detection
 
 Mejoras v4:
 - Elo inicializado por ranking ATP (no todos en 1500)
 - Retiradas ponderadas (pesan menos en Elo y win rates)
 - Momentum de torneo (wins consecutivas esta semana)
 - H2H con decay temporal
 - Interfaz de predicción: cuota mínima rentable + Kelly criterion
=============================================================================
"""

# %% [markdown]
# # 1. CARGA Y LIMPIEZA

# %%
import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

#files = sorted(glob.glob("*.csv"))
#dfs = [pd.read_excel(f) for f in files]
# Para CSVs locales:
files = sorted(glob.glob("*.csv"))
dfs = [pd.read_csv(f, encoding="latin-1") for f in files]

df_raw = pd.concat(dfs, ignore_index=True)
df_raw["Date"] = pd.to_datetime(df_raw["Date"])
print(f"Cargados: {len(df_raw):,} partidos")

df = df_raw[df_raw["Comment"].isin(["Completed", "Retired"])].copy()
for col in ["B365W","B365L","PSW","PSL","MaxW","MaxL","AvgW","AvgL"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["W_sets_played"] = df[["Wsets","Lsets"]].sum(axis=1)
df["was_retirement"] = (df["Comment"]=="Retired").astype(int)
df = df.dropna(subset=["AvgW","AvgL","WRank","LRank","WPts","LPts"]).copy()
df = df.sort_values("Date").reset_index(drop=True)
print(f"Tras limpieza: {len(df):,} partidos")

# %% [markdown]
# # 2. HISTORIAL CRONOLÓGICO

# %%
print("Construyendo historial...")
records = []
for i, row in df.iterrows():
    base = {
        "match_idx": i, "date": row["Date"],
        "tournament": row["Tournament"], "location": row["Location"],
        "series": row["Series"], "surface": row["Surface"],
        "court": row["Court"], "round": row["Round"],
        "best_of": row["Best of"], "was_retirement": row["was_retirement"],
    }
    records.append({**base, "player": row["Winner"], "opponent": row["Loser"],
        "won": 1, "retired": 0, "player_rank": row["WRank"],
        "player_pts": row["WPts"], "opp_rank": row["LRank"],
        "opp_pts": row["LPts"], "sets_played": row["W_sets_played"]})
    records.append({**base, "player": row["Loser"], "opponent": row["Winner"],
        "won": 0, "retired": row["was_retirement"], "player_rank": row["LRank"],
        "player_pts": row["LPts"], "opp_rank": row["WRank"],
        "opp_pts": row["WPts"], "sets_played": row["W_sets_played"]})

history = pd.DataFrame(records).sort_values(["player","date"]).reset_index(drop=True)
print(f"Historial: {len(history):,} registros ({history['player'].nunique():,} jugadores)")

# %% [markdown]
# # 3. ELO MEJORADO
# - Inicialización por ranking ATP (no todos en 1500)
# - Retiradas ponderadas (cuentan menos)
# - Decay por inactividad
# - K ajustado por torneo + ronda

# %%
print("Calculando Elo mejorado (global + superficie)...")

K_BASE = 32
ELO_START = 1500
ELO_DECAY_DAYS = 180
RETIREMENT_WEIGHT = 0.5  # Una victoria por retirada cuenta la mitad

def rank_to_initial_elo(rank):
    """Convierte ranking ATP a Elo inicial estimado."""
    # Top 1 ~ 2100, Top 10 ~ 1900, Top 50 ~ 1700, Top 100 ~ 1600, >200 ~ 1450
    if pd.isna(rank) or rank <= 0:
        return ELO_START
    return max(1300, 2100 - 150 * np.log2(max(1, rank)))

elo_global = {}
elo_surface = defaultdict(dict)
last_match_date = {}
elo_pre_global = {}
elo_pre_surface = {}

for i, row in df.iterrows():
    winner, loser = row["Winner"], row["Loser"]
    surface = row["Surface"]
    match_date = row["Date"]
    is_retirement = row["was_retirement"]
    
    # Inicializar Elo si es primera vez (basado en ranking)
    if winner not in elo_global:
        elo_global[winner] = rank_to_initial_elo(row["WRank"])
    if loser not in elo_global:
        elo_global[loser] = rank_to_initial_elo(row["LRank"])
    if surface not in elo_surface.get(winner, {}):
        elo_surface[winner] = elo_surface.get(winner, {})
        elo_surface[winner][surface] = elo_global[winner]  # hereda del global
    if surface not in elo_surface.get(loser, {}):
        elo_surface[loser] = elo_surface.get(loser, {})
        elo_surface[loser][surface] = elo_global[loser]
    
    # Decay por inactividad
    for player in [winner, loser]:
        if player in last_match_date:
            days_inactive = (match_date - last_match_date[player]).days
            if days_inactive > ELO_DECAY_DAYS:
                decay = 0.5 ** (days_inactive / 365)
                elo_global[player] = ELO_START + (elo_global[player] - ELO_START) * decay
                if surface in elo_surface.get(player, {}):
                    elo_surface[player][surface] = ELO_START + (elo_surface[player][surface] - ELO_START) * decay
    
    # Guardar PRE-partido
    elo_pre_global[(i, winner)] = elo_global[winner]
    elo_pre_global[(i, loser)] = elo_global[loser]
    elo_pre_surface[(i, winner)] = elo_surface[winner].get(surface, elo_global[winner])
    elo_pre_surface[(i, loser)] = elo_surface[loser].get(surface, elo_global[loser])
    
    # K ajustado
    series = row["Series"]
    k_mult = {"Grand Slam": 1.5, "Masters 1000": 1.2, "ATP500": 1.0}.get(series, 0.8)
    rnd = row["Round"]
    if rnd == "The Final": k_mult *= 1.3
    elif rnd in ["Semifinals","Quarterfinals"]: k_mult *= 1.1
    k = K_BASE * k_mult
    
    # Score ajustado por retirada
    actual_w = 1.0 - (RETIREMENT_WEIGHT * is_retirement * 0.5)  # ~0.75 si retirada
    actual_l = 1.0 - actual_w
    
    # Update global
    ew = elo_global[winner]
    el = elo_global[loser]
    exp_w = 1 / (1 + 10**((el - ew)/400))
    elo_global[winner] += k * (actual_w - exp_w)
    elo_global[loser] += k * (actual_l - (1 - exp_w))
    
    # Update surface
    ew_s = elo_surface[winner].get(surface, ew)
    el_s = elo_surface[loser].get(surface, el)
    exp_w_s = 1 / (1 + 10**((el_s - ew_s)/400))
    elo_surface[winner][surface] = ew_s + k * (actual_w - exp_w_s)
    elo_surface[loser][surface] = el_s + k * (actual_l - (1 - exp_w_s))
    
    last_match_date[winner] = match_date
    last_match_date[loser] = match_date

print(f"Elo calculado para {len(elo_global):,} jugadores")

# %% [markdown]
# # 4. FEATURES ROLLING + MOMENTUM + H2H CON DECAY

# %%
print("Calculando features rolling...")

DECAY_HALFLIFE = 60
H2H_DECAY_HALFLIFE = 365  # H2H decae más lento

player_features = {}

for player, group in history.groupby("player"):
    group = group.sort_values("date").reset_index(drop=True)
    dates = group["date"].values
    wins = group["won"].values
    surfaces = group["surface"].values
    tournaments = group["tournament"].values
    locations = group["location"].values
    opp_pts = group["opp_pts"].values
    match_idxs = group["match_idx"].values
    sets_played = group["sets_played"].values
    retirements = group["retired"].values
    rounds_arr = group["round"].values
    was_ret = group["was_retirement"].values
    
    for j in range(len(group)):
        current_date = dates[j]
        midx = match_idxs[j]
        feats = {}
        
        if j == 0:
            for k in ["win_rate_30d","win_rate_60d","win_rate_90d","win_rate_180d",
                       "win_rate_decay","weighted_wr_decay","surface_wr_decay",
                       "avg_opp_pts_won_decay","avg_opp_pts_lost_decay",
                       "retirement_rate_90d"]:
                feats[k] = np.nan
            feats.update({"matches_7d":0,"matches_30d":0,"matches_60d":0,
                          "sets_14d":0,"days_since_last":np.nan,"streak":0,
                          "tourn_wins":0,"tourn_apps":0,"tourn_finals":0,
                          "loc_wins":0,"loc_apps":0,
                          "tourn_momentum":0,
                          "elo_global": elo_pre_global.get((midx,player),ELO_START),
                          "elo_surface": elo_pre_surface.get((midx,player),ELO_START)})
            player_features[(midx, player)] = feats
            continue
        
        pd_dates = dates[:j]
        pw = wins[:j]
        ps = surfaces[:j]
        pt = tournaments[:j]
        pl = locations[:j]
        po = opp_pts[:j]
        psets = sets_played[:j]
        pret = retirements[:j]
        p_was_ret = was_ret[:j]
        p_rounds = rounds_arr[:j]
        
        days_ago = (current_date - pd_dates).astype("timedelta64[D]").astype(float)
        decay = np.exp(-np.log(2) * days_ago / DECAY_HALFLIFE)
        
        # --- Retiradas ponderadas: victorias por retirada cuentan 0.75 ---
        adjusted_wins = pw.copy().astype(float)
        for k in range(len(adjusted_wins)):
            if adjusted_wins[k] == 1 and p_was_ret[k] == 1 and pret[k] == 0:
                adjusted_wins[k] = 0.75  # ganó porque el otro se retiró
        
        # --- Win rates por ventana ---
        for window, name in [(30,"30d"),(60,"60d"),(90,"90d"),(180,"180d")]:
            m = days_ago <= window
            feats[f"win_rate_{name}"] = adjusted_wins[m].mean() if m.sum() > 0 else np.nan
        
        # --- Win rate con decay ---
        feats["win_rate_decay"] = np.average(adjusted_wins, weights=decay)
        
        # --- Win rate ponderado por calidad rival ---
        qw = po * decay
        feats["weighted_wr_decay"] = np.average(adjusted_wins, weights=qw) if qw.sum() > 0 else np.nan
        
        # --- Superficie con decay ---
        cs = surfaces[j]
        sm = ps == cs
        feats["surface_wr_decay"] = np.average(adjusted_wins[sm], weights=decay[sm]) if sm.sum() > 0 else np.nan
        
        # --- Carga/fatiga ---
        feats["matches_7d"] = int((days_ago <= 7).sum())
        feats["matches_30d"] = int((days_ago <= 30).sum())
        feats["matches_60d"] = int((days_ago <= 60).sum())
        m14 = days_ago <= 14
        feats["sets_14d"] = int(psets[m14].sum()) if m14.sum() > 0 else 0
        feats["days_since_last"] = float(days_ago.min())
        
        # --- Racha ---
        streak = 0
        if pw[j-1] == 1:
            for k in range(j-1, -1, -1):
                if pw[k] == 1: streak += 1
                else: break
        else:
            for k in range(j-1, -1, -1):
                if pw[k] == 0: streak -= 1
                else: break
        feats["streak"] = streak
        
        # --- MOMENTUM DE TORNEO ---
        # Victorias consecutivas en el torneo actual esta semana
        ct = tournaments[j]
        tourn_momentum = 0
        for k in range(j-1, -1, -1):
            if pt[k] == ct and days_ago[k] <= 14 and pw[k] == 1:
                tourn_momentum += 1
            else:
                break
        feats["tourn_momentum"] = tourn_momentum
        
        # --- Historial torneo/location ---
        tm = pt == ct
        feats["tourn_wins"] = int(pw[tm].sum()) if tm.sum() > 0 else 0
        feats["tourn_apps"] = int(tm.sum())
        feats["tourn_finals"] = int((tm & (p_rounds == "The Final")).sum())
        lm = pl == locations[j]
        feats["loc_wins"] = int(pw[lm].sum()) if lm.sum() > 0 else 0
        feats["loc_apps"] = int(lm.sum())
        
        # --- Calidad rivales con decay ---
        wm = pw == 1
        feats["avg_opp_pts_won_decay"] = np.average(po[wm], weights=decay[wm]) if wm.sum() > 0 else np.nan
        lm2 = pw == 0
        feats["avg_opp_pts_lost_decay"] = np.average(po[lm2], weights=decay[lm2]) if lm2.sum() > 0 else np.nan
        
        # --- Retiradas recientes ---
        m90 = days_ago <= 90
        feats["retirement_rate_90d"] = pret[m90].mean() if m90.sum() > 0 else np.nan
        
        # --- Elo ---
        feats["elo_global"] = elo_pre_global.get((midx, player), ELO_START)
        feats["elo_surface"] = elo_pre_surface.get((midx, player), ELO_START)
        
        player_features[(midx, player)] = feats

print(f"Features: {len(player_features):,} pares")

# %% [markdown]
# # 5. ENSAMBLAR DATASET + H2H CON DECAY

# %%
print("Ensamblando dataset...")
np.random.seed(42)
swap = np.random.random(len(df)) < 0.5

matches = pd.DataFrame()
matches["date"] = df["Date"].values
matches["year"] = df["Date"].dt.year.values
matches["tournament"] = df["Tournament"].values
matches["location"] = df["Location"].values
matches["series"] = df["Series"].values
matches["court"] = df["Court"].values
matches["surface"] = df["Surface"].values
matches["round"] = df["Round"].values
matches["best_of"] = df["Best of"].values

matches["p1"] = np.where(swap, df["Winner"].values, df["Loser"].values)
matches["p2"] = np.where(swap, df["Loser"].values, df["Winner"].values)
matches["p1_rank"] = np.where(swap, df["WRank"].values, df["LRank"].values)
matches["p2_rank"] = np.where(swap, df["LRank"].values, df["WRank"].values)
matches["p1_pts"] = np.where(swap, df["WPts"].values, df["LPts"].values)
matches["p2_pts"] = np.where(swap, df["LPts"].values, df["WPts"].values)

# Odds SOLO para backtesting
matches["p1_avg_odds"] = np.where(swap, df["AvgW"].values, df["AvgL"].values)
matches["p2_avg_odds"] = np.where(swap, df["AvgL"].values, df["AvgW"].values)
matches["p1_max_odds"] = np.where(swap, df["MaxW"].values, df["MaxL"].values)
matches["p2_max_odds"] = np.where(swap, df["MaxL"].values, df["MaxW"].values)
p1i = 1/matches["p1_avg_odds"]; p2i = 1/matches["p2_avg_odds"]
ti = p1i + p2i
matches["p1_market_prob"] = p1i / ti
matches["p1_wins"] = swap.astype(int)

# Asignar player features
pfeat_names = [
    "win_rate_30d","win_rate_60d","win_rate_90d","win_rate_180d",
    "win_rate_decay","weighted_wr_decay",
    "matches_7d","matches_30d","matches_60d","sets_14d","days_since_last",
    "streak","tourn_momentum",
    "surface_wr_decay",
    "tourn_wins","tourn_apps","tourn_finals",
    "loc_wins","loc_apps",
    "avg_opp_pts_won_decay","avg_opp_pts_lost_decay",
    "retirement_rate_90d",
    "elo_global","elo_surface",
]

midxs = df.index.values
for feat in pfeat_names:
    p1v, p2v = [], []
    for idx, s in zip(midxs, swap):
        w, l = df.loc[idx,"Winner"], df.loc[idx,"Loser"]
        p1, p2 = (w,l) if s else (l,w)
        p1v.append(player_features.get((idx,p1),{}).get(feat, np.nan))
        p2v.append(player_features.get((idx,p2),{}).get(feat, np.nan))
    matches[f"p1_{feat}"] = p1v
    matches[f"p2_{feat}"] = p2v

# %% [markdown]
# ## 5.1 H2H CON DECAY TEMPORAL

# %%
print("Calculando H2H con decay...")
h2h_history = defaultdict(list)  # (pA, pB) sorted -> [(date, winner)]
h2h_feats = {"p1": [], "p2": [], "total": [], "ratio": []}

for i, row in matches.iterrows():
    p1, p2 = row["p1"], row["p2"]
    key = tuple(sorted([p1, p2]))
    current_date = row["date"]
    
    past = h2h_history[key]
    if len(past) == 0:
        h2h_feats["p1"].append(0)
        h2h_feats["p2"].append(0)
        h2h_feats["total"].append(0)
        h2h_feats["ratio"].append(0.5)
    else:
        p1_score = 0
        p2_score = 0
        for d, w in past:
            days = (current_date - d).days
            weight = np.exp(-np.log(2) * days / H2H_DECAY_HALFLIFE)
            if w == p1:
                p1_score += weight
            else:
                p2_score += weight
        total = p1_score + p2_score
        h2h_feats["p1"].append(p1_score)
        h2h_feats["p2"].append(p2_score)
        h2h_feats["total"].append(total)
        h2h_feats["ratio"].append(p1_score / total if total > 0 else 0.5)
    
    winner = p1 if row["p1_wins"] == 1 else p2
    h2h_history[key].append((current_date, winner))

matches["h2h_p1_decay"] = h2h_feats["p1"]
matches["h2h_p2_decay"] = h2h_feats["p2"]
matches["h2h_total_decay"] = h2h_feats["total"]
matches["h2h_ratio_decay"] = h2h_feats["ratio"]

# %% [markdown]
# ## 5.2 Features derivadas

# %%
matches["rank_diff"] = matches["p1_rank"] - matches["p2_rank"]
matches["log_rank_ratio"] = np.log1p(matches["p1_rank"]) - np.log1p(matches["p2_rank"])
matches["pts_diff"] = matches["p1_pts"] - matches["p2_pts"]
matches["pts_ratio"] = np.where(matches["p2_pts"]>0, matches["p1_pts"]/matches["p2_pts"], np.nan)

matches["elo_global_diff"] = matches["p1_elo_global"] - matches["p2_elo_global"]
matches["elo_surface_diff"] = matches["p1_elo_surface"] - matches["p2_elo_surface"]
matches["p1_elo_surf_bonus"] = matches["p1_elo_surface"] - matches["p1_elo_global"]
matches["p2_elo_surf_bonus"] = matches["p2_elo_surface"] - matches["p2_elo_global"]
matches["elo_surf_bonus_diff"] = matches["p1_elo_surf_bonus"] - matches["p2_elo_surf_bonus"]

for wr in ["win_rate_30d","win_rate_60d","win_rate_90d","win_rate_180d",
           "win_rate_decay","weighted_wr_decay"]:
    matches[f"diff_{wr}"] = matches[f"p1_{wr}"] - matches[f"p2_{wr}"]

matches["diff_surface_wr"] = matches["p1_surface_wr_decay"] - matches["p2_surface_wr_decay"]
matches["diff_streak"] = matches["p1_streak"] - matches["p2_streak"]
matches["diff_momentum"] = matches["p1_tourn_momentum"] - matches["p2_tourn_momentum"]
matches["diff_matches_7d"] = matches["p1_matches_7d"] - matches["p2_matches_7d"]
matches["diff_matches_30d"] = matches["p1_matches_30d"] - matches["p2_matches_30d"]
matches["diff_sets_14d"] = matches["p1_sets_14d"] - matches["p2_sets_14d"]
matches["diff_opp_pts_won"] = matches["p1_avg_opp_pts_won_decay"] - matches["p2_avg_opp_pts_won_decay"]
matches["diff_tourn_wins"] = matches["p1_tourn_wins"] - matches["p2_tourn_wins"]
matches["diff_loc_wins"] = matches["p1_loc_wins"] - matches["p2_loc_wins"]

matches["p1_tourn_wr"] = np.where(matches["p1_tourn_apps"]>0,
    matches["p1_tourn_wins"]/matches["p1_tourn_apps"], np.nan)
matches["p2_tourn_wr"] = np.where(matches["p2_tourn_apps"]>0,
    matches["p2_tourn_wins"]/matches["p2_tourn_apps"], np.nan)

matches["is_grand_slam"] = (matches["series"]=="Grand Slam").astype(int)
matches["is_masters"] = (matches["series"]=="Masters 1000").astype(int)

print(f"Dataset: {matches.shape}")

# %% [markdown]
# # 6. FEATURES Y ENTRENAMIENTO

# %%
feature_cols = [
    "p1_rank","p2_rank","rank_diff","log_rank_ratio",
    "p1_pts","p2_pts","pts_diff","pts_ratio",
    "p1_elo_global","p2_elo_global","elo_global_diff",
    "p1_elo_surface","p2_elo_surface","elo_surface_diff",
    "p1_elo_surf_bonus","p2_elo_surf_bonus","elo_surf_bonus_diff",
    "p1_win_rate_30d","p2_win_rate_30d","diff_win_rate_30d",
    "p1_win_rate_60d","p2_win_rate_60d","diff_win_rate_60d",
    "p1_win_rate_90d","p2_win_rate_90d","diff_win_rate_90d",
    "p1_win_rate_180d","p2_win_rate_180d","diff_win_rate_180d",
    "p1_win_rate_decay","p2_win_rate_decay","diff_win_rate_decay",
    "p1_weighted_wr_decay","p2_weighted_wr_decay","diff_weighted_wr_decay",
    "p1_surface_wr_decay","p2_surface_wr_decay","diff_surface_wr",
    "p1_matches_7d","p2_matches_7d","diff_matches_7d",
    "p1_matches_30d","p2_matches_30d","diff_matches_30d",
    "p1_matches_60d","p2_matches_60d",
    "p1_sets_14d","p2_sets_14d","diff_sets_14d",
    "p1_days_since_last","p2_days_since_last",
    "p1_streak","p2_streak","diff_streak",
    "p1_tourn_momentum","p2_tourn_momentum","diff_momentum",
    "p1_avg_opp_pts_won_decay","p2_avg_opp_pts_won_decay","diff_opp_pts_won",
    "p1_avg_opp_pts_lost_decay","p2_avg_opp_pts_lost_decay",
    "p1_retirement_rate_90d","p2_retirement_rate_90d",
    "p1_tourn_wins","p2_tourn_wins","diff_tourn_wins",
    "p1_tourn_apps","p2_tourn_apps",
    "p1_tourn_wr","p2_tourn_wr",
    "p1_tourn_finals","p2_tourn_finals",
    "p1_loc_wins","p2_loc_wins","diff_loc_wins",
    "p1_loc_apps","p2_loc_apps",
    "h2h_p1_decay","h2h_p2_decay","h2h_total_decay","h2h_ratio_decay",
    "is_grand_slam","is_masters",
    "surface","court","round","series",
]

cat_features = ["surface","court","round","series"]
cat_indices = [feature_cols.index(c) for c in cat_features]
target = "p1_wins"
print(f"Features: {len(feature_cols)} (0 odds)")

# %%
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

train = matches[(matches["year"] <= 2025) & (matches["year"] != 2022)].copy()
val = matches[matches["year"] == 2022].copy()
test = matches[matches["year"] == 2026].copy()
print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

model = CatBoostClassifier(
    iterations=2500, learning_rate=0.02, depth=7,
    l2_leaf_reg=3, min_data_in_leaf=20, random_seed=42,
    eval_metric="Logloss", early_stopping_rounds=200,
    verbose=200, cat_features=cat_indices, task_type="CPU",
)

train_pool = Pool(train[feature_cols], train[target], cat_features=cat_indices)
val_pool = Pool(val[feature_cols], val[target], cat_features=cat_indices)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# %% [markdown]
# # 7. EVALUACIÓN + BACKTEST

# %%
def evaluate(model, data, name=""):
    X, y = data[feature_cols], data[target]
    probs = model.predict_proba(X)[:,1]
    preds = (probs > 0.5).astype(int)
    mkt = data["p1_market_prob"].values
    mkt_preds = (mkt > 0.5).astype(int)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  {'Métrica':<20s} {'Modelo':>10s} {'Mercado':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")
    for mn, mv, mkv in [
        ("Accuracy", accuracy_score(y,preds), accuracy_score(y,mkt_preds)),
        ("ROC AUC", roc_auc_score(y,probs), roc_auc_score(y,mkt)),
        ("Log Loss", log_loss(y,probs), log_loss(y,mkt)),
        ("Brier", brier_score_loss(y,probs), brier_score_loss(y,mkt)),
    ]:
        d = mv - mkv
        print(f"  {mn:<20s} {mv:>10.4f} {mkv:>10.4f} {d:>+10.4f}")
    return probs

val_probs = evaluate(model, val, "VAL (2022)")
test_probs = evaluate(model, test, "TEST (2026)")

# %%
def backtest(data, probs, thr=0.05, stake=1.0, label=""):
    results = []
    for i in range(len(data)):
        row = data.iloc[i]
        mp1, mp2 = probs[i], 1-probs[i]
        mkt1 = row["p1_market_prob"]; mkt2 = 1-mkt1
        actual = row["p1_wins"]
        for bp, mp, mkp, oc, wc in [
            (row["p1"],mp1,mkt1,"p1_max_odds",actual==1),
            (row["p2"],mp2,mkt2,"p2_max_odds",actual==0)]:
            edge = mp - mkp
            if edge > thr:
                odds = row[oc]
                if pd.notna(odds) and odds > 1:
                    pft = (odds-1)*stake if wc else -stake
                    results.append({"date":row["date"],"player":bp,
                        "series":row["series"],"round":row["round"],
                        "surface":row["surface"],"edge":edge,"odds":odds,
                        "profit":pft,"won":wc,"model_prob":mp,"market_prob":mkp})
    if not results:
        print(f"  {label}thr={thr}: Sin apuestas"); return pd.DataFrame()
    bets = pd.DataFrame(results); bets["cum"] = bets["profit"].cumsum()
    t = len(bets)*stake; p = bets["profit"].sum(); roi = p/t*100
    print(f"  {label}thr={thr:.2f}: {len(bets):>4d} bets | ROI:{roi:>+7.2f}% | "
          f"P/L:{p:>+8.2f}€ | WR:{bets['won'].mean()*100:.1f}% | DD:{bets['cum'].min():.2f}€")
    return bets

print(f"\n{'='*70}\n  BACKTEST 2026\n{'='*70}")
for t in [0.02,0.03,0.05,0.07,0.10,0.15]:
    backtest(test, test_probs, thr=t)

print(f"\n--- Por Serie ---")
for s in ["Grand Slam","Masters 1000","ATP500","ATP250"]:
    m = test["series"]==s
    if m.sum()>30: backtest(test[m], test_probs[m.values], thr=0.05, label=f"{s:15s}| ")

print(f"\n--- Por Ronda ---")
for r in ["1st Round","2nd Round","Quarterfinals","Semifinals","The Final"]:
    m = test["round"]==r
    if m.sum()>10: backtest(test[m], test_probs[m.values], thr=0.05, label=f"{r:15s}| ")

print(f"\n--- Por Superficie ---")
for s in ["Hard","Clay","Grass"]:
    m = test["surface"]==s
    if m.sum()>30: backtest(test[m], test_probs[m.values], thr=0.05, label=f"{s:15s}| ")

# %% [markdown]
# # 8. VISUALIZACIONES

# %%
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

imp = model.get_feature_importance()
fi = pd.DataFrame({"f":feature_cols,"i":imp}).sort_values("i",ascending=True).tail(30)
fig,ax = plt.subplots(figsize=(10,12))
ax.barh(fi["f"],fi["i"],color="#2563eb")
ax.set_xlabel("Importance"); ax.set_title("Top 30 Features - Tennis v4")
plt.tight_layout(); plt.savefig("./grafica/fi_v4.png",dpi=150); plt.close()

fig,axes = plt.subplots(2,2,figsize=(16,10))
for ax,t in zip(axes.flat,[0.03,0.05,0.07,0.10]):
    b = backtest(test,test_probs,thr=t)
    if len(b)>0:
        ax.plot(range(len(b)),b["cum"],lw=1.2,color="#2563eb")
        ax.axhline(0,color="red",ls="--",alpha=.5)
        roi=b["profit"].sum()/len(b)*100
        ax.set_title(f"thr={t} | {len(b)} bets | ROI:{roi:+.1f}%")
        ax.set_xlabel("Nº apuesta"); ax.set_ylabel("Profit (€)"); ax.grid(True,alpha=.3)
plt.suptitle("Profit Curves v4",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig("./grafica/profit_v4.png",dpi=150); plt.close()

fig,axes = plt.subplots(1,2,figsize=(14,6))
for ax,(pr,d,n) in zip(axes,[(val_probs,val,"Val 2022"),(test_probs,test,"Test 2026")]):
    pt,pp = calibration_curve(d[target],pr,n_bins=10)
    ax.plot(pp,pt,"s-",label="Modelo v4",color="#2563eb")
    pm,ppm = calibration_curve(d[target],d["p1_market_prob"],n_bins=10)
    ax.plot(ppm,pm,"o-",label="Mercado",color="#f59e0b")
    ax.plot([0,1],[0,1],"k--",alpha=.5)
    ax.set_xlabel("P predicha"); ax.set_ylabel("Freq real")
    ax.set_title(f"Calibración - {n}"); ax.legend(); ax.grid(True,alpha=.3)
plt.tight_layout(); plt.savefig("./grafica/cal_v4.png",dpi=150); plt.close()
print("✅ Gráficos guardados")

# %% [markdown]
# # 9. GUARDAR MODELO

# %%
model.save_model("./tennis_v4.cbm")
matches.to_csv("./tennis_v4.csv", index=False)

# %% [markdown]
# # 10. CONFIGURACIÓN ADAPTATIVA POR SEGMENTO
#
# Thresholds y Kelly dinámicos según dónde el modelo tiene alpha.
# Basado en backtest histórico por ronda/superficie/serie.

# %%
SEGMENT_CONFIG = {
    # Por ronda — prioridad alta
    "round": {
        "Semifinals":    {"min_edge": 0.05, "kelly_mult": 1.5, "enabled": True},   # ROI +80%
        "The Final":     {"min_edge": 0.05, "kelly_mult": 1.5, "enabled": True},   # ROI +63%
        "2nd Round":     {"min_edge": 0.05, "kelly_mult": 1.2, "enabled": True},   # ROI +23%
        "3rd Round":     {"min_edge": 0.075,"kelly_mult": 1.0, "enabled": True},
        "4th Round":     {"min_edge": 0.075,"kelly_mult": 1.0, "enabled": True},
        "Round Robin":   {"min_edge": 0.075,"kelly_mult": 1.0, "enabled": True},
        "Quarterfinals": {"min_edge": 0.10, "kelly_mult": 0.8, "enabled": True},
        "1st Round":     {"min_edge": 0.0,  "kelly_mult": 0.0, "enabled": False},  # NO apostar
    },
    # Por superficie — ajuste sumado al threshold de ronda
    "surface": {
        "Clay":  {"edge_bonus": -0.02, "kelly_mult": 1.3},   # ROI +60%, bajar threshold
        "Hard":  {"edge_bonus":  0.00, "kelly_mult": 1.0},
        "Grass": {"edge_bonus":  0.02, "kelly_mult": 0.8},   # menos datos, más exigente
    },
    # Por serie — ajuste sumado al threshold de ronda
    "series": {
        "Grand Slam":   {"edge_bonus": -0.02, "kelly_mult": 1.3},  # ROI +17%
        "ATP500":       {"edge_bonus":  0.00, "kelly_mult": 1.0},
        "Masters 1000": {"edge_bonus":  0.03, "kelly_mult": 0.7},  # peor historial
        "ATP250":       {"edge_bonus":  0.02, "kelly_mult": 0.8},
        "Masters Cup":  {"edge_bonus":  0.00, "kelly_mult": 1.0},
    },
}

BASE_KELLY_FRACTION = 0.25   # 1/4 Kelly
MAX_BET_PCT = 0.03           # Cap 3% del bankroll por apuesta

def get_segment_config(round_name, surface, series):
    """Calcula threshold efectivo y multiplicador Kelly para un partido."""
    rnd_cfg = SEGMENT_CONFIG["round"].get(round_name,
        {"min_edge": 0.075, "kelly_mult": 1.0, "enabled": True})

    if not rnd_cfg["enabled"]:
        return None  # No apostar en este segmento

    surf_cfg = SEGMENT_CONFIG["surface"].get(surface,
        {"edge_bonus": 0, "kelly_mult": 1.0})
    ser_cfg = SEGMENT_CONFIG["series"].get(series,
        {"edge_bonus": 0, "kelly_mult": 1.0})

    effective_threshold = rnd_cfg["min_edge"] + surf_cfg["edge_bonus"] + ser_cfg["edge_bonus"]
    effective_threshold = max(0.02, effective_threshold)  # nunca < 2%

    kelly_mult = rnd_cfg["kelly_mult"] * surf_cfg["kelly_mult"] * ser_cfg["kelly_mult"]

    return {
        "threshold": effective_threshold,
        "kelly_mult": kelly_mult,
        "reason": f"Ronda({rnd_cfg['min_edge']*100:.0f}%) + Sup({surf_cfg['edge_bonus']*100:+.0f}%) + Ser({ser_cfg['edge_bonus']*100:+.0f}%) = {effective_threshold*100:.1f}%"
    }

# %% [markdown]
# # 11. INTERFAZ DE PREDICCIÓN ADAPTATIVA
#
# Introduce dos jugadores + contexto → predicción + cuota mínima + Kelly adaptativo

# %%
def predict_match(p1_name, p2_name, surface, court, round_name, series,
                  p1_odds=None, p2_odds=None, bankroll=None):
    """
    Predice un partido y calcula si hay value con estrategia adaptativa.

    Args:
        p1_name: Nombre del jugador 1 (como aparece en datos, ej "Sinner J.")
        p2_name: Nombre del jugador 2
        surface: "Hard", "Clay", "Grass"
        court: "Outdoor", "Indoor"
        round_name: "1st Round", "2nd Round", "Quarterfinals", "Semifinals", "The Final"
        series: "Grand Slam", "Masters 1000", "ATP500", "ATP250"
        p1_odds: Cuota ofrecida para P1 (opcional)
        p2_odds: Cuota ofrecida para P2 (opcional)
        bankroll: Tu bankroll total (para Kelly sizing)
    """
    # --- Verificar segmento ---
    seg = get_segment_config(round_name, surface, series)
    if seg is None:
        print(f"\n  ⛔ {round_name} | {surface} | {series}")
        print(f"     Este segmento está DESACTIVADO (historial negativo).")
        print(f"     No apostar en 1st Round — el modelo pierde ahí.")
        return

    # --- Buscar jugadores ---
    hist_p1 = history[history["player"] == p1_name].sort_values("date")
    hist_p2 = history[history["player"] == p2_name].sort_values("date")

    if len(hist_p1) == 0:
        print(f"⚠ No encontrado: '{p1_name}'")
        print(f"  Similares: {[p for p in history['player'].unique() if p1_name.split()[0] in p][:5]}")
        return
    if len(hist_p2) == 0:
        print(f"⚠ No encontrado: '{p2_name}'")
        print(f"  Similares: {[p for p in history['player'].unique() if p2_name.split()[0] in p][:5]}")
        return

    # --- Features ---
    last_p1_idx = hist_p1.iloc[-1]["match_idx"]
    last_p2_idx = hist_p2.iloc[-1]["match_idx"]
    p1_feats = player_features.get((last_p1_idx, p1_name), {})
    p2_feats = player_features.get((last_p2_idx, p2_name), {})

    row = {}
    last_p1 = hist_p1.iloc[-1]
    last_p2 = hist_p2.iloc[-1]
    row["p1_rank"] = last_p1["player_rank"]
    row["p2_rank"] = last_p2["player_rank"]
    row["p1_pts"] = last_p1["player_pts"]
    row["p2_pts"] = last_p2["player_pts"]
    row["rank_diff"] = row["p1_rank"] - row["p2_rank"]
    row["log_rank_ratio"] = np.log1p(row["p1_rank"]) - np.log1p(row["p2_rank"])
    row["pts_diff"] = row["p1_pts"] - row["p2_pts"]
    row["pts_ratio"] = row["p1_pts"] / row["p2_pts"] if row["p2_pts"] > 0 else np.nan

    row["p1_elo_global"] = elo_global.get(p1_name, ELO_START)
    row["p2_elo_global"] = elo_global.get(p2_name, ELO_START)
    row["elo_global_diff"] = row["p1_elo_global"] - row["p2_elo_global"]
    p1_es = elo_surface.get(p1_name, {}).get(surface, row["p1_elo_global"])
    p2_es = elo_surface.get(p2_name, {}).get(surface, row["p2_elo_global"])
    row["p1_elo_surface"] = p1_es
    row["p2_elo_surface"] = p2_es
    row["elo_surface_diff"] = p1_es - p2_es
    row["p1_elo_surf_bonus"] = p1_es - row["p1_elo_global"]
    row["p2_elo_surf_bonus"] = p2_es - row["p2_elo_global"]
    row["elo_surf_bonus_diff"] = row["p1_elo_surf_bonus"] - row["p2_elo_surf_bonus"]

    for feat in pfeat_names:
        if feat in ["elo_global", "elo_surface"]:
            continue
        row[f"p1_{feat}"] = p1_feats.get(feat, np.nan)
        row[f"p2_{feat}"] = p2_feats.get(feat, np.nan)

    key = tuple(sorted([p1_name, p2_name]))
    past_h2h = h2h_history.get(key, [])
    if past_h2h:
        now = pd.Timestamp.now()
        p1s, p2s = 0, 0
        for d, w in past_h2h:
            days = (now - d).days
            weight = np.exp(-np.log(2) * days / H2H_DECAY_HALFLIFE)
            if w == p1_name: p1s += weight
            else: p2s += weight
        total = p1s + p2s
        row["h2h_p1_decay"] = p1s
        row["h2h_p2_decay"] = p2s
        row["h2h_total_decay"] = total
        row["h2h_ratio_decay"] = p1s / total if total > 0 else 0.5
    else:
        row["h2h_p1_decay"] = 0
        row["h2h_p2_decay"] = 0
        row["h2h_total_decay"] = 0
        row["h2h_ratio_decay"] = 0.5

    for wr in ["win_rate_30d","win_rate_60d","win_rate_90d","win_rate_180d",
               "win_rate_decay","weighted_wr_decay"]:
        v1 = row.get(f"p1_{wr}", np.nan)
        v2 = row.get(f"p2_{wr}", np.nan)
        row[f"diff_{wr}"] = (v1 if v1 is not None else np.nan) - (v2 if v2 is not None else np.nan)

    row["diff_surface_wr"] = (row.get("p1_surface_wr_decay") or np.nan) - (row.get("p2_surface_wr_decay") or np.nan)
    row["diff_streak"] = (row.get("p1_streak",0) or 0) - (row.get("p2_streak",0) or 0)
    row["diff_momentum"] = (row.get("p1_tourn_momentum",0) or 0) - (row.get("p2_tourn_momentum",0) or 0)
    row["diff_matches_7d"] = (row.get("p1_matches_7d",0) or 0) - (row.get("p2_matches_7d",0) or 0)
    row["diff_matches_30d"] = (row.get("p1_matches_30d",0) or 0) - (row.get("p2_matches_30d",0) or 0)
    row["diff_sets_14d"] = (row.get("p1_sets_14d",0) or 0) - (row.get("p2_sets_14d",0) or 0)
    row["diff_opp_pts_won"] = (row.get("p1_avg_opp_pts_won_decay") or np.nan) - (row.get("p2_avg_opp_pts_won_decay") or np.nan)
    row["diff_tourn_wins"] = (row.get("p1_tourn_wins",0) or 0) - (row.get("p2_tourn_wins",0) or 0)
    row["diff_loc_wins"] = (row.get("p1_loc_wins",0) or 0) - (row.get("p2_loc_wins",0) or 0)
    row["p1_tourn_wr"] = row["p1_tourn_wins"] / row["p1_tourn_apps"] if row.get("p1_tourn_apps",0) > 0 else np.nan
    row["p2_tourn_wr"] = row["p2_tourn_wins"] / row["p2_tourn_apps"] if row.get("p2_tourn_apps",0) > 0 else np.nan
    row["is_grand_slam"] = 1 if series == "Grand Slam" else 0
    row["is_masters"] = 1 if series == "Masters 1000" else 0
    row["surface"] = surface
    row["court"] = court
    row["round"] = round_name
    row["series"] = series

    # --- Predecir ---
    X = pd.DataFrame([row])[feature_cols]
    prob_p1 = model.predict_proba(X)[0][1]
    prob_p2 = 1 - prob_p1

    min_odds_p1 = 1 / prob_p1 if prob_p1 > 0 else float('inf')
    min_odds_p2 = 1 / prob_p2 if prob_p2 > 0 else float('inf')

    nan_count = X.isna().sum().sum()
    total_feats = len(feature_cols)
    confidence = 1 - (nan_count / total_feats)

    # --- Output ---
    print(f"\n{'='*65}")
    print(f"  PREDICCIÓN: {p1_name} vs {p2_name}")
    print(f"  {surface} | {court} | {round_name} | {series}")
    print(f"{'='*65}")

    print(f"\n  Probabilidades del modelo:")
    print(f"    {p1_name:<25s} {prob_p1*100:5.1f}%")
    print(f"    {p2_name:<25s} {prob_p2*100:5.1f}%")

    print(f"\n  Cuota mínima rentable:")
    print(f"    {p1_name:<25s} {min_odds_p1:.2f}")
    print(f"    {p2_name:<25s} {min_odds_p2:.2f}")

    print(f"\n  Elo global:  {p1_name}: {row['p1_elo_global']:.0f}  |  {p2_name}: {row['p2_elo_global']:.0f}")
    print(f"  Elo {surface:5s}:  {p1_name}: {p1_es:.0f}  |  {p2_name}: {p2_es:.0f}")

    h2h_raw = h2h_history.get(key, [])
    p1_h2h = sum(1 for _,w in h2h_raw if w==p1_name)
    p2_h2h = sum(1 for _,w in h2h_raw if w==p2_name)
    print(f"  H2H:         {p1_h2h}-{p2_h2h} ({len(h2h_raw)} partidos)")
    print(f"  Confianza:   {confidence*100:.0f}% ({nan_count} features sin dato de {total_feats})")

    print(f"\n  Estrategia adaptativa:")
    print(f"    Threshold: {seg['threshold']*100:.1f}%  ({seg['reason']})")
    print(f"    Kelly mult: x{seg['kelly_mult']:.1f}")

    # --- Análisis de apuesta ---
    if p1_odds is not None and p2_odds is not None:
        print(f"\n  {'='*55}")
        print(f"  ANÁLISIS DE APUESTA")
        print(f"  {'='*55}")

        for name, prob, odds, min_o in [
            (p1_name, prob_p1, p1_odds, min_odds_p1),
            (p2_name, prob_p2, p2_odds, min_odds_p2)
        ]:
            edge = prob - (1/odds)
            ev = prob * (odds-1) - (1-prob)

            passes_threshold = edge > seg["threshold"]

            if passes_threshold and odds >= min_o:
                b = odds - 1
                kelly_full = max(0, (b * prob - (1-prob)) / b)
                bet_pct = min(kelly_full * BASE_KELLY_FRACTION * seg["kelly_mult"], MAX_BET_PCT)

                print(f"\n    ✅ APOSTAR: {name} @ {odds:.2f}")
                print(f"       Prob modelo: {prob*100:.1f}% | Cuota mín: {min_o:.2f}")
                print(f"       Edge: {edge*100:+.1f}% (threshold: {seg['threshold']*100:.1f}%) ✓")
                print(f"       EV por €1: {ev:+.3f}€")
                print(f"       Kelly full: {kelly_full*100:.1f}% | x{seg['kelly_mult']:.1f} = {kelly_full*seg['kelly_mult']*100:.1f}%")
                print(f"       Apuesta: {bet_pct*100:.2f}% del bankroll")
                if bankroll:
                    bet = bankroll * bet_pct
                    print(f"       → APOSTAR: {bet:.2f}€ de {bankroll:.0f}€")
            else:
                reason = ""
                if not passes_threshold:
                    reason = f"edge {edge*100:.1f}% < threshold {seg['threshold']*100:.1f}%"
                elif odds < min_o:
                    reason = f"odds {odds:.2f} < cuota mín {min_o:.2f}"
                print(f"\n    ❌ NO APOSTAR: {name} @ {odds:.2f}")
                print(f"       Prob modelo: {prob*100:.1f}% | Edge: {edge*100:+.1f}%")
                print(f"       Motivo: {reason}")

    return {"p1_prob": prob_p1, "p2_prob": prob_p2,
            "min_odds_p1": min_odds_p1, "min_odds_p2": min_odds_p2,
            "confidence": confidence,
            "threshold": seg["threshold"], "kelly_mult": seg["kelly_mult"]}

# %% [markdown]
# ## Ejemplos de uso

# %%
# Semifinal GS en clay — segmento dorado (threshold bajo, Kelly agresivo)
predict_match("Sinner J.", "Alcaraz C.", "Clay", "Outdoor", "Semifinals", "Grand Slam",
              p1_odds=2.40, p2_odds=1.60, bankroll=1000)

# %%
# Final hard GS — segmento bueno
predict_match("Sinner J.", "Alcaraz C.", "Hard", "Outdoor", "The Final", "Grand Slam",
              p1_odds=2.10, p2_odds=1.75, bankroll=1000)

# %%
# 1st Round — BLOQUEADO (no apuesta)
predict_match("Sinner J.", "Alcaraz C.", "Hard", "Outdoor", "1st Round", "ATP250",
              p1_odds=1.05, p2_odds=12.00, bankroll=1000)

# %%
# 2nd Round ATP500 clay — threshold permisivo
predict_match("Nadal R.", "Djokovic N.", "Clay", "Outdoor", "2nd Round", "ATP500",
              p1_odds=3.50, p2_odds=1.30, bankroll=500)

print("\n✅ Pipeline v4 completo")
