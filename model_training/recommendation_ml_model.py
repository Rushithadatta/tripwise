import argparse
import json
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# We will lightly reuse some logic; if the original recommendation_model changes heavily, consider factoring common utilities.

# ------------------ Utility Functions ------------------
EARTH_R = 6371.0

def haversine_array(lat1, lon1, lats, lons):
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r - lat1r
    dlon = lons_r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lats_r)*np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_R * c

def _parse_price(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ('', 'na', 'n/a', 'none', 'nil', '-'): return np.nan
    if any(x in s for x in ['free','no fee','nil']): return 0.0
    s = s.replace('rs','').replace('inr','').replace('₹','')
    s = s.replace('/-','').replace('approx','').replace('~','').replace('*','')
    s = s.replace(',','').replace('to','-').replace('–','-').replace('—','-')
    rng = s.split('-')
    nums = []
    for part in rng:
        part = part.strip()
        num = ''.join(ch for ch in part if (ch.isdigit() or ch=='.'))
        if num:
            try: nums.append(float(num))
            except: pass
    if not nums: return np.nan
    return float(sum(nums)/len(nums))

RENAME_MAP = {
    'place_name':'name','hotel_name':'name','restaurant':'name','lat':'latitude','lon':'longitude','long':'longitude'
}

NEARBY_COL_CANDIDATES = ['Nearby_Attractions','Nearby','Nearby Attractions']

CATEGORY_INFER = {
    'attractions':'attraction','attraction':'attraction','hotel':'hotel','hotels':'hotel','restaurant':'restaurant','restaurants':'restaurant'
}

PRICE_COLS = {
    'attraction':['Entry_Fee','Fee','Ticket','Price'],
    'hotel':['Price_per_night','Price','Cost_per_night'],
    'restaurant':['Avg_Cost','Average_Cost','Cost','Price']
}

def load_all(attractions_path, hotels_path, restaurants_path):
    def load_one(path, category_key):
        if not Path(path).exists():
            return pd.DataFrame(columns=['name','latitude','longitude','rating','price','category','nearby_list'])
        df = pd.read_excel(path)
        low = {c.lower():c for c in df.columns}
        # rename generically
        for k,v in list(RENAME_MAP.items()):
            if k in low:
                df.rename(columns={low[k]:v}, inplace=True)
        if 'name' not in df.columns:
            # fallback first text column
            for c in df.columns:
                if df[c].dtype==object:
                    df.rename(columns={c:'name'}, inplace=True)
                    break
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            df['latitude'] = np.nan
            df['longitude'] = np.nan
        # rating attempt
        rating_col = None
        for c in df.columns:
            if 'rating' in c.lower():
                rating_col = c; break
        if rating_col and rating_col != 'rating':
            df.rename(columns={rating_col:'rating'}, inplace=True)
        if 'rating' not in df.columns:
            df['rating'] = np.nan
        # price
        price_val = np.nan
        for cand in PRICE_COLS.get(category_key, []):
            if cand in df.columns:
                price_val = df[cand].apply(_parse_price)
                df['price'] = price_val
                break
        if 'price' not in df.columns:
            df['price'] = np.nan
        # nearby list
        nearby_col = None
        for cand in NEARBY_COL_CANDIDATES:
            if cand in df.columns:
                nearby_col = cand; break
        if nearby_col:
            df['nearby_list'] = df[nearby_col].fillna('').astype(str).str.lower().str.split('[;,]')
            df['nearby_list'] = df['nearby_list'].apply(lambda lst:[x.strip() for x in lst if x.strip()])
        else:
            df['nearby_list'] = [[] for _ in range(len(df))]
        df['category'] = category_key
        # Clean name
        df['name'] = df['name'].astype(str).str.strip()
        return df[['name','latitude','longitude','rating','price','category','nearby_list']]
    a = load_one(attractions_path,'attraction')
    h = load_one(hotels_path,'hotel')
    r = load_one(restaurants_path,'restaurant')
    return a,h,r

# ------------------ Feature Engineering ------------------

def build_feature_frame(anchor_row: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    lat1, lon1 = anchor_row['latitude'], anchor_row['longitude']
    cand = candidates.copy()
    cand['distance_km'] = haversine_array(lat1, lon1, cand['latitude'].fillna(lat1).values, cand['longitude'].fillna(lon1).values)
    # Nearby flag
    anchor_near = set(anchor_row.get('nearby_list', []))
    cand['nearby_flag'] = cand['name'].str.lower().isin(anchor_near).astype(int)
    # Category one-hot
    for cat in ['attraction','hotel','restaurant']:
        cand[f'cat_{cat}'] = (cand['category']==cat).astype(int)
    # Basic interactions
    cand['dist_price'] = cand['distance_km'] * cand['price'].fillna(cand['price'].median())
    cand['rating_price'] = cand['rating'].fillna(cand['rating'].median()) * cand['price'].fillna(cand['price'].median())
    cand['dist_over_rating'] = cand['distance_km'] / (cand['rating'].fillna(0)+0.1)
    # Ranks within category for distance and price
    cand['cat_dist_rank'] = cand.groupby('category')['distance_km'].rank(method='first')
    cand['cat_price_rank'] = cand.groupby('category')['price'].rank(method='first')
    # Fill NaN numerics
    num_cols = cand.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if cand[c].isna().any():
            cand[c] = cand[c].fillna(cand[c].median())
    feature_cols = [c for c in num_cols if c not in ['latitude','longitude']]
    return cand, feature_cols

# ------------------ Pseudo Label Generation ------------------

def pseudo_label(anchor: pd.Series, row: pd.Series) -> float:
    # Lower distance better, higher rating better, lower price better
    d = row['distance_km']
    rating = row['rating'] if not pd.isna(row['rating']) else 0
    price = row['price'] if not pd.isna(row['price']) else row['price']
    # normalized components (simple)
    d_score = 1 / (1 + d)
    r_score = rating / 5.0
    p_score = 1 / (1 + (price if price and price>0 else 0))
    base = 0.5*d_score + 0.35*r_score + 0.15*p_score
    if row.get('nearby_flag',0)==1:
        base += 0.05
    # category diversity slight bump (encourage hotels and restaurants vs only attractions)
    if row['category'] != anchor['category']:
        base += 0.02
    return float(base)

# ------------------ Training Pipeline ------------------

def build_training_set(a: pd.DataFrame, h: pd.DataFrame, r: pd.DataFrame, sample_anchors: int = 25, radius_km: float = 8.0) -> Tuple[pd.DataFrame, List[str]]:
    all_df = pd.concat([a,h,r], ignore_index=True)
    # Filter anchors with coordinates
    anchors = all_df.dropna(subset=['latitude','longitude'])
    if anchors.empty:
        raise ValueError('No anchors with coordinates to train on.')
    if sample_anchors < len(anchors):
        anchors = anchors.sample(sample_anchors, random_state=42)
    rows = []
    for _, anchor in anchors.iterrows():
        cand_pool = all_df[all_df['name'] != anchor['name']].copy()
        # Temporary feature build to compute distance
        cand_pool['distance_km'] = haversine_array(anchor['latitude'], anchor['longitude'],
                                                   cand_pool['latitude'].fillna(anchor['latitude']).values,
                                                   cand_pool['longitude'].fillna(anchor['longitude']).values)
        subset = cand_pool[cand_pool['distance_km'] <= radius_km].copy()
        if subset.empty:
            continue
        # Build full features
        feat_df, feat_cols = build_feature_frame(anchor, subset)
        # Generate pseudo labels
        feat_df['target'] = feat_df.apply(lambda r_: pseudo_label(anchor, r_), axis=1)
        # Keep anchor context features? For now only candidate relative features
        rows.append(feat_df[['name','category'] + feat_cols + ['target']])
    if not rows:
        raise ValueError('No training rows generated.')
    train_df = pd.concat(rows, ignore_index=True)
    # Deduplicate by (name,category) keeping mean target
    grouped = train_df.groupby(['name','category']).mean(numeric_only=True).reset_index()
    feature_cols_final = [c for c in grouped.columns if c not in ['name','category','target']]
    return grouped, feature_cols_final

# ------------------ Inference ------------------

def rank_candidates(model, anchor_row: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    feat_df, feat_cols = build_feature_frame(anchor_row, candidates)
    X = feat_df[feat_cols].values
    preds = model.predict(X)
    feat_df['ml_score'] = preds
    return feat_df.sort_values('ml_score', ascending=False)

# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description='ML-based recommender (RandomForest surrogate of heuristic)')
    parser.add_argument('--name', required=True, help='Anchor place name')
    parser.add_argument('--datasets', type=str, nargs=3, metavar=('ATTRACTIONS','HOTELS','RESTAURANTS'), default=['Attractions.xlsx','Hotel.xlsx','restaurants.xlsx'])
    parser.add_argument('--radius', type=float, default=5.0, help='Search radius in km for inference')
    parser.add_argument('--train-radius', type=float, default=8.0, help='Radius used for generating training samples')
    parser.add_argument('--anchors', type=int, default=30, help='Max anchor samples for training (synthetic)')
    parser.add_argument('--top', type=int, default=15, help='Top N results to show')
    parser.add_argument('--retrain', action='store_true', help='Force retrain model even if pickle exists')
    parser.add_argument('--model-path', type=str, default='ml_model.pkl', help='Path to save/load model')
    parser.add_argument('--format', choices=['table','json'], default='table', help='Output format')
    parser.add_argument('--explain', action='store_true', help='Show feature importances')
    args = parser.parse_args()

    a,h,r = load_all(*args.datasets)
    all_df = pd.concat([a,h,r], ignore_index=True)

    # Find anchor (partial match normalized)
    norm_input = args.name.lower().strip()
    anchor_row = None
    for _, row in all_df.iterrows():
        nrm = row['name'].lower()
        if nrm == norm_input or norm_input in nrm:
            anchor_row = row
            break
    if anchor_row is None:
        raise SystemExit(f"Anchor place '{args.name}' not found in datasets.")

    need_train = args.retrain or (not Path(args.model_path).exists())
    if need_train:
        train_df, feat_cols = build_training_set(a,h,r, sample_anchors=args.anchors, radius_km=args.train_radius)
        X = train_df[feat_cols].values
        y = train_df['target'].values
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=160, random_state=42, n_jobs=-1)
        model.fit(Xtr,ytr)
        pred = model.predict(Xte)
        r2 = r2_score(yte,pred)
        joblib.dump({'model':model,'features':feat_cols,'r2':r2}, args.model_path)
        print(f"[Train] Samples={len(train_df)} Features={len(feat_cols)} R2={r2:.4f}")
    else:
        bundle = joblib.load(args.model_path)
        model = bundle['model']
        feat_cols = bundle['features']
        print(f"[Load] Model loaded (features={len(feat_cols)} R2={bundle.get('r2','NA')})")

    # Build candidate set within radius (exclude anchor)
    candidates = all_df[all_df['name'] != anchor_row['name']].copy()
    candidates['distance_km'] = haversine_array(anchor_row['latitude'], anchor_row['longitude'],
                                               candidates['latitude'].fillna(anchor_row['latitude']).values,
                                               candidates['longitude'].fillna(anchor_row['longitude']).values)
    cand_in = candidates[candidates['distance_km'] <= args.radius].copy()
    if cand_in.empty:
        print('No candidates in radius; expanding to double radius once.')
        cand_in = candidates[candidates['distance_km'] <= args.radius*2].copy()
    ranked = rank_candidates(model, anchor_row, cand_in)
    ranked = ranked.head(args.top)

    label_map = {
        'attraction':'entry_fee(approx)',
        'hotel':'cost_per_night(approx)',
        'restaurant':'avg_cost(expected)'
    }

    if args.format == 'json':
        out = []
        for _, row in ranked.iterrows():
            price_label = label_map.get(row['category'],'price')
            out.append({
                'name': row['name'],
                'category': row['category'],
                'distance_km': round(float(row['distance_km']),4),
                'rating': None if pd.isna(row['rating']) else float(row['rating']),
                price_label: None if pd.isna(row['price']) else float(row['price']),
                'ml_score': round(float(row['ml_score']),6)
            })
        print(json.dumps({'results':out}, ensure_ascii=False, indent=2))
    else:
        display = ranked[['name','category','distance_km','rating','price','ml_score']].copy()
        # rename price column per category (simple per-row apply)
        def price_col_name(cat): return label_map.get(cat,'price')
        # We will pivot by rewriting column name dynamically in output lines
        print('\n=== ML RECOMMENDATIONS ===')
        for _, row in display.iterrows():
            price_label = price_col_name(row['category'])
            price_val = 'NA' if pd.isna(row['price']) else f"{row['price']:.2f}"
            print(f"{row['name']} | {row['category']} | dist={row['distance_km']:.2f} km | rating={row['rating'] if not pd.isna(row['rating']) else 'NA'} | {price_label}={price_val} | ml_score={row['ml_score']:.4f}")

    if args.explain and not need_train:
        if hasattr(model,'feature_importances_'):
            importances = sorted(list(zip(feat_cols, model.feature_importances_)), key=lambda x: x[1], reverse=True)[:20]
            print('\nTop Feature Importances:')
            for f,v in importances:
                print(f"{f}: {v:.4f}")

if __name__ == '__main__':
    main()
