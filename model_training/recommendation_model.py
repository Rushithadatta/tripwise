import pandas as pd
import numpy as np
import os
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import joblib
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False
from math import radians, cos, sin, asin, sqrt
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

EARTH_RADIUS_KM = 6371.0

# ---------------- Global ML Training Utilities -----------------
_GLOBAL_ML_MODEL = None
_GLOBAL_ML_FEATURES: List[str] = []
_GLOBAL_KMEANS_MODEL = None
_GLOBAL_KMEANS_SCALER = None
_GLOBAL_KMEANS_FEATURES: List[str] = []

def _build_kmeans_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    feat = pd.DataFrame(index=df.index)
    # Core numeric features
    for c in ['distance_km','rating','price','duration_hrs']:
        if c in df.columns:
            feat[c] = pd.to_numeric(df[c], errors='coerce')
            if feat[c].isna().all():
                feat[c] = 0
            else:
                feat[c] = feat[c].fillna(feat[c].median())
            cols.append(c)
    # Interaction features
    if 'distance_km' in feat.columns and 'rating' in feat.columns:
        feat['dist_over_rating'] = feat['distance_km'] / (feat['rating'] + 0.1)
        cols.append('dist_over_rating')
    if 'price' in feat.columns and 'distance_km' in feat.columns:
        feat['dist_price'] = feat['distance_km'] * feat['price']
        cols.append('dist_price')
    # Category one-hots
    if 'category' in df.columns:
        for cat in ['attraction','hotel','restaurant']:
            feat[f'cat_{cat}'] = (df['category']==cat).astype(int)
            cols.append(f'cat_{cat}')
    return feat[cols]

def train_global_kmeans(all_df: pd.DataFrame, k: int = 8, auto: bool = False, max_k: int = 15,
                        model_path: str = 'kmeans_global.pkl', retrain: bool = False, explain: bool = False,
                        metrics: bool = False):
    global _GLOBAL_KMEANS_MODEL, _GLOBAL_KMEANS_SCALER, _GLOBAL_KMEANS_FEATURES
    if not _ML_AVAILABLE:
        return
    if (not retrain) and os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            _GLOBAL_KMEANS_MODEL = bundle['model']
            _GLOBAL_KMEANS_SCALER = bundle['scaler']
            _GLOBAL_KMEANS_FEATURES = bundle['features']
            return
        except Exception:
            pass
    work = all_df.copy()
    # Derive a reference anchor to compute distances if not present (choose first attraction or first row)
    if 'distance_km' not in work.columns:
        # pick arbitrary anchor (median lat/lon) for global feature uniformity
        anchor_lat = work['latitude'].median()
        anchor_lon = work['longitude'].median()
        work['distance_km'] = haversine_vec(anchor_lat, anchor_lon, work['latitude'].values, work['longitude'].values)
    feat = _build_kmeans_features(work)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values)
    chosen_k = k
    if auto:
        sil_scores = {}
        for kk in range(2, max(3, min(max_k, max(3, len(work)//5)))+1):
            try:
                km_tmp = KMeans(n_clusters=kk, random_state=42, n_init='auto')
                labels_tmp = km_tmp.fit_predict(X)
                sil = silhouette_score(X, labels_tmp)
                sil_scores[kk] = sil
            except Exception:
                continue
        if sil_scores:
            chosen_k = max(sil_scores, key=lambda z: sil_scores[z])
            print(f"[KMeans Auto] Selected k={chosen_k} (silhouette={sil_scores[chosen_k]:.4f})")
    km = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    _GLOBAL_KMEANS_MODEL = km
    _GLOBAL_KMEANS_SCALER = scaler
    _GLOBAL_KMEANS_FEATURES = list(feat.columns)
    try:
        joblib.dump({'model': km, 'scaler': scaler, 'features': _GLOBAL_KMEANS_FEATURES}, model_path)
    except Exception:
        pass
    if explain:
        try:
            centers = km.cluster_centers_
            print("\n[KMeans Global Cluster Centers] (standardized space)")
            for idx, cvec in enumerate(centers):
                vals = ", ".join(f"{_GLOBAL_KMEANS_FEATURES[i]}={cvec[i]:.2f}" for i in range(len(cvec)))
                print(f"  Cluster {idx}: {vals}")
        except Exception:
            pass
    if metrics or explain:
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = float('nan')
        try:
            ch = calinski_harabasz_score(X, labels)
        except Exception:
            ch = float('nan')
        try:
            db = davies_bouldin_score(X, labels)
        except Exception:
            db = float('nan')
        inertia = km.inertia_ if hasattr(km,'inertia_') else float('nan')
        # Purity (category majority fraction) if category info available
        purity = float('nan')
        purity_details = []
        if 'category' in all_df.columns:
            # align labels length to all_df length after any filtering (feat index == work index)
            work_labels = pd.Series(labels, index=feat.index)
            cats = all_df.loc[feat.index, 'category']
            grouped = pd.DataFrame({'cluster': work_labels, 'category': cats})
            total = len(grouped)
            majority_sum = 0
            for cl, sub in grouped.groupby('cluster'):
                counts = sub['category'].value_counts()
                maj_cat = counts.idxmax()
                maj_count = counts.max()
                majority_sum += maj_count
                purity_details.append((cl, maj_cat, maj_count, len(sub)))
            if total > 0:
                purity = majority_sum / total
        print(f"[KMeans Global Metrics] k={chosen_k} samples={len(X)} silhouette={sil:.4f} CH={ch:.2f} DB={db:.3f} inertia={inertia:.2f} purity={purity:.4f}")
        if purity_details:
            print("[KMeans Cluster Purity Details]")
            for cl, maj_cat, maj_count, size in purity_details:
                frac = maj_count/size if size else 0
                print(f"  cluster={cl} size={size} majority={maj_cat} frac={frac:.3f}")

def _build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat['dist_price'] = df_feat['distance_km'] * df_feat['price'].fillna(df_feat['price'].median())
    df_feat['rating_price'] = df_feat['rating'].fillna(df_feat['rating'].median()) * df_feat['price'].fillna(df_feat['price'].median())
    df_feat['dist_over_rating'] = df_feat['distance_km'] / (df_feat['rating'].fillna(0) + 0.1)
    for cat in ['attraction','hotel','restaurant']:
        df_feat[f'cat_{cat}'] = (df_feat['category']==cat).astype(int)
    df_feat['cat_dist_rank'] = df_feat.groupby('category')['distance_km'].rank(method='first')
    df_feat['cat_price_rank'] = df_feat.groupby('category')['price'].rank(method='first')
    for col in df_feat.select_dtypes(include=[np.number]).columns:
        if df_feat[col].isna().any():
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())
    return df_feat

def train_global_ml(attractions: pd.DataFrame, hotels: pd.DataFrame, restaurants: pd.DataFrame,
                    anchors: int = 40, radius_km: float = 8.0, model_path: str = 'ml_global.pkl', retrain: bool = False,
                    metrics: bool = False, test_size: float = 0.2, cv_folds: int = 0):
    global _GLOBAL_ML_MODEL, _GLOBAL_ML_FEATURES
    if not _ML_AVAILABLE:
        return
    if (not retrain) and os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            _GLOBAL_ML_MODEL = bundle['model']
            _GLOBAL_ML_FEATURES = bundle['features']
            # If explain requested later we will print at inference time
            return
        except Exception:
            pass
    all_df = pd.concat([attractions, hotels, restaurants], ignore_index=True)
    candidates_all = all_df.dropna(subset=['latitude','longitude'])
    if candidates_all.empty:
        return
    if len(candidates_all) > anchors:
        anchors_df = candidates_all.sample(anchors, random_state=42)
    else:
        anchors_df = candidates_all
    rows = []
    for _, anchor in anchors_df.iterrows():
        pool = all_df[all_df['name'] != anchor['name']].copy()
        pool['distance_km'] = haversine_vec(anchor['latitude'], anchor['longitude'], pool['latitude'].fillna(anchor['latitude']).values, pool['longitude'].fillna(anchor['longitude']).values)
        subset = pool[pool['distance_km'] <= radius_km].copy()
        if subset.empty:
            continue
        subset['nearby_flag'] = 0
        feat = _build_ml_features(subset)
        # Ensure rating & price numeric filled (robust)
        if feat['rating'].isna().any():
            feat['rating'] = feat['rating'].fillna(0)
        if feat['price'].isna().any() or feat['price'].dtype == object:
            feat['price'] = pd.to_numeric(feat['price'], errors='coerce')
            feat['price'] = feat['price'].fillna(feat['price'].median() if not feat['price'].dropna().empty else 0)
        # Pseudo target heuristic
        d_score = 1 / (1 + feat['distance_km'])
        r_score = feat['rating'] / 5.0
        p_score = 1 / (1 + feat['price'].clip(lower=0))
        feat['target'] = 0.5*d_score + 0.35*r_score + 0.15*p_score
        feat = feat[~feat['target'].isna()]  # drop any residual NaN
        rows.append(feat)
    if not rows:
        return
    train_df = pd.concat(rows, ignore_index=True)
    feature_cols = ['distance_km','rating','price','dist_price','rating_price','dist_over_rating',
                    'cat_attraction','cat_hotel','cat_restaurant','cat_dist_rank','cat_price_rank']
    X = train_df[feature_cols].values
    y = train_df['target'].values
    # Train/test split for metrics (only if sufficient data)
    if len(train_df) > 50 and test_size > 0 and test_size < 0.9:
        try:
            Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size,random_state=42)
        except Exception:
            Xtr,Xte,ytr,yte = X,X,y,y
    else:
        Xtr,Xte,ytr,yte = X,X,y,y
    rf = RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1)
    rf.fit(Xtr,ytr)
    _GLOBAL_ML_MODEL = rf
    _GLOBAL_ML_FEATURES = feature_cols
    try:
        joblib.dump({'model': rf, 'features': feature_cols}, model_path)
    except Exception:
        pass
    globals()['_ML_MODEL_USED'] = 'global'
    if metrics:
        try:
            y_pred_test = rf.predict(Xte)
            r2 = r2_score(yte, y_pred_test) if len(np.unique(yte)) > 1 else float('nan')
            mae = mean_absolute_error(yte, y_pred_test)
            rmse = mean_squared_error(yte, y_pred_test, squared=False)
            corr = np.corrcoef(yte, y_pred_test)[0,1] if len(yte) > 1 else float('nan')
            print("\n[ML Global Metrics - Holdout]")
            print(f"  samples_train={len(ytr)} samples_test={len(yte)}")
            print(f"  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  Corr={corr:.4f}")
            if cv_folds and cv_folds > 1 and len(train_df) >= cv_folds:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                r2_list, mae_list, rmse_list = [], [], []
                for tr_idx, te_idx in kf.split(X):
                    Xtr_cv, Xte_cv = X[tr_idx], X[te_idx]
                    ytr_cv, yte_cv = y[tr_idx], y[te_idx]
                    rf_cv = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
                    rf_cv.fit(Xtr_cv, ytr_cv)
                    pred_cv = rf_cv.predict(Xte_cv)
                    if len(np.unique(yte_cv)) > 1:
                        r2_list.append(r2_score(yte_cv, pred_cv))
                    mae_list.append(mean_absolute_error(yte_cv, pred_cv))
                    rmse_list.append(mean_squared_error(yte_cv, pred_cv, squared=False))
                if r2_list:
                    print(f"[ML Global Metrics - {cv_folds}Fold CV]")
                    print(f"  R2_mean={np.mean(r2_list):.4f}  R2_std={np.std(r2_list):.4f}")
                print(f"  MAE_mean={np.mean(mae_list):.4f}  RMSE_mean={np.mean(rmse_list):.4f}")
        except Exception as me:
            print(f"[ML Global Metrics Warning] {me}")
    if globals().get('_ML_EXPLAIN'):
        try:
            importances = _GLOBAL_ML_MODEL.feature_importances_
            feat_names = _GLOBAL_ML_FEATURES
            order = np.argsort(importances)[::-1][:12]
            print("\n[ML Global Feature Importances]")
            for i in order:
                print(f"  {feat_names[i]}: {importances[i]:.4f}")
        except Exception:
            pass


def haversine_vec(lat1, lon1, lats2, lons2):
    """Vectorized haversine returning distances (km) between single point and arrays."""
    lat1, lon1 = map(radians, [lat1, lon1])
    lats2 = np.radians(lats2)
    lons2 = np.radians(lons2)
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

# Load datasets
def load_data(attractions_path='Attractions.xlsx', hotels_path='Hotel.xlsx', restaurants_path='restaurants.xlsx') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [attractions_path, hotels_path, restaurants_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required dataset not found: {p}")
    attractions = pd.read_excel(attractions_path)
    hotels = pd.read_excel(hotels_path)
    restaurants = pd.read_excel(restaurants_path)
    std_attractions = _standardize_df(attractions, 'attraction')
    std_hotels = _standardize_df(hotels, 'hotel')
    std_restaurants = _standardize_df(restaurants, 'restaurant')
    # Fallback coordinate enrichment for hotels/restaurants missing latitude/longitude
    city_means = std_attractions.groupby('city')[['latitude','longitude']].mean().dropna()
    # Pre-build attraction name index for fuzzy token contains matching
    attraction_names = std_attractions['name'].str.lower().tolist()
    attraction_coords = std_attractions[['name','latitude','longitude']]

    def match_tokens_get_centroid(tokens: List[str]):
        matches = attraction_coords[attraction_coords['name'].str.lower().isin(tokens)]
        # If no direct match, attempt partial containment
        if matches.empty:
            lowers = attraction_coords['name'].str.lower()
            mask = False
            for t in tokens:
                if len(t) < 3:
                    continue
                mask = lowers.str.contains(t) | mask
            matches = attraction_coords[mask]
        if matches.empty:
            return np.nan, np.nan
        return matches['latitude'].mean(), matches['longitude'].mean()

    def fill_missing_coords(df):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            if df['latitude'].isna().all() or df['longitude'].isna().all():
                pass
        else:
            # Add empty columns
            df['latitude'] = np.nan
            df['longitude'] = np.nan
        mask_missing = df['latitude'].isna() | df['longitude'].isna()
        if mask_missing.any():
            for idx in df[mask_missing].index:
                city = str(df.at[idx, 'city']).strip()
                # If we have nearby_list tokens try centroid of matched attractions first
                tokens = df.at[idx, 'nearby_list'] if 'nearby_list' in df.columns else []
                lat_c, lon_c = (np.nan, np.nan)
                if tokens:
                    lat_c, lon_c = match_tokens_get_centroid(tokens)
                if not np.isnan(lat_c) and not np.isnan(lon_c):
                    df.at[idx, 'latitude'] = lat_c
                    df.at[idx, 'longitude'] = lon_c
                elif city in city_means.index:
                    df.at[idx, 'latitude'] = city_means.loc[city, 'latitude']
                    df.at[idx, 'longitude'] = city_means.loc[city, 'longitude']
            # Global mean fallback
            if df['latitude'].isna().any():
                df['latitude'] = df['latitude'].fillna(std_attractions['latitude'].mean())
            if df['longitude'].isna().any():
                df['longitude'] = df['longitude'].fillna(std_attractions['longitude'].mean())
        # Jitter identical coordinate groups slightly to avoid zero distance ties
        dup_groups = df.groupby(['latitude','longitude']).size()
        jitter_targets = dup_groups[dup_groups > 3].index  # only jitter larger clusters
        for lat_val, lon_val in jitter_targets:
            sel = (df['latitude'] == lat_val) & (df['longitude'] == lon_val)
            n = sel.sum()
            jitter = (np.random.RandomState(42).randn(n,2) * 0.0005)
            df.loc[sel, 'latitude'] += jitter[:,0]
            df.loc[sel, 'longitude'] += jitter[:,1]
        return df
    std_hotels = fill_missing_coords(std_hotels)
    std_restaurants = fill_missing_coords(std_restaurants)
    return std_attractions, std_hotels, std_restaurants

def _standardize_df(df: pd.DataFrame, category: str) -> pd.DataFrame:
    # Normalize column names
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        'lat': 'latitude', 'lng': 'longitude', 'lon': 'longitude', 'long': 'longitude',
        'place_name': 'name', 'hotel_name': 'name', 'restaurant': 'name',
        'price_per_night': 'price', 'avg_cost': 'price'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # If name absent but place_name exists, rename above handled; else try other common variants
    if 'name' not in df.columns:
        for alt in ['place', 'place_name', 'title']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'name'})
                break
    required = {'name', 'latitude', 'longitude'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        # For attractions we must have coordinates; for others we create placeholders
        if category == 'attraction':
            raise ValueError(f"Dataset for {category} missing columns: {missing}")
        else:
            if 'latitude' not in df.columns:
                df['latitude'] = np.nan
            if 'longitude' not in df.columns:
                df['longitude'] = np.nan
            if 'name' not in df.columns:
                raise ValueError(f"Dataset for {category} missing name column after attempts")
    # Ensure numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    if category == 'attraction':
        df = df.dropna(subset=['latitude', 'longitude'])
    # Rating / price defaults
    # Normalize rating column name if capitalized
    if 'rating' not in df.columns:
        for ralt in ['ratings', 'review_rating']:
            if ralt in df.columns:
                df = df.rename(columns={ralt: 'rating'})
                break
    if 'rating' not in df.columns:
        df['rating'] = np.nan
    if 'price' not in df.columns:
        for palt in ['cost', 'avg_price']:
            if palt in df.columns:
                df = df.rename(columns={palt: 'price'})
                break
    if 'price' not in df.columns:
        df['price'] = np.nan
    # Nearby attractions parsing if exists
    for near_col in ['nearby_attractions', 'nearby', 'nearby_places', 'nearby_attraction', 'nearby_attraction_list']:
        if near_col in df.columns:
            df['nearby_list'] = df[near_col].fillna('').apply(_parse_nearby_list)
            break
    if 'nearby_list' not in df.columns:
        df['nearby_list'] = [[] for _ in range(len(df))]
    df['category'] = category
    # Category-specific price extraction logic
    original_price = None
    if category == 'attraction' and 'entry_fee' in df.columns:
        original_price = df['entry_fee']
    elif category == 'hotel':
        # Source from price_per_night if present (renamed earlier to price)
        if 'price_per_night' in df.columns:
            original_price = df['price_per_night']
        elif 'price' in df.columns:
            original_price = df['price']
    elif category == 'restaurant':
        if 'avg_cost' in df.columns:
            original_price = df['avg_cost']
        elif 'price' in df.columns:
            original_price = df['price']
    if original_price is None and 'price' in df.columns:
        original_price = df['price']
    if original_price is not None:
        df['price'] = original_price.apply(_parse_price)
    else:
        df['price'] = np.nan
    # Parse time_hrs / time or duration text forms
    if 'time_hrs' in df.columns:
        df['duration_hrs'] = df['time_hrs'].apply(_parse_duration)
    elif 'duration' in df.columns:
        df['duration_hrs'] = df['duration'].apply(_parse_duration)
    else:
        df['duration_hrs'] = np.nan
    return df

def _parse_price(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if not s or s in {'free', 'na', 'n/a', '-'}:
        return 0.0
    import re
    # Handle textual descriptors
    if 'small fee' in s:
        return 30.0  # heuristic nominal small fee
    if 'free/small fee' in s or 'free / small fee' in s:
        return 15.0
    # Replace unicode dashes with standard
    s_norm = s.replace('–','-').replace('—','-')
    # Extract all numbers
    nums = re.findall(r'\d+(?:\.\d+)?', s_norm)
    if not nums:
        return 0.0 if 'free' in s_norm else np.nan
    # Detect range like '20-50'
    if '-' in s_norm:
        parts = s_norm.split('-')
        nums_in_parts = [re.findall(r'\d+(?:\.\d+)?', p) for p in parts]
        flat = [float(n) for sub in nums_in_parts for n in sub]
        if len(flat) >= 2:
            return sum(flat[:2]) / 2.0
    # If multiple numbers but not a range, pick first (avoid concatenation like 20 + 50 -> 2050)
    return float(nums[0])

def _parse_duration(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if not s:
        return np.nan
    # Replace unicode dash
    s = s.replace('–', '-').replace('—', '-')
    # Extract ranges like 2-3 hrs or 2-3 hours
    import re
    pattern = r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)'
    m = re.search(pattern, s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2.0
    # Single number
    m2 = re.search(r'(\d+(?:\.\d+)?)', s)
    if m2:
        return float(m2.group(1))
    return np.nan

def _parse_nearby_list(val: Any) -> List[str]:
    if isinstance(val, list):
        return [str(v).strip().lower() for v in val if str(v).strip()]
    if isinstance(val, str):
        # Split on common delimiters
        parts = [p.strip().lower() for p in val.replace(';', ',').split(',')]
        return [p for p in parts if p]
    return []

# TOPSIS ranking function
def topsis(matrix: np.ndarray, weights: np.ndarray, criteria: List[str]) -> np.ndarray:
    if matrix.shape[1] != len(weights) or len(weights) != len(criteria):
        raise ValueError("Mismatch among matrix columns, weights, and criteria length")
    # Normalize decision matrix using vector normalization
    norm = np.linalg.norm(matrix, axis=0)
    norm[norm == 0] = 1
    norm_matrix = matrix / norm
    weights = weights / weights.sum()
    weighted = norm_matrix * weights
    # Determine ideal best/worst based on criterion type (max=benefit, min=cost)
    ideal_best = np.zeros(weighted.shape[1])
    ideal_worst = np.zeros(weighted.shape[1])
    for j, c in enumerate(criteria):
        col = weighted[:, j]
        if c == 'max':
            ideal_best[j] = col.max()
            ideal_worst[j] = col.min()
        elif c == 'min':
            ideal_best[j] = col.min()
            ideal_worst[j] = col.max()
        else:
            raise ValueError(f"Invalid criterion '{c}' (use 'max' or 'min')")
    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denom = d_best + d_worst
    denom[denom == 0] = 1
    return d_worst / denom

# Recommendation function
def recommend_nearby(place_name: str,
                     place_type: str,
                     radius_km: float = 5.0,
                     top_n: int = 5,
                     distance_weight: float = 0.4,
                     rating_weight: float = 0.35,
                     price_weight: float = 0.25,
                     duration_weight: float = 0.0,
                     use_duration: bool = True,
                     include_categories: str = 'attraction,hotel,restaurant',
                     min_per_category: int = 0,
                     budget_mode: bool = False,
                     max_attraction_fee: float = None,
                     max_restaurant_cost: float = None,
                     max_hotel_cost: float = None,
                     attractions_path='Attractions.xlsx',
                     hotels_path='Hotel.xlsx',
                     restaurants_path='restaurants.xlsx') -> pd.DataFrame:
    attractions, hotels, restaurants = load_data(attractions_path, hotels_path, restaurants_path)
    type_map = {'attraction': attractions, 'hotel': hotels, 'restaurant': restaurants}
    if place_type not in type_map:
        raise ValueError("place_type must be one of: attraction, hotel, restaurant")
    base_df = type_map[place_type]
    def _normalize(s: str) -> str:
        s = str(s).lower().strip()
        # remove parenthetical content
        import re
        s = re.sub(r"\([^)]*\)", "", s)
        # replace multiple spaces
        s = re.sub(r"\s+", " ", s)
        return s
    # Alias normalization (manual common aliases)
    alias_map = {
        'rk beach': 'ramakrishna beach',
        'rama krishna beach': 'ramakrishna beach',
        'ramakrishna beach': 'ramakrishna beach'
    }
    norm_in = _normalize(place_name)
    if norm_in in alias_map:
        norm_in = alias_map[norm_in]
    target_lower = norm_in
    base_df['__norm_name'] = base_df['name'].apply(_normalize)
    mask = base_df['__norm_name'] == target_lower
    if not mask.any():
        # Fallback: search other datasets
        for alt_type, df_alt in type_map.items():
            if alt_type == place_type:
                continue
            df_alt['__norm_name'] = df_alt['name'].apply(_normalize)
            alt_mask = df_alt['__norm_name'] == target_lower
            if alt_mask.any():
                place_type = alt_type
                base_df = df_alt
                mask = alt_mask
                break
        # Partial matching if still not found
    if not mask.any():
        # attempt partial within current base_df first
        partial = base_df['__norm_name'].str.contains(target_lower, regex=False)
        if partial.any():
            mask = partial
        else:
            # search across others
            for alt_type, df_alt in type_map.items():
                alt_partial = df_alt['__norm_name'].str.contains(target_lower, regex=False)
                if alt_partial.any():
                    place_type = alt_type
                    base_df = df_alt
                    mask = alt_partial
                    break
                    break
    if not mask.any():
        raise ValueError(f"Place '{place_name}' not found in any dataset (even partial match)")
    place = base_df[mask].iloc[0]
    lat1, lon1 = place['latitude'], place['longitude']
    # Combine all other places
    all_df = pd.concat([attractions, hotels, restaurants], ignore_index=True)
    all_df = all_df[all_df['name'].str.lower() != place_name.lower()].copy()
    # Filter by requested categories
    allowed = {c.strip().lower() for c in include_categories.split(',') if c.strip()}
    all_df = all_df[all_df['category'].isin(allowed)]
    dists = haversine_vec(lat1, lon1, all_df['latitude'].values, all_df['longitude'].values)
    all_df['distance_km'] = dists
    nearby = all_df[all_df['distance_km'] <= radius_km].copy()
    # Enforce hard 10 km cap for hotels and restaurants (even if user radius larger or augmentation would add farther ones)
    cap_mask = ~(((all_df['category'].isin(['hotel','restaurant'])) & (all_df['distance_km'] > 10.0)))
    # Apply cap also to the pool used later for augmentation by narrowing all_df accordingly
    all_df = all_df[cap_mask].copy()
    # Recompute nearby subset after cap for consistency
    nearby = all_df[all_df['distance_km'] <= radius_km].copy()
    # Apply budget filtering if enabled
    if budget_mode:
        def within_budget(row):
            p = row.get('price', np.nan)
            if np.isnan(p):
                return False  # drop unknown priced items in strict budget mode
            cat = row['category']
            if cat == 'attraction' and max_attraction_fee is not None:
                return p <= max_attraction_fee
            if cat == 'restaurant' and max_restaurant_cost is not None:
                return p <= max_restaurant_cost
            if cat == 'hotel' and max_hotel_cost is not None:
                return p <= max_hotel_cost
            # If no threshold provided for that category keep it
            return True
        filtered = nearby[nearby.apply(within_budget, axis=1)]
        # If filtering removed everything for a category, relax that category's constraint
        if filtered.empty:
            filtered = nearby  # total fallback
        nearby = filtered
    if nearby.empty:
        return pd.DataFrame(columns=['name', 'category', 'distance_km', 'rating', 'price', 'score'])

    # If a minimum per category is requested, pre-augment with nearest items outside radius
    if min_per_category > 0:
        allowed_cats = {c.strip().lower() for c in include_categories.split(',') if c.strip()}
        # Ensure top_n can accommodate requirement
        min_needed = len(allowed_cats) * min_per_category
        if top_n < min_needed:
            top_n = min_needed
        # Helper to fetch additional items for a category
        def augment_category(cat: str, current_df: pd.DataFrame) -> pd.DataFrame:
            current_count = (current_df['category'] == cat).sum()
            needed = min_per_category - current_count
            if needed <= 0:
                return current_df
            source_df = attractions if cat == 'attraction' else hotels if cat == 'hotel' else restaurants
            # Exclude already included names and the base place
            existing_names = set(current_df['name'].str.lower()) | {place_name.lower()}
            pool = source_df[~source_df['name'].str.lower().isin(existing_names)].copy()
            if pool.empty:
                return current_df
            pool['distance_km'] = haversine_vec(lat1, lon1, pool['latitude'].values, pool['longitude'].values)
            # Enforce 10 km cap for hotels and restaurants when augmenting
            if cat in ('hotel','restaurant'):
                pool = pool[pool['distance_km'] <= 10.0]
            if pool.empty:
                return current_df
            # Take nearest items within constraint
            addition = pool.sort_values('distance_km').head(needed)
            if not addition.empty:
                nearby_rows = addition.copy()
                nearby_rows['outside_radius_fill'] = 1
                nonlocal_nearby_cols = ['name','category','latitude','longitude','rating','price','distance_km']
                # Ensure columns existence
                for col in ['rating','price']:
                    if col not in nearby_rows.columns:
                        nearby_rows[col] = np.nan
                # Align columns present in nearby (some helper flag optional)
                # Build a list of columns to copy without introducing duplicates
                common_cols = [c for c in current_df.columns if c in nearby_rows.columns]
                if 'distance_km' in nearby_rows.columns and 'distance_km' not in common_cols:
                    common_cols.append('distance_km')
                nearby_extra = nearby_rows[common_cols].copy()
                # Ensure uniqueness after selection (defensive)
                nearby_extra = nearby_extra.loc[:, ~nearby_extra.columns.duplicated()]
                # Append retaining any extra fill flag for later debugging (ignored if not present in schema)
                nearby_extra['outside_radius_fill'] = 1
                # Add missing columns in original nearby to appended rows
                for col in current_df.columns:
                    if col not in nearby_extra.columns:
                        nearby_extra[col] = np.nan
                current_df = pd.concat([current_df, nearby_extra[current_df.columns]], ignore_index=True)
            return current_df
        for cat in allowed_cats:
            nearby = augment_category(cat, nearby)
        # Final prune: enforce hard 10 km cap for hotels/restaurants regardless of augmentation
        if not nearby.empty:
            mask_cap = ~(((nearby['category'].isin(['hotel','restaurant'])) & (nearby['distance_km'] > 10.0)))
            nearby = nearby[mask_cap].copy()
    # Score boost if candidate appears in base place nearby_list (if present)
    base_near = set(place.get('nearby_list', []))
    nearby['nearby_flag'] = nearby['name'].str.lower().isin(base_near).astype(int)
    # Prepare decision matrix: distance (cost), rating (benefit), price (cost)
    # Fill NaNs: rating -> median, price -> median, distance already computed
    for col, default in [('rating', nearby['rating'].median()), ('price', nearby['price'].median())]:
        nearby[col] = pd.to_numeric(nearby[col], errors='coerce')
        if nearby[col].isna().all():
            nearby[col] = 0 if col == 'price' else 0
        else:
            nearby[col] = nearby[col].fillna(default)
    # If budget_mode emphasize distance and price; optionally still use rating but at reduced weight
    if budget_mode:
        features = ['distance_km', 'price']
        criteria = ['min', 'min']
        weights_list = [max(distance_weight, 0.5), max(price_weight, 0.5)]  # ensure strong emphasis
    else:
        features = ['distance_km', 'rating', 'price']
        criteria = ['min', 'max', 'min']
        weights_list = [distance_weight, rating_weight, price_weight]
    if (not budget_mode) and use_duration and 'duration_hrs' in nearby.columns and not nearby['duration_hrs'].isna().all():
        # Duration treated as benefit? More time could mean richer experience. If user wants shorter visits, change to 'min'.
        nearby['duration_hrs'] = nearby['duration_hrs'].fillna(nearby['duration_hrs'].median())
        features.append('duration_hrs')
        criteria.append('max')
        weights_list.append(duration_weight if duration_weight > 0 else 0.15)  # default if not provided
    decision_matrix = nearby[features].to_numpy(dtype=float)
    weights = np.array(weights_list, dtype=float)
    topsis_scores = topsis(decision_matrix, weights, criteria)
    # Incorporate nearby flag (small bonus)
    nearby['heuristic_score'] = topsis_scores + 0.05 * nearby['nearby_flag']

    # Optional ML ensemble (prefer global model if trained)
    if globals().get('_ENABLE_ML', False) and _ML_AVAILABLE:
        # Build features similar to separate ML script
        def build_features(df: pd.DataFrame) -> pd.DataFrame:
            df_feat = df.copy()
            df_feat['dist_price'] = df_feat['distance_km'] * df_feat['price'].fillna(df_feat['price'].median())
            df_feat['rating_price'] = df_feat['rating'].fillna(df_feat['rating'].median()) * df_feat['price'].fillna(df_feat['price'].median())
            df_feat['dist_over_rating'] = df_feat['distance_km'] / (df_feat['rating'].fillna(0) + 0.1)
            for cat in ['attraction','hotel','restaurant']:
                df_feat[f'cat_{cat}'] = (df_feat['category']==cat).astype(int)
            df_feat['cat_dist_rank'] = df_feat.groupby('category')['distance_km'].rank(method='first')
            df_feat['cat_price_rank'] = df_feat.groupby('category')['price'].rank(method='first')
            # fill NAs
            for col in df_feat.select_dtypes(include=[np.number]).columns:
                if df_feat[col].isna().any():
                    df_feat[col] = df_feat[col].fillna(df_feat[col].median())
            return df_feat
        ml_mode = globals().get('_ML_MODE','rf')
        if ml_mode == 'kmeans' and globals().get('_GLOBAL_KMEANS_MODEL') is not None:
            # Use global KMeans cluster distance to centroid as score (invert distance)
            feat_k = _build_kmeans_features(nearby)
            scaler = globals().get('_GLOBAL_KMEANS_SCALER')
            km_model = globals().get('_GLOBAL_KMEANS_MODEL')
            if scaler is not None and km_model is not None and not feat_k.empty:
                Xk = scaler.transform(feat_k.values)
                centers = km_model.cluster_centers_
                labels = km_model.predict(Xk)
                # distance to own centroid
                dists_centroid = np.linalg.norm(Xk - centers[labels], axis=1)
                # Convert to similarity score (smaller distance -> higher score)
                # Use max distance for scaling
                if dists_centroid.max() > dists_centroid.min():
                    similarity = 1 - (dists_centroid - dists_centroid.min()) / (dists_centroid.max() - dists_centroid.min())
                else:
                    similarity = np.ones_like(dists_centroid)
                nearby['ml_score'] = similarity
                w = globals().get('_ML_WEIGHT',0.5)
                nearby['score'] = (1-w)*nearby['heuristic_score'] + w*nearby['ml_score']
                globals()['_ML_MODEL_USED'] = 'kmeans'
            else:
                nearby['score'] = nearby['heuristic_score']
        else:
            feat_df = build_features(nearby)
            feature_cols = ['distance_km','rating','price','dist_price','rating_price','dist_over_rating',
                            'cat_attraction','cat_hotel','cat_restaurant','cat_dist_rank','cat_price_rank','nearby_flag']
            global _GLOBAL_ML_MODEL, _GLOBAL_ML_FEATURES
            model_used = None
            if _GLOBAL_ML_MODEL is not None and _GLOBAL_ML_FEATURES:
                feats_present = [c for c in _GLOBAL_ML_FEATURES if c in feat_df.columns]
                preds = _GLOBAL_ML_MODEL.predict(feat_df[feats_present].values)
                feat_df['ml_score'] = preds
                model_used = 'global'
                if globals().get('_ML_EXPLAIN'):
                    try:
                        importances = _GLOBAL_ML_MODEL.feature_importances_
                        fnames = _GLOBAL_ML_FEATURES
                        order = np.argsort(importances)[::-1][:12]
                        print("\n[ML Global Feature Importances]")
                        for i in order:
                            print(f"  {fnames[i]}: {importances[i]:.4f}")
                    except Exception:
                        pass
            else:
                model_path = globals().get('_ML_MODEL_PATH','ml_model.pkl')
                retrain = globals().get('_ML_RETRAIN', False)
                model_bundle = None
                if (not os.path.exists(model_path)) or retrain:
                    tmp = feat_df.copy()
                    hs = tmp['heuristic_score']
                    if hs.max() > hs.min():
                        pseudo = (hs - hs.min()) / (hs.max() - hs.min())
                    else:
                        pseudo = np.zeros(len(tmp))
                    tmp['target'] = pseudo
                    X = tmp[feature_cols].values
                    y = tmp['target'].values
                    if len(tmp) > 10:
                        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
                    else:
                        Xtr,Xte,ytr,yte = X,X,y,y
                    rf = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
                    rf.fit(Xtr,ytr)
                    if globals().get('_ML_METRICS') and Xte is not Xtr:
                        try:
                            pred_local = rf.predict(Xte)
                            r2 = r2_score(yte, pred_local) if len(np.unique(yte)) > 1 else float('nan')
                            mae = mean_absolute_error(yte, pred_local)
                            rmse = mean_squared_error(yte, pred_local, squared=False)
                            corr = np.corrcoef(yte, pred_local)[0,1] if len(yte) > 1 else float('nan')
                            print("\n[ML Local Metrics]")
                            print(f"  samples_train={len(ytr)} samples_test={len(yte)}")
                            print(f"  R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  Corr={corr:.4f}")
                        except Exception as le:
                            print(f"[ML Local Metrics Warning] {le}")
                    model_bundle = {'model': rf, 'features': feature_cols}
                    try:
                        joblib.dump(model_bundle, model_path)
                    except Exception:
                        pass
                else:
                    try:
                        model_bundle = joblib.load(model_path)
                    except Exception:
                        model_bundle = None
                if model_bundle:
                    rf = model_bundle['model']
                    feats_present = [c for c in model_bundle['features'] if c in feat_df.columns]
                    preds = rf.predict(feat_df[feats_present].values)
                    feat_df['ml_score'] = preds
                    model_used = 'local'
            if model_used:
                w = globals().get('_ML_WEIGHT',0.5)
                nearby = feat_df.copy()
                nearby['score'] = (1-w)*nearby['heuristic_score'] + w*nearby['ml_score']
                globals()['_ML_MODEL_USED'] = model_used if globals().get('_ML_MODEL_USED') != 'global' else 'global'
                if globals().get('_ML_EXPLAIN') and model_used == 'local':
                    try:
                        importances = rf.feature_importances_
                        top_idx = np.argsort(importances)[::-1][:10]
                        feat_names = [c for c in feats_present]
                        print("\n[ML Local Feature Importances]")
                        for i in top_idx:
                            print(f"  {feat_names[i]}: {importances[i]:.4f}")
                    except Exception:
                        pass
            else:
                nearby['score'] = nearby['heuristic_score']
    else:
        nearby['score'] = nearby['heuristic_score']

    nearby = nearby.sort_values('score', ascending=False)
    result = nearby[['name', 'category', 'distance_km', 'rating', 'price', 'score']]
    # Absolute enforcement: drop any hotel/restaurant beyond 10 km
    if not result.empty:
        result = result[~(((result['category'].isin(['hotel','restaurant'])) & (result['distance_km'] > 10.0)))]
    requested_cats = {c.strip() for c in include_categories.split(',') if c.strip()}
    # Adaptive fill BEFORE trimming top_n so augmented categories are retained
    if {'hotel','restaurant'} & requested_cats:
        global_has = {'hotel': not hotels.empty, 'restaurant': not restaurants.empty}
        def fetch_nearest(cat: str):
            if cat not in requested_cats or not global_has.get(cat, False):
                return None
            if (result['category'] == cat).any():
                return None
            df_full = hotels if cat == 'hotel' else restaurants
            df_full = df_full[df_full['name'].str.lower() != place_name.lower()].copy()
            if df_full.empty:
                return None
            df_full['distance_km'] = haversine_vec(lat1, lon1, df_full['latitude'].values, df_full['longitude'].values)
            # Enforce 10 km cap for fallback fetch
            if cat in ('hotel','restaurant'):
                df_full = df_full[df_full['distance_km'] <= 10.0]
                if df_full.empty:
                    return None
            nearest = df_full.sort_values('distance_km').head(1).copy()
            nearest['score'] = 0.25 / (1 + nearest['distance_km'])  # modest placeholder score
            return nearest[['name','category','distance_km','rating','price','score']]
        for cat_need in ['hotel','restaurant']:
            extra = fetch_nearest(cat_need)
            if extra is not None:
                result = pd.concat([result, extra], ignore_index=True)
    if min_per_category > 0:
        buckets = []
        for cat in sorted(result['category'].unique()):
            subset = result[result['category'] == cat].head(min_per_category)
            buckets.append(subset)
        preselected = pd.concat(buckets).drop_duplicates()
        remaining_slots = max(top_n - len(preselected), 0)
        remaining = result[~result.index.isin(preselected.index)].head(remaining_slots)
        result = pd.concat([preselected, remaining]).sort_values('score', ascending=False)
        # If still over top_n due to duplicates removal mismatch, trim
        result = result.head(top_n)
    else:
        result = result.head(top_n)
    # Post-trim guarantee: ensure at least one hotel and one restaurant if requested
    guarantee_cats = [c for c in ['hotel','restaurant'] if c in requested_cats]
    missing = [c for c in guarantee_cats if not (result['category'] == c).any()]
    if missing:
        for cat in missing:
            df_full = hotels if cat == 'hotel' else restaurants
            df_full = df_full[df_full['name'].str.lower() != place_name.lower()].copy()
            if df_full.empty:
                continue
            df_full['distance_km'] = haversine_vec(lat1, lon1, df_full['latitude'].values, df_full['longitude'].values)
            # Enforce 10 km cap for guarantee injection
            if cat in ('hotel','restaurant'):
                df_full = df_full[df_full['distance_km'] <= 10.0]
                if df_full.empty:
                    continue
            nearest = df_full.sort_values('distance_km').head(1).copy()
            # Assign score slightly below max to avoid dominating
            inject_score = (result['score'].max() if not result.empty else 1.0) * 0.85
            nearest['score'] = inject_score
            result = pd.concat([result, nearest[['name','category','distance_km','rating','price','score']]])
        # Re-trim if we exceeded top_n: keep at least one per guarantee category
        if len(result) > top_n:
            # Sort so injected items retained (they have moderate high score)
            result = result.sort_values('score', ascending=False).drop_duplicates(subset=['name']).head(top_n)
    # Final cap enforcement before return
    if not result.empty:
        result = result[~(((result['category'].isin(['hotel','restaurant'])) & (result['distance_km'] > 10.0)))]
    return result

def parse_weights(weight_str: str) -> Tuple[float, float, float]:
    parts = weight_str.split(',')
    if len(parts) != 3:
        raise ValueError("Weights must be three comma-separated numbers: dist,rating,price")
    vals = [float(p) for p in parts]
    if sum(vals) == 0:
        raise ValueError("Sum of weights must be > 0")
    return tuple(vals)

def format_and_print_results(df: pd.DataFrame):
    if df.empty:
        print('No nearby places found within the specified radius.')
        return
    label_map = {
        'attraction': 'entry_fee(approx)',
        'restaurant': 'avg_cost(expected)',
        'hotel': 'cost_per_night(approx)'
    }
    common_cols = ['name', 'category', 'distance_km', 'rating']
    # Group by category and print separate tables with renamed price column
    outputs = []
    for cat, group in df.groupby('category'):
        col_label = label_map.get(cat, 'price')
        display = group.copy()
        display = display.rename(columns={'price': col_label})
        display = display[common_cols + [col_label, 'score']]
        outputs.append((cat, display))
    for idx, (cat, table) in enumerate(outputs):
        print(f"\n=== {cat.upper()} ===")
        print(table.to_string(index=False))

def results_to_json(df: pd.DataFrame) -> str:
    import json
    if df.empty:
        return json.dumps({"results": []}, ensure_ascii=False, indent=2)
    label_map = {
        'attraction': 'entry_fee(approx)',
        'restaurant': 'avg_cost(expected)',
        'hotel': 'cost_per_night(approx)'
    }
    out = []
    for _, row in df.iterrows():
        price_label = label_map.get(row['category'], 'price')
        item = {
            'name': row['name'],
            'category': row['category'],
            'distance_km': round(float(row['distance_km']), 4) if not pd.isna(row['distance_km']) else None,
            'rating': None if pd.isna(row['rating']) else float(row['rating']),
            price_label: None if pd.isna(row['price']) else float(row['price']),
            'score': round(float(row['score']), 6)
        }
        out.append(item)
    return json.dumps({"results": out}, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Nearby place recommender using Haversine + TOPSIS')
    parser.add_argument('--name', required=False, help='Name of the input place')
    parser.add_argument('--type', required=False, choices=['attraction', 'hotel', 'restaurant'], help='Type of the input place')
    parser.add_argument('--radius', type=float, default=5.0, help='Search radius in km (default 5)')
    parser.add_argument('--top', type=int, default=10, help='Number of recommendations to return (default 10)')
    parser.add_argument('--weights', type=str, default='0.4,0.35,0.25', help='Weights for distance,rating,price (and optionally duration if --duration-weight supplied)')
    parser.add_argument('--duration-weight', type=float, default=0.0, help='Optional weight for duration (if >0 enables duration criterion)')
    parser.add_argument('--no-duration', action='store_true', help='Disable using duration even if available')
    parser.add_argument('--include', type=str, default='attraction,hotel,restaurant', help='Comma-separated categories to include')
    parser.add_argument('--min-per-category', type=int, default=5, help='Guarantee at least N results per included category (default 5)')
    parser.add_argument('--budget-mode', action='store_true', help='Enable budget-focused recommendations (prioritize distance & price)')
    parser.add_argument('--max-attraction-fee', type=float, default=None, help='Maximum entry fee for attractions when budget-mode enabled')
    parser.add_argument('--max-restaurant-cost', type=float, default=None, help='Maximum avg cost for restaurants when budget-mode enabled')
    parser.add_argument('--max-hotel-cost', type=float, default=None, help='Maximum cost per night for hotels when budget-mode enabled')
    parser.add_argument('--datasets', type=str, nargs=3, metavar=('ATTRACTIONS','HOTELS','RESTAURANTS'), default=['Attractions.xlsx','Hotel.xlsx','restaurants.xlsx'], help='Custom dataset paths')
    parser.add_argument('--format', type=str, default='table', choices=['table','json'], help='Output format: table or json')
    # ML ensemble options
    parser.add_argument('--enable-ml', action='store_true', help='Enable ML model blending with heuristic score')
    parser.add_argument('--ml-model-path', type=str, default='ml_model.pkl', help='Path to persisted ML model bundle')
    parser.add_argument('--ml-retrain', action='store_true', help='Force retrain ML model (if enabled)')
    parser.add_argument('--ml-weight', type=float, default=0.5, help='Blend weight for ML score (0-1). final = (1-w)*heuristic + w*ml')
    parser.add_argument('--ml-global-train', action='store_true', help='Train a global ML model across multiple anchors before scoring')
    parser.add_argument('--ml-train-radius', type=float, default=8.0, help='Radius for generating global training samples')
    parser.add_argument('--ml-train-anchors', type=int, default=40, help='Maximum anchor samples for global ML training')
    parser.add_argument('--ml-explain', action='store_true', help='Print feature importances after ML scoring (global or local)')
    parser.add_argument('--ml-metrics', action='store_true', help='Compute and print ML regression metrics (pseudo-label fit)')
    parser.add_argument('--ml-test-size', type=float, default=0.2, help='Test split size for ML metrics (default 0.2)')
    parser.add_argument('--ml-cv-folds', type=int, default=0, help='Optional K-fold CV for global model metrics (0=disable)')
    parser.add_argument('--ml-mode', type=str, default='rf', choices=['rf','kmeans'], help='ML mode: rf (RandomForest) or kmeans (unsupervised)')
    parser.add_argument('--kmeans-clusters', type=int, default=8, help='Number of clusters for KMeans (ignored if --kmeans-auto)')
    parser.add_argument('--kmeans-auto', action='store_true', help='Auto-select K via silhouette score')
    parser.add_argument('--kmeans-max-clusters', type=int, default=15, help='Maximum clusters to try when auto-selecting K')
    args = parser.parse_args()
    # Set global flags for ML ensemble
    globals()['_ENABLE_ML'] = args.enable_ml
    globals()['_ML_MODEL_PATH'] = args.ml_model_path
    globals()['_ML_RETRAIN'] = args.ml_retrain
    globals()['_ML_WEIGHT'] = max(0.0, min(1.0, args.ml_weight))

    # If global training requested, load datasets early and train
    attractions = hotels = restaurants = None
    if args.ml_global_train and _ML_AVAILABLE and args.enable_ml and args.ml_mode == 'rf':
        try:
            attractions, hotels, restaurants = load_data(*args.datasets)
            train_global_ml(attractions, hotels, restaurants,
                            anchors=args.ml_train_anchors,
                            radius_km=args.ml_train_radius,
                            model_path='ml_global.pkl',
                            retrain=args.ml_retrain,
                            metrics=args.ml_metrics,
                            test_size=args.ml_test_size,
                            cv_folds=args.ml_cv_folds)
        except Exception as e:
            print(f"[Global ML Train Warning] {e}")
    # KMeans global training path
    if args.ml_global_train and _ML_AVAILABLE and args.enable_ml and args.ml_mode == 'kmeans':
        try:
            attractions, hotels, restaurants = load_data(*args.datasets)
            all_df = pd.concat([attractions, hotels, restaurants], ignore_index=True)
            train_global_kmeans(all_df,
                                k=args.kmeans_clusters,
                                auto=args.kmeans_auto,
                                max_k=args.kmeans_max_clusters,
                                model_path='kmeans_global.pkl',
                                retrain=args.ml_retrain,
                                explain=args.ml_explain)
            globals()['_ML_MODEL_USED'] = 'kmeans'
        except Exception as e:
            print(f"[Global KMeans Train Warning] {e}")
    globals()['_ML_EXPLAIN'] = args.ml_explain
    globals()['_ML_METRICS'] = args.ml_metrics
    globals()['_ML_MODE'] = args.ml_mode

    def auto_detect_type(place_name: str,
                         attractions_path='Attractions.xlsx',
                         hotels_path='Hotel.xlsx',
                         restaurants_path='restaurants.xlsx') -> str:
        try:
            a = pd.read_excel(attractions_path)
            if a.columns.str.lower().str.contains('place_name').any() or 'Place_Name' in a.columns:
                names = a.get('Place_Name', a.get('place_name'))
                low = names.astype(str).str.lower()
                target = place_name.lower().strip()
                if (low == target).any():
                    return 'attraction'
                # partial match heuristic
                if low.str.contains(target, regex=False).any():
                    return 'attraction'
        except Exception:
            pass
        try:
            h = pd.read_excel(hotels_path)
            if (h.columns.str.lower().str.contains('hotel_name')).any():
                low = h['Hotel_Name'].astype(str).str.lower()
                target = place_name.lower().strip()
                if (low == target).any() or low.str.contains(target, regex=False).any():
                    return 'hotel'
        except Exception:
            pass
        try:
            r = pd.read_excel(restaurants_path)
            if (r.columns.str.lower().str.contains('restaurant')).any():
                low = r['Restaurant'].astype(str).str.lower()
                target = place_name.lower().strip()
                if (low == target).any() or low.str.contains(target, regex=False).any():
                    return 'restaurant'
        except Exception:
            pass
        # Fallback: default to attraction
        return 'attraction'

    def run_interactive():
        place = input("Enter place name: ").strip()
        if not place:
            print("No place name provided.")
            return
        detected = auto_detect_type(place)
        dw, rw, pw = parse_weights('0.4,0.35,0.25')
        try:
            recs = recommend_nearby(place, detected,
                                    radius_km=5.0,
                                    top_n=10,
                                    distance_weight=dw,
                                    rating_weight=rw,
                                    price_weight=pw,
                                    include_categories='attraction,hotel,restaurant',
                                    min_per_category=5)
            print(results_to_json(recs))
        except Exception as ie:
            print(f"Error (interactive): {ie}")
        return
    # If no name provided, switch to interactive prompt
    if args.name and not args.type:
        # Auto-detect type
        args.type = auto_detect_type(args.name)
    if not args.name:
        return run_interactive()

    try:
        dw, rw, pw = parse_weights(args.weights)
        recs = recommend_nearby(args.name, args.type, args.radius, args.top, dw, rw, pw,
                                duration_weight=args.duration_weight,
                                use_duration=not args.no_duration,
                                include_categories=args.include,
                                min_per_category=args.min_per_category,
                                budget_mode=args.budget_mode,
                                max_attraction_fee=args.max_attraction_fee,
                                max_restaurant_cost=args.max_restaurant_cost,
                                max_hotel_cost=args.max_hotel_cost,
                                attractions_path=args.datasets[0], hotels_path=args.datasets[1], restaurants_path=args.datasets[2])
        if args.format == 'json':
            print(results_to_json(recs))
        else:
            format_and_print_results(recs)
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == '__main__':
    main()
