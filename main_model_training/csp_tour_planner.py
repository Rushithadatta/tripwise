
# Print logs to stderr, only final output to stdout
import pandas as pd
import numpy as np
import sys
import os
import math
import random
import json

def log(msg):
    print(msg, file=sys.stderr)
# Add the recommendation_model_training folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../recommendation_model_training')))
from preprocess_data import preprocess_all_datasets
from data_loading import load_datasets, files, data_folder


# Fetch user input from backend (sys.argv[1] as JSON)
if len(sys.argv) > 1:
    try:
        raw_input = json.loads(sys.argv[1])
        # Map backend keys to model keys
        user_input = {
            'source': raw_input.get('source', ''),
            'destination': raw_input.get('destination', ''),
            'num_people': int(raw_input.get('num_people', 1)),  # fallback to 1 if not provided
            'num_days': int(raw_input.get('num_days', 1)),
            'budget': int(raw_input.get('budget', 0)),
            'transport': raw_input.get('transport', ''),
            'stay_type': raw_input.get('stay_type', ''),
            'food': raw_input.get('food', '')
        }
    except Exception as e:
        print(json.dumps({'error': f'Failed to parse input: {e}'}))
        sys.exit(1)
else:
    # Fallback to default for manual testing
    user_input = {
        'source': 'Tirupati',
        'destination': 'Visakhapatnam',
        'num_people': 2,
        'num_days': 4,
        'budget': 25000,
        'transport': 'flight',
        'stay_type': 'AC',
        'food': 'Veg'
    }

def get_cost(row, num_people):
    # Example: sum up relevant cost columns, scale by people
    cost_cols = [col for col in row.index if ('price' in col or 'cost' in col or 'fare' in col or 'rate' in col)]
    total = 0
    for col in cost_cols:
        try:
            total += float(row[col])
        except Exception:
            continue
    return total * num_people

def safe_name(df, idx):
    row = df.loc[idx]
    candidates = [
        'name', 'attraction_name', 'attraction', 'place_name', 'place',
        'hotel_name', 'hotel', 'restaurant_name', 'restaurant', 'title', 'location'
    ]
    for c in candidates:
        if c in df.columns:
            val = row[c]
            if pd.notna(val) and str(val).strip() and str(val) != 'Unknown':
                return str(val)
    # Fallback to first textual column
    for c in df.columns:
        if df[c].dtype == 'O':
            val = row[c]
            if pd.notna(val) and str(val).strip() and str(val) != 'Unknown':
                return str(val)
    return str(idx)

def get_rating(df, idx):
    row = df.loc[idx]
    candidates = ['rating', 'review', 'score']
    for c in candidates:
        if c in df.columns:
            val = row[c]
            if pd.notna(val):
                try:
                    return float(val)
                except:
                    continue
    return None

def get_timings(df, idx):
    row = df.loc[idx]
    candidates = ['timings', 'time', 'hours', 'opening_hours']
    for c in candidates:
        if c in df.columns:
            val = row[c]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return None

def get_season(df, idx):
    row = df.loc[idx]
    candidates = ['season', 'best_season', 'visit_season']
    for c in candidates:
        if c in df.columns:
            val = row[c]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return None

def get_avg_cost(df, idx, num_people=1):
    row = df.loc[idx]
    candidates = ['avg_cost', 'price', 'cost', 'fare', 'rate', 'amount', 'entry_fee']
    for c in candidates:
        if c in df.columns:
            val = row[c]
            if pd.notna(val):
                val_str = str(val).strip()
                if val_str.lower() == 'free':
                    return 0.0
                # Extract numeric part, e.g., '₹90 ropeway' -> 90
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', val_str)
                if match:
                    try:
                        cost = float(match.group(1))
                        return cost * num_people
                    except:
                        continue
                # Try direct float
                try:
                    cost = float(val)
                    return cost * num_people
                except:
                    continue
    return None

def get_cost(row, num_people):
    # Example: sum up relevant cost columns, scale by people
    cost_cols = [col for col in row.index if ('price' in col or 'cost' in col or 'fare' in col or 'rate' in col)]
    total = 0
    for col in cost_cols:
        try:
            total += float(row[col])
        except Exception:
            continue
    return total * num_people

def safe_name(df, idx):
    row = df.loc[idx]
    candidates = [
        'name', 'attraction_name', 'attraction', 'place_name', 'place',
        'hotel_name', 'hotel', 'restaurant_name', 'restaurant', 'title', 'location'
    ]
    for c in candidates:
        if c in df.columns:
            val = row[c]
            if pd.notna(val) and str(val).strip() and str(val) != 'Unknown':
                return str(val)
    # Fallback to first textual column
    for c in df.columns:
        if df[c].dtype == 'O':
            val = row[c]
            if pd.notna(val) and str(val).strip() and str(val) != 'Unknown':
                return str(val)
    return str(idx)

def normalize_name(s: str) -> str:
    import re
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def safe_name_for(df, idx, kind: str):
    kind = (kind or '').lower()
    if kind == 'restaurant':
        candidates = ['restaurant_name', 'restaurant', 'name', 'place_name', 'business_name', 'shop_name', 'eatery', 'hotel_name', 'outlet', 'brand', 'establishment']
        skip_cols = ['location', 'area', 'neighborhood', 'junction', 'colony', 'road', 'street', 'zone', 'locality', 'address', 'district', 'city', 'state', 'ward', 'pincode', 'landmark']
    elif kind == 'hotel':
        candidates = ['hotel_name', 'name', 'place_name']
        skip_cols = ['location', 'area', 'neighborhood', 'junction', 'colony']
    elif kind == 'attraction':
        candidates = ['attraction_name', 'name', 'attraction', 'place_name', 'place', 'title']
        skip_cols = ['location', 'area']
    else:
        return safe_name(df, idx)
    # Score columns and values to pick best human name
    col_pos = ['restaurant', 'eatery', 'outlet', 'brand', 'business', 'cafe', 'hotel', 'dhaba', 'mess', 'canteen', 'bar', 'baker', 'bakery', 'sweet', 'veg', 'nonveg', 'food', 'shop']
    col_neg = skip_cols
    val_neg_tokens = ['colony', 'junction', 'beach', 'park', 'road', 'street', 'nagar', 'puram', 'layout', 'valley', 'gate', 'circle', 'ward', 'zone', 'city', 'state']
    best = None
    best_score = -1e9
    for c in df.columns:
        if df[c].dtype != 'O':
            continue
        val = df.loc[idx, c]
        if pd.isna(val):
            continue
        sval = str(val).strip()
        if not sval or sval == 'Unknown':
            continue
        colname = c.lower()
        # Base score
        score = 0
        score += sum(1 for t in col_pos if t in colname) * 5
        score -= sum(1 for t in col_neg if t in colname) * 5
        # Penalize values that look like localities
        sval_l = sval.lower()
        score -= sum(1 for t in val_neg_tokens if t in sval_l) * 4
        # Prefer names with 1-4 words and reasonable length
        words = [w for w in sval.split() if w]
        if 1 <= len(words) <= 4:
            score += 3
        if 3 <= len(sval) <= 40:
            score += 2
        # Mild boost if Title Case-like
        if any(ch.isupper() for ch in sval):
            score += 1
        # Final candidate weight if in preferred list
        if c in candidates:
            score += 5
        if score > best_score:
            best_score = score
            best = sval
    if best:
        return best
    # Last resort fallbacks
    for c in candidates:
        if c in df.columns and not any(k in c for k in skip_cols):
            val = df.loc[idx, c]
            if pd.notna(val) and str(val).strip() and str(val) != 'Unknown':
                return str(val)
    for c in df.columns:
        if df[c].dtype == 'O' and not any(k in c for k in skip_cols):
            val = df.loc[idx, c]
            if pd.notna(val) and str(val).strip() and str(val) != 'Unknown':
                return str(val)
    return safe_name(df, idx)

def get_coords(df):
    lat_col = next((c for c in df.columns if c in ['lat', 'latitude', 'latitute', 'y', 'lat_deg']), None)
    lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude', 'x', 'long', 'lon_deg']), None)
    return lat_col, lon_col

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return float('inf')
    R = 6371.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def compute_distance_matrix(attractions, lat_col, lon_col):
    indices = attractions.index.tolist()
    n = len(indices)
    dist_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            idx1, idx2 = indices[i], indices[j]
            lat1, lon1 = attractions.loc[idx1, lat_col], attractions.loc[idx1, lon_col]
            lat2, lon2 = attractions.loc[idx2, lat_col], attractions.loc[idx2, lon_col]
            d = haversine_km(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix, indices

def topsis_score(df, benefit_cols=None, cost_cols=None, weights=None):
    if df.empty:
        return pd.Series(dtype=float)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if benefit_cols is None and cost_cols is None:
        benefit_cols = [c for c in num_df.columns if any(k in c for k in ['rating','score','popularity','review'])]
        cost_cols = [c for c in num_df.columns if any(k in c for k in ['price','cost','fare','rate'])]
    benefit_cols = [c for c in (benefit_cols or []) if c in num_df.columns]
    cost_cols = [c for c in (cost_cols or []) if c in num_df.columns]
    cols = benefit_cols + cost_cols
    if not cols:
        return pd.Series(0.5, index=df.index)
    M = num_df[cols].fillna(num_df[cols].median())
    norm = np.sqrt((M**2).sum(axis=0))
    norm[norm == 0] = 1
    V = M / norm
    if weights is None:
        weights = np.ones(len(cols)) / len(cols)
    W = V * weights
    ideal_pos = pd.Series({c: (W[c].max() if c in benefit_cols else W[c].min()) for c in cols})
    ideal_neg = pd.Series({c: (W[c].min() if c in benefit_cols else W[c].max()) for c in cols})
    S_pos = np.sqrt(((W - ideal_pos)**2).sum(axis=1))
    S_neg = np.sqrt(((W - ideal_neg)**2).sum(axis=1))
    C = S_neg / (S_pos + S_neg + 1e-9)
    return C.reindex(df.index).fillna(0.5)

def inverse_cost_score(df, num_people):
    if df.empty:
        return pd.Series(dtype=float)
    costs = df.apply(lambda r: get_cost(r, num_people), axis=1)
    costs = costs.replace([np.inf, -np.inf], np.nan).fillna(costs.median() if not costs.empty else 1.0)
    maxc, minc = costs.max(), costs.min()
    if maxc == minc:
        return pd.Series(0.5, index=df.index)
    inv = (maxc - costs) / (maxc - minc)
    return inv

def filter_hotels_by_stay(hotels, stay_type):
    if not stay_type:
        return hotels
    st = str(stay_type).lower()
    cols = hotels.columns
    mask = pd.Series(True, index=hotels.index)
    ac_cols = [c for c in cols if 'ac' in c]
    if 'ac' in st and ac_cols:
        col = ac_cols[0]
        mask = mask & hotels[col].astype(str).str.contains('1|true|yes|ac', case=False, regex=True)
    if 'non' in st and ac_cols:
        col = ac_cols[0]
        mask = mask & ~hotels[col].astype(str).str.contains('1|true|yes|ac', case=False, regex=True)
    return hotels[mask]

def filter_restaurants_by_food(restaurants, food):
    if not food:
        return restaurants
    f = str(food).lower()
    cols = restaurants.columns
    food_cols = [c for c in cols if any(k in c for k in ['veg','nonveg','non_veg','type','cuisine'])]
    if not food_cols:
        return restaurants
    col = food_cols[0]
    if 'veg' in f and 'non' not in f:
        return restaurants[restaurants[col].astype(str).str.contains('veg', case=False, na=False)]
    if 'non' in f:
        return restaurants[restaurants[col].astype(str).str.contains('non', case=False, na=False)]
    return restaurants

def choose_triplets_within_radius(attractions, scores, radius_km=50, max_seed=50, dist_matrix=None, indices=None):
    lat_col, lon_col = get_coords(attractions)
    if not lat_col or not lon_col or dist_matrix is None or indices is None:
        top = scores.sort_values(ascending=False).index.tolist()
        triplets = [top[i:i+3] for i in range(0, len(top), 3) if len(top[i:i+3])==3]
        return triplets
    ordered = scores.sort_values(ascending=False).index.tolist()
    triplets = []
    used = set()
    idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
    for seed in ordered[:max_seed]:
        if seed in used:
            continue
        seed_pos = idx_to_pos[seed]
        candidates = [idx for idx in ordered if idx != seed and idx not in used]
        close = []
        for idx in candidates:
            idx_pos = idx_to_pos[idx]
            d = dist_matrix[seed_pos, idx_pos]
            if d <= radius_km:
                close.append(idx)
            if len(close) >= 10:
                break
        found = None
        for i in range(len(close)):
            for j in range(i+1, len(close)):
                a, b = close[i], close[j]
                a_pos, b_pos = idx_to_pos[a], idx_to_pos[b]
                d_ab = dist_matrix[a_pos, b_pos]
                if d_ab <= radius_km:
                    found = (seed, a, b)
                    break
            if found:
                break
        if found:
            triplets.append(list(found))
            used.update(found)
        if len(triplets) >= 20:
            break
    return triplets

def csp_tour_plan(user_input, datasets):
    # Helper: get index by name
    def get_index_by_name(df, name):
        for idx in df.index:
            if safe_name(df, idx).lower().strip() == name.lower().strip():
                return idx
        return None

    num_days = user_input['num_days']
    budget = user_input['budget']
    num_people = user_input['num_people']
    attractions = datasets['attractions']
    hotels = datasets['hotels']
    restaurants = datasets['restaurants']
    transport = datasets['transport']

    # Filter attractions to those near Visakhapatnam (within 100 km)
    lat_col, lon_col = get_coords(attractions)
    if lat_col and lon_col:
        vizag_lat, vizag_lon = 17.6868, 83.2185  # Visakhapatnam coordinates
        attractions = attractions[attractions.apply(lambda row: haversine_km(row[lat_col], row[lon_col], vizag_lat, vizag_lon) <= 100, axis=1)].copy()
        if attractions.empty:
            print("Warning: No attractions found within 100 km of Visakhapatnam.", file=sys.stderr)
            return []

    # TOPSIS scores
    attr_scores_t = topsis_score(attractions)
    attr_scores_c = inverse_cost_score(attractions, num_people)
    attr_scores = 0.7 * attr_scores_t.add(0, fill_value=0) + 0.3 * attr_scores_c.add(0, fill_value=0)
    hotel_candidates = filter_hotels_by_stay(hotels, user_input.get('stay_type'))
    rest_candidates = filter_restaurants_by_food(restaurants, user_input.get('food'))
    hotel_scores_t = topsis_score(hotel_candidates)
    hotel_scores_c = inverse_cost_score(hotel_candidates, num_people)
    hotel_scores = 0.7 * hotel_scores_t.add(0, fill_value=0) + 0.3 * hotel_scores_c.add(0, fill_value=0)
    rest_scores_t = topsis_score(rest_candidates)
    rest_scores_c = inverse_cost_score(rest_candidates, num_people)
    rest_scores = 0.7 * rest_scores_t.add(0, fill_value=0) + 0.3 * rest_scores_c.add(0, fill_value=0)
    # Full restaurant scores for fallback when filtered set is exhausted
    rest_scores_all = 0.7 * topsis_score(restaurants).add(0, fill_value=0) + 0.3 * inverse_cost_score(restaurants, num_people).add(0, fill_value=0)

    lat_col, lon_col = get_coords(attractions)
    dist_matrix = None
    indices = None
    if lat_col and lon_col:
        dist_matrix, indices = compute_distance_matrix(attractions, lat_col, lon_col)

    # Build attraction triplets per day
    triplets = choose_triplets_within_radius(attractions, attr_scores, radius_km=30, dist_matrix=dist_matrix, indices=indices)
    # Only use valid clusters within radius. If not enough, warn and proceed with fewer days.
    if len(triplets) < num_days:
        print(f"Warning: Only {len(triplets)} valid attraction clusters found within 20km radius. Returning fewer days.", file=sys.stderr)

    total_cost = 0.0
    day_plans = []
    # Ensure travel_costs_df is always defined
    travel_costs_df = datasets.get('travel_costs', pd.DataFrame())

    # Misc cost: fixed per person per day (e.g., 100 INR)
    FIXED_MISC_PER_PERSON_PER_DAY = 100
    misc_per_person_per_day = FIXED_MISC_PER_PERSON_PER_DAY
    misc_total = misc_per_person_per_day * max(1, int(user_input.get('num_people',1))) * max(1, int(num_days))

    total_cost += misc_total

    used_attr = set()
    used_attr_names = set()
    used_hotels = set()
    used_hotel_names = set()
    used_restaurants = set()
    used_restaurant_names = set()
    used_hospitals = set()
    used_rentals = set()
    used_police = set()
    used_camera = set()
    used_local = set()
    # Precompute normalized names for deduplication by display names
    attr_name_map = {idx: normalize_name(safe_name_for(attractions, idx, 'attraction')) for idx in attractions.index}
    hotel_name_map = {idx: normalize_name(safe_name_for(hotels, idx, 'hotel')) for idx in hotels.index}
    # lat_col, lon_col = get_coords(attractions)  # Already computed above
    # dist_matrix = None
    # indices = None
    # if lat_col and lon_col:
    #     dist_matrix, indices = compute_distance_matrix(attractions, lat_col, lon_col)
    rest_name_map = {idx: normalize_name(safe_name_for(restaurants, idx, 'restaurant')) for idx in restaurants.index}

    # Precompute cheaper triplets list (cost and cluster already satisfies 30km)
    pre_costed_triplets = []
    if 'triplets' in locals() and triplets:
        for tri in triplets:
            c = sum(get_cost(attractions.loc[a], num_people) for a in tri)
            pre_costed_triplets.append((c, tri))
        pre_costed_triplets.sort(key=lambda x: x[0])

    day_plans = []
    used_attr = set()
    used_attr_names = set()
    for d in range(num_days):
        trip = None
        # Find first triplet that does not overlap with used attractions
        for tri in triplets:
            if any(a in used_attr for a in tri):
                continue
            # Custom logic: If Araku Valley and Indira Gandhi Zoological Park both in triplet, replace IGZP with Katika Waterfalls if possible
            araku_idx = get_index_by_name(attractions, "Araku Valley")
            igzp_idx = get_index_by_name(attractions, "Indira Gandhi Zoological Park")
            katika_idx = get_index_by_name(attractions, "Katika Waterfalls")
            if araku_idx in tri and igzp_idx in tri and katika_idx in tri:
                tri = [a if a != igzp_idx else katika_idx for a in tri]
            elif araku_idx in tri and igzp_idx in tri and katika_idx is not None:
                tri = [a for a in tri if a != igzp_idx]
                if katika_idx not in tri:
                    tri.append(katika_idx)
            trip = tri
            used_attr.update(trip)
            used_attr_names.update([attr_name_map.get(a, "") for a in trip])
            break
        if trip is None:
            trip = []
    # Do not mark attractions as used yet; final trip may change after budget adjustments

        if trip is None:
            # No valid triplet found for this day, skip
            continue
        if d >= 1 and lat_col and lon_col:
            prev_points = day_plans[-1]['attractions']
            prev_lat = [attractions.loc[i, lat_col] for i in prev_points]
            prev_lon = [attractions.loc[i, lon_col] for i in prev_points]
            cur_lat = [attractions.loc[i, lat_col] for i in trip]
            cur_lon = [attractions.loc[i, lon_col] for i in trip]
            prev_centroid = (np.nanmean(prev_lat), np.nanmean(prev_lon))
            cur_centroid = (np.nanmean(cur_lat), np.nanmean(cur_lon))
            dist_centroids = haversine_km(prev_centroid[0], prev_centroid[1], cur_centroid[0], cur_centroid[1])
        else:
            dist_centroids = 0.0

        hotel_pick_df = hotel_candidates.copy()
        rest_pick_df = rest_candidates.copy()

        # Compute centroid of day's attractions
        centroid_lat = attractions.loc[trip, 'lat'].mean()
        centroid_lon = attractions.loc[trip, 'lon'].mean()

        # Hotel proximity score (inverse distance)
        hotel_proximity = hotel_candidates.apply(lambda r: 1 / (1 + haversine_km(r.get('lat', 0), r.get('lon', 0), centroid_lat, centroid_lon)) if pd.notna(r.get('lat', None)) and pd.notna(r.get('lon', None)) else 0, axis=1)
        hotel_scores_with_prox = hotel_scores.add(hotel_proximity, fill_value=0)

        # Restaurant proximity score (count of matching nearby attractions)
        rest_proximity = rest_candidates.apply(lambda r: sum(1 for a in trip if safe_name(attractions, a).lower() in str(r['nearby_attractions']).lower()), axis=1)
        rest_scores_with_prox = rest_scores.add(rest_proximity, fill_value=0)

        hotel_order = hotel_scores_with_prox.sort_values(ascending=False).index.tolist() if not hotel_scores_with_prox.empty else (hotel_pick_df.index.tolist() if not hotel_pick_df.empty else hotels.index.tolist())
        rest_order = rest_scores_with_prox.sort_values(ascending=False).index.tolist() if not rest_scores_with_prox.empty else (rest_pick_df.index.tolist() if not rest_pick_df.empty else restaurants.index.tolist())
        hotel_idx = next((hid for hid in hotel_order if (hid not in used_hotels) and (hotel_name_map.get(hid, "") not in used_hotel_names)), (hotel_order[0] if hotel_order else None))
        rest_idx = next((rid for rid in rest_order if (rid not in used_restaurants) and (rest_name_map.get(rid, "") not in used_restaurant_names)), None)
        if rest_idx is None:
            # Fallback to full restaurant pool excluding used
            all_order = rest_scores_all.sort_values(ascending=False).index.tolist() if not rest_scores_all.empty else restaurants.index.tolist()
            rest_idx = next((rid for rid in all_order if (rid not in used_restaurants) and (rest_name_map.get(rid, "") not in used_restaurant_names)), (all_order[0] if all_order else None))

        if d == 0:
            pass
        elif d == 1:
            if dist_centroids > 70 and hotel_idx is not None:
                prev_hotel = day_plans[-1].get('hotel') if day_plans else None
                ordered_hotels = [hid for hid in hotel_order if hid != prev_hotel and hid not in used_hotels]
                for hid in ordered_hotels:
                        hotel_idx = hid
                        break
            else:
                hotel_idx = None
        else:
            # For days beyond Day 2, recommend a hotel normally (no skip)
            pass

        day_cost = 0.0
        for a in trip:
            day_cost += get_cost(attractions.loc[a], num_people)
        if hotel_idx is not None:
            day_cost += get_cost(hotels.loc[hotel_idx], num_people)
        if rest_idx is not None:
            day_cost += get_cost(restaurants.loc[rest_idx], num_people)

        # Local travel recommendation per day from travel_costs
        local_pick = None
        local_source = None
        if not travel_costs_df.empty:
            local_scores = topsis_score(travel_costs_df)
            order_local = local_scores.sort_values(ascending=False).index.tolist() if not local_scores.empty else travel_costs_df.index.tolist()
            local_pick = next((lid for lid in order_local if lid not in used_local), (order_local[0] if order_local else None))
            local_source = 'travel_costs'
            # Local travel cost treated as per-day group cost
            if local_pick is not None:
                day_cost += get_cost(travel_costs_df.loc[local_pick], 1)

        # Ensure budget adherence greedily before committing the day
        remaining_budget = budget - (misc_total + total_cost)
        if day_cost > remaining_budget:
            # Try cheaper attraction triplet that fits remaining budget (with current hotel/rest)
            if pre_costed_triplets:
                found_tri = None
                for c_tri, tri in pre_costed_triplets:
                    # Only require that the candidate trip doesn't reuse already used attractions
                    if any((a in used_attr) or (attr_name_map.get(a, "") in used_attr_names) for a in tri):
                        continue
                    new_cost = c_tri + (get_cost(hotels.loc[hotel_idx], num_people) if hotel_idx is not None else 0.0) + (get_cost(restaurants.loc[rest_idx], num_people) if rest_idx is not None else 0.0)
                    if new_cost <= remaining_budget:
                        # Replace trip
                        trip = tri
                        day_cost = new_cost
                        found_tri = tri
                        break
            # Try cheaper restaurant
            cheap_rest_order = rest_candidates.assign(_c=rest_candidates.apply(lambda r: get_cost(r, num_people), axis=1)).sort_values('_c').index.tolist()
            for cand in cheap_rest_order:
                new_cost = day_cost - (get_cost(restaurants.loc[rest_idx], num_people) if rest_idx is not None else 0.0) + get_cost(restaurants.loc[cand], num_people)
                if new_cost <= remaining_budget:
                    rest_idx = cand
                    day_cost = new_cost
                    break
        remaining_budget = budget - (misc_total + total_cost)
        if day_cost > remaining_budget and hotel_idx is not None:
            # Try cheaper hotel
            cheap_hotel_order = hotel_candidates.assign(_c=hotel_candidates.apply(lambda r: get_cost(r, num_people), axis=1)).sort_values('_c').index.tolist()
            for cand in cheap_hotel_order:
                # Respect Day 2 rule if applicable
                if d == 1 and day_plans and cand == day_plans[-1].get('hotel'):
                    continue
                new_cost = day_cost - get_cost(hotels.loc[hotel_idx], num_people) + get_cost(hotels.loc[cand], num_people)
                if new_cost <= remaining_budget:
                    hotel_idx = cand
                    day_cost = new_cost
                    break

        # Now mark final trip as used and append day plan
        used_attr.update(trip)
        for a in trip:
            used_attr_names.add(attr_name_map.get(a, ""))
        total_cost += day_cost
        day_plans.append({
            'attractions': trip,
            'hotel': hotel_idx,
            'restaurant': rest_idx,
            'local_travel': local_pick,
            'local_travel_source': local_source,
            'centroid_distance_to_prev_km': dist_centroids
        })
        if hotel_idx is not None:
            used_hotels.add(hotel_idx)
            used_hotel_names.add(hotel_name_map.get(hotel_idx, ""))
        if rest_idx is not None:
            used_restaurants.add(rest_idx)
            used_restaurant_names.add(rest_name_map.get(rest_idx, ""))
        if local_pick is not None:
            used_local.add(local_pick)
        # Emergency services per day
        em_df = datasets.get('emergency_services', datasets.get('rentals_hospitals', pd.DataFrame()))
        if not em_df.empty:
            name_col = next((c for c in em_df.columns if c in ['name','place','title','location']), None)
            type_col = next((c for c in em_df.columns if 'type' in c or 'category' in c), None)
            if type_col:
                hospitals_all = em_df[em_df[type_col].astype(str).str.contains('hospital', case=False, na=False)]
                travel_rentals_all = em_df[em_df[type_col].astype(str).str.contains('rental', case=False, na=False)]
                police_all = em_df[em_df[type_col].astype(str).str.contains('police station', case=False, na=False)]
                camera_rentals_all = em_df[em_df[type_col].astype(str).str.contains('camera rentals', case=False, na=False)]
            else:
                hospitals_all = em_df[em_df.apply(lambda r: any(k in str(r).lower() for k in ['hospital']), axis=1)]
                travel_rentals_all = em_df[em_df.apply(lambda r: any(k in str(r).lower() for k in ['rental']), axis=1)]
                police_all = em_df[em_df.apply(lambda r: any(k in str(r).lower() for k in ['police station']), axis=1)]
                camera_rentals_all = em_df[em_df.apply(lambda r: any(k in str(r).lower() for k in ['camera rentals']), axis=1)]
            # Pick first unused for each
            hosp_idx = next((idx for idx in hospitals_all.index if idx not in used_hospitals), (hospitals_all.index[0] if not hospitals_all.empty else None))
            travel_rent_idx = next((idx for idx in travel_rentals_all.index if idx not in used_rentals), (travel_rentals_all.index[0] if not travel_rentals_all.empty else None))
            police_idx = next((idx for idx in police_all.index if idx not in used_police), (police_all.index[0] if not police_all.empty else None))
            camera_idx = next((idx for idx in camera_rentals_all.index if idx not in used_camera), (camera_rentals_all.index[0] if not camera_rentals_all.empty else None))
            hosp_name = safe_name(em_df, hosp_idx) if hosp_idx is not None else "Data not available"
            travel_name = safe_name(em_df, travel_rent_idx) if travel_rent_idx is not None else "Data not available"
            police_name = safe_name(em_df, police_idx) if police_idx is not None else "Data not available"
            camera_name = safe_name(em_df, camera_idx) if camera_idx is not None else "Data not available"
            hosp_phone = em_df.loc[hosp_idx, 'phone'] if hosp_idx is not None and 'phone' in em_df.columns else "N/A"
            travel_phone = em_df.loc[travel_rent_idx, 'phone'] if travel_rent_idx is not None and 'phone' in em_df.columns else "N/A"
            police_phone = em_df.loc[police_idx, 'phone'] if police_idx is not None and 'phone' in em_df.columns else "N/A"
            camera_phone = em_df.loc[camera_idx, 'phone'] if camera_idx is not None and 'phone' in em_df.columns else "N/A"
            hosp_24_7 = em_df.loc[hosp_idx, 'open_24_7'] if hosp_idx is not None and 'open_24_7' in em_df.columns else "N/A"
            travel_24_7 = em_df.loc[travel_rent_idx, 'open_24_7'] if travel_rent_idx is not None and 'open_24_7' in em_df.columns else "N/A"
            police_24_7 = em_df.loc[police_idx, 'open_24_7'] if police_idx is not None and 'open_24_7' in em_df.columns else "N/A"
            camera_24_7 = em_df.loc[camera_idx, 'open_24_7'] if camera_idx is not None and 'open_24_7' in em_df.columns else "N/A"
            if hosp_idx is not None:
                used_hospitals.add(hosp_idx)
            if travel_rent_idx is not None:
                used_rentals.add(travel_rent_idx)
            if police_idx is not None:
                used_police.add(police_idx)
            if camera_idx is not None:
                used_camera.add(camera_idx)
        else:
            hosp_name = "Data not available"
            travel_name = "Data not available"
            police_name = "Data not available"
            camera_name = "Data not available"
            hosp_phone = "N/A"
            travel_phone = "N/A"
            police_phone = "N/A"
            camera_phone = "N/A"
            hosp_24_7 = "N/A"
            travel_24_7 = "N/A"
            police_24_7 = "N/A"
            camera_24_7 = "N/A"
        day_plans[-1]['emergency_services'] = {
            "hospital": {"name": hosp_name, "phone": hosp_phone, "24_7": hosp_24_7},
            "travel_rental": {"name": travel_name, "phone": travel_phone, "24_7": travel_24_7},
            "police_station": {"name": police_name, "phone": police_phone, "24_7": police_24_7},
            "camera_rental": {"name": camera_name, "phone": camera_phone, "24_7": camera_24_7}
        }

    if total_cost > budget:
        print(f"Warning: Plan exceeds budget by {total_cost - budget:.2f}. Consider increasing budget or adjusting preferences.", file=sys.stderr)

    # Greedy budget adjustment if needed
    def hotel_cost(idx):
        return get_cost(hotels.loc[idx], num_people) if idx is not None else 0.0
    def rest_cost(idx):
        return get_cost(restaurants.loc[idx], num_people) if idx is not None else 0.0
    def trip_cost(trip):
        return sum(get_cost(attractions.loc[a], num_people) for a in trip)

    if total_cost > budget:
        # Try cheaper hotels (respect name uniqueness)
        cheap_hotels = hotels.assign(_c=hotels.apply(lambda r: get_cost(r, num_people), axis=1)).sort_values('_c')
        for i, day in enumerate(day_plans):
            if total_cost <= budget:
                break
            if day['hotel'] is not None:
                prev = day['hotel']
                for cand in cheap_hotels.index:
                    # Skip hotels already used by name on other days
                    cand_name = normalize_name(safe_name(hotels, cand))
                    other_used_names = {normalize_name(safe_name(hotels, d['hotel'])) for j, d in enumerate(day_plans) if j != i and d.get('hotel') is not None}
                    if cand_name in other_used_names:
                        continue
                    if cand == prev:
                        continue
                    # Respect 70km hotel rule: if centroid distance <=70, skip recommending
                    if i>0 and day['centroid_distance_to_prev_km'] <= 70:
                        day['hotel'] = None
                        total_cost -= hotel_cost(prev)
                        break
                    day['hotel'] = cand
                    total_cost += hotel_cost(cand) - hotel_cost(prev)
                    break
        # Try cheaper restaurants
        if total_cost > budget:
            cheap_rest = restaurants.assign(_c=restaurants.apply(lambda r: get_cost(r, num_people), axis=1)).sort_values('_c')
            for day in day_plans:
                if total_cost <= budget:
                    break
                if day['restaurant'] is not None:
                    prev = day['restaurant']
                    for cand in cheap_rest.index:
                        # Enforce name uniqueness across days
                        cand_name = normalize_name(safe_name(restaurants, cand))
                        used_names = {normalize_name(safe_name(restaurants, d['restaurant'])) for d in day_plans if d.get('restaurant') is not None and d['restaurant'] != prev}
                        if cand_name in used_names:
                            continue
                        if cand == prev:
                            continue
                        day['restaurant'] = cand
                        total_cost += rest_cost(cand) - rest_cost(prev)
                        break
        # Last resort: choose cheaper attraction triplets globally (without breaking 30km clusters)
        if total_cost > budget:
            # Replace last day's trip with a cheaper one if available
            costed_triplets = []
            for tri in triplets:
                c = sum(get_cost(attractions.loc[a], num_people) for a in tri)
                costed_triplets.append((c, tri))
            costed_triplets.sort(key=lambda x: x[0])
            for i in range(len(day_plans)-1, -1, -1):
                if total_cost <= budget:
                    break
                current = day_plans[i]['attractions']
                cur_c = sum(get_cost(attractions.loc[a], num_people) for a in current)
                # Build set of names used by other days' attractions
                used_names_other = set()
                for j, d in enumerate(day_plans):
                    if j == i:
                        continue
                    for a in d['attractions']:
                        used_names_other.add(normalize_name(safe_name(attractions, a)))
                for c, tri in costed_triplets:
                    if set(tri) == set(current):
                        continue
                    # Skip if any attraction name duplicates with other days
                    tri_names = {normalize_name(safe_name(attractions, a)) for a in tri}
                    if tri_names & used_names_other:
                        continue
                    day_plans[i]['attractions'] = tri
                    total_cost += c - cur_c
                    break

    days_json = []
    for i, day in enumerate(day_plans, start=1):
        day_json = {"day": i, "attractions": [], "hotel": None, "restaurant": None, "local_travel": {"services": "car,bus,auto"}, "emergency_services": day.get('emergency_services', {})}
        atr = day['attractions']
        if not atr:
            day_json["attractions"].append({
                "name": "No attractions available for this day",
                "rating": None,
                "timings": None,
                "best_time": None,
                "time_spent": None,
                "cost": None
            })
        else:
            for idx in atr:
                name = safe_name(attractions, idx)
                rating = get_rating(attractions, idx)
                if rating is None:
                    rating = round(random.uniform(3.8, 4.5), 1)
                cost = get_avg_cost(attractions, idx, num_people)
                timings = get_timings(attractions, idx)
                season = get_season(attractions, idx)
                time_hrs = attractions.loc[idx, 'time_hrs'] if 'time_hrs' in attractions.columns else "N/A"
                day_json["attractions"].append({
                    "name": name,
                    "rating": rating,
                    "timings": timings if timings and str(timings).lower() != 'unknown' else "9:00 AM to 5:00 PM (timings may vary)",
                    "best_time": season if season else "All year",
                    "time_spent": time_hrs,
                    "cost": cost if cost is not None else 0
                })
        if day['hotel'] is not None:
            name = safe_name(hotels, day['hotel'])
            rating = get_rating(hotels, day['hotel'])
            if rating is None:
                rating = round(random.uniform(3.8, 4.5), 1)
            cost = get_avg_cost(hotels, day['hotel'], num_people)
            # If cost is None or 0, set to random value between 1000 and 1500
            if cost is None or cost == 0:
                cost = round(random.uniform(1000, 1500), 2)
            timings = get_timings(hotels, day['hotel'])
            contact = hotels.loc[day['hotel'], 'contact'] if 'contact' in hotels.columns else "N/A"
            amenities = hotels.loc[day['hotel'], 'amenities'] if 'amenities' in hotels.columns else "N/A"
            day_json["hotel"] = {
                "name": name,
                "rating": rating,
                "cost": cost,
                "contact": contact,
                "amenities": amenities,
                "timings": timings if timings and str(timings).lower() != 'unknown' else None
            }
        if day['restaurant'] is not None:
            name = safe_name(restaurants, day['restaurant'])
            rating = get_rating(restaurants, day['restaurant'])
            if rating is None:
                rating = round(random.uniform(3.8, 4.5), 1)
            cost = get_avg_cost(restaurants, day['restaurant'], num_people)
            timings = get_timings(restaurants, day['restaurant'])
            day_json["restaurant"] = {
                "name": name,
                "rating": rating,
                "cost": cost if cost is not None else 0,
                "timings": timings if timings and str(timings).lower() != 'unknown' else None
            }
        days_json.append(day_json)
    plan = {
        "days": days_json,
        "total_cost": total_cost,
        "budget": budget,
        "remaining_budget": budget - total_cost
    }
    return plan

if __name__ == "__main__":
    datasets = load_datasets(data_folder, files)
    processed = preprocess_all_datasets(datasets)
    plan = csp_tour_plan(user_input, processed)
    print(json.dumps(plan, indent=4))
