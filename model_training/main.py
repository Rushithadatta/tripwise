import argparse
import sys
import pandas as pd
from recommendation_model import load_data, recommend_nearby, train_global_kmeans

DEFAULT_DATASETS = ['Attractions.xlsx','Hotel.xlsx','restaurants.xlsx']

def ensure_global_kmeans_auto():
    from recommendation_model import _GLOBAL_KMEANS_MODEL as GK
    if GK is not None:
        return  # already trained
    a,h,r = load_data(*DEFAULT_DATASETS)
    all_df = pd.concat([a,h,r], ignore_index=True)
    train_global_kmeans(all_df, k=8, auto=True, max_k=15, model_path='kmeans_global.pkl', retrain=False, explain=False, metrics=False)

def format_three_blocks(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure exactly 10 per category (truncate or pad if necessary)
    blocks = []
    for cat in ['attraction','hotel','restaurant']:
        sub = df[df['category']==cat].head(10)
        blocks.append(sub)
    merged = pd.concat(blocks, ignore_index=True)
    return merged

DEFAULT_RADIUS = 5.0
DEFAULT_WEIGHTS = (0.4, 0.35, 0.25)  # distance, rating, price

def main():
    import json
    input_data = None
    # Try to read JSON from stdin (for backend calls)
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw:
                input_data = json.loads(raw)
    except Exception:
        input_data = None

    if input_data and input_data.get('recommend'):
        place_name = input_data.get('place', None)
    else:
        # Fallback to CLI
        parser = argparse.ArgumentParser(description='Tourism recommender: provide only the anchor place name.')
        parser.add_argument('name', nargs='?', help='Anchor place name (e.g., "Ramakrishna Beach")')
        args = parser.parse_args()
        place_name = args.name

    if not place_name:
        print('Usage: python main.py "<Anchor Place Name>"')
        print('Example: python main.py "Ramakrishna Beach"')
        sys.exit(1)

    import recommendation_model as rm
    rm._ML_MODE = 'kmeans'
    rm._ENABLE_ML = True
    rm._ML_WEIGHT = 0.5  # enforce 50/50 blend
    rm._ML_MODEL_USED = None

    ensure_global_kmeans_auto()

    # We ask for 30 total (10 per category) using min_per_category=10
    dw, rw, pw = DEFAULT_WEIGHTS
    recs = recommend_nearby(
        place_name=place_name,
        place_type='attraction',
        radius_km=DEFAULT_RADIUS,
        top_n=30,
        distance_weight=dw,
        rating_weight=rw,
        price_weight=pw,
        include_categories='attraction,hotel,restaurant',
        min_per_category=10,
        attractions_path=DEFAULT_DATASETS[0],
        hotels_path=DEFAULT_DATASETS[1],
        restaurants_path=DEFAULT_DATASETS[2]
    )

    # Enforce exactly 10 per category in display
    display_df = format_three_blocks(recs)

    # Group recommendations by category
    places = display_df[display_df['category'] == 'attraction']['name'].tolist()[:5]
    hotels = display_df[display_df['category'] == 'hotel']['name'].tolist()[:5]
    food = display_df[display_df['category'] == 'restaurant']['name'].tolist()[:5]

    import json as _json
    print(_json.dumps({
        "places": places,
        "hotels": hotels,
        "food": food
    }))

if __name__ == '__main__':
    main()
