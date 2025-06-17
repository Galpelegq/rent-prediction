import pandas as pd
import numpy as np
from scipy.stats import zscore

def prepare_data(df, dataset_type="train"):
    df = df.copy()

    # Remove irrelevant property types
    excluded_types = ['מחסן', 'חניה', 'כללי']
    df = df[~df['property_type'].isin(excluded_types)]

    # Handle price-related filters only on training data
    if dataset_type == "train":
        if 'price' in df.columns:
            df = df[df['price'].notna()]
            df = df[(df['price'] >= 1500) & (df['price'] <= 9000)]
            df = df[(np.abs(zscore(df['price'])) < 3)]
            price_col = df.pop('price')  # Move price to the end
            df['price'] = price_col

    # Drop rows with missing values in key columns
    df = df.dropna(subset=['property_type', 'neighborhood', 'floor', 'num_of_images', 'distance_from_center'])

    # Convert distance to meters if needed
    df['distance_from_center'] = df['distance_from_center'].apply(lambda x: x * 1000 if x < 100 else x)

    # Filter by reasonable ranges
    df = df[(df['area'] >= 20) & (df['area'] <= 200)]
    df = df[df['distance_from_center'] <= 10000]
    df = df[(df['room_num'] >= 1) & (df['room_num'] <= 7)]

    # Fill missing values
    df['garden_area'] = df['garden_area'].fillna(0)
    df['handicap'] = df['handicap'].fillna(0)
    if 'num_of_payments' in df.columns and not df['num_of_payments'].mode().empty:
        df['num_of_payments'] = df['num_of_payments'].fillna(df['num_of_payments'].mode()[0])
    if 'days_to_enter' in df.columns and not df['days_to_enter'].mode().empty:
        df['days_to_enter'] = df['days_to_enter'].fillna(df['days_to_enter'].mode()[0])

    # Drop non-informative column
    df = df.drop(columns=['address'], errors='ignore')

    # Map neighborhoods to broader zones
    zone_map = {
        'לב תל אביב החלק הצפוני': 'מרכז',
        'הצפון הישן החלק הצפוני': 'צפון',
        'הצפון הישן החלק המרכזי': 'צפון',
        'הצפון החדש החלק הצפוני': 'צפון',
        'הצפון החדש החלק הדרומי': 'צפון',
        'הגוש הגדול': 'צפון',
        'תל ברוך צפון': 'צפון',
        'נחלת יצחק': 'מזרח',
        'המרכז הישן': 'מרכז',
        'צפון יפו': 'יפו',
        'לב יפו': 'יפו',
        'יפו ד': 'יפו',
        'שפירא': 'דרום',
        'פלורנטין': 'דרום',
        'נווה שאנן': 'דרום',
        'קריית שלום': 'דרום'
    }
    df['location_zone'] = df['neighborhood'].map(zone_map).fillna('אחר')

    # Feature engineering and type conversions
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    df['total_floors'] = pd.to_numeric(df['total_floors'], errors='coerce')
    df['floor_ratio'] = df['floor'] / df['total_floors'].replace(0, np.nan)

    df['room_x_center'] = df['room_num'] * (df['distance_from_center'] < 2000).astype(int)
    df['area_x_central'] = df['area'] * (df['distance_from_center'] < 2000).astype(int)
    df['is_luxury_apartment'] = ((df['has_balcony'] == 1) & (df['elevator'] == 1) & (df['is_renovated'] == 1)).astype(int)
    df['luxury_score'] = df[['has_balcony', 'is_renovated', 'ac', 'elevator']].sum(axis=1)
    df['luxury_score_cat'] = pd.cut(df['luxury_score'], bins=[-1, 0, 2, 4], labels=['Low', 'Medium', 'High'])

    # Extract features from description text
    df['has_view'] = df['description'].str.contains("נוף|צופה ל", na=False).astype(int)
    df['near_beach'] = df['description'].str.contains("ים|חוף", na=False).astype(int)
    df['quiet_area'] = df['description'].str.contains("שקטה|רחוב שקט", na=False).astype(int)

    # Additional interactions and ratios
    df['area_luxury_interaction'] = df['area'] * df['luxury_score']
    df['room_area_interaction'] = df['room_num'] * df['area']
    df['log_area'] = np.log1p(df['area'])
    df['log_room_area_interaction'] = np.log1p(df['room_area_interaction'])
    df['room_density'] = df['area'] / df['room_num']
    df['log_room_density'] = np.log1p(df['room_density'])

    df['floor_bonus'] = ((df['floor'] > 4) & (df['elevator'] == 1) & (df['has_balcony'] == 1)).astype(int)
    df['many_images'] = (df['num_of_images'] >= 6).astype(int)
    df['central_luxury'] = df['is_luxury_apartment'] * (df['distance_from_center'] < 2000).astype(int)
    df['floor_area_interaction'] = df['floor'] * df['area']
    df['zone_luxury_interaction'] = df['luxury_score'] * (df['location_zone'] == 'מרכז').astype(int)
    df['room_area_ratio'] = df['room_num'] / (df['area'] + 1)
    df['room_to_total_floor_ratio'] = df['room_num'] / (df['total_floors'] + 1)

    # Binning numeric features
    df['distance_cat'] = pd.cut(df['distance_from_center'], bins=[0, 2000, 5000, 10000], labels=['Center', 'Mid', 'Far'])
    df['room_num_binned'] = pd.cut(df['room_num'], bins=[0, 1.5, 3.5, 7], labels=['Small', 'Medium', 'Large'])
    df['garden_area_flag'] = (df['garden_area'] > 0).astype(int)
    df['floor_level'] = pd.cut(df['floor'], bins=[-1, 0, 3, 7, 100], labels=['Basement', 'Low', 'Mid', 'High']).astype('string')

    # Rarity feature based on unique combinations
    df['unique_combo'] = df['room_num'].astype(str) + "_" + df['floor_level'].astype(str) + "_" + df['location_zone'].astype(str)
    combo_counts = df['unique_combo'].value_counts()
    df['rarity_score'] = df['unique_combo'].map(combo_counts)
    df['rarity_score'] = 1 / (df['rarity_score'] + 1)
    df['rare_luxury_combo'] = df['rarity_score'] * df['luxury_score']
    df = df.drop(columns=['unique_combo'], errors='ignore')

    # More interaction features
    df['area_per_floor'] = df['area'] / (df['total_floors'] + 1)
    df['room_num_squared'] = df['room_num'] ** 2
    df['area_room_ratio_squared'] = (df['area'] / (df['room_num'] + 1)) ** 2
    df['floor_weighted_luxury'] = df['floor'] * df['luxury_score']
    df['center_luxury_interaction'] = df['is_luxury_apartment'] * (df['location_zone'] == 'מרכז').astype(int)
    df['elevator_balcony_combo'] = ((df['elevator'] == 1) & (df['has_balcony'] == 1)).astype(int)

    # Define column types for final conversion
    str_cols = ['property_type', 'neighborhood', 'location_zone', 'luxury_score_cat', 'distance_cat', 'room_num_binned', 'floor_level']
    float_cols = ['room_num', 'distance_from_center']
    int_cols = ['floor', 'area', 'garden_area', 'num_of_payments', 'monthly_arnona', 'building_tax', 'total_floors', 'num_of_images']
    binary_cols = ['has_parking', 'has_storage', 'elevator', 'ac', 'handicap', 'has_bars', 'has_safe_room',
                   'has_balcony', 'is_furnished', 'is_renovated', 'has_view', 'near_beach', 'quiet_area',
                   'is_luxury_apartment', 'floor_bonus', 'many_images', 'central_luxury',
                   'garden_area_flag', 'center_luxury_interaction', 'elevator_balcony_combo']

    # Convert column types
    for col in str_cols:
        df[col] = df[col].astype('string')
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=str_cols, drop_first=True)

    # Final cleanup
    df = df.dropna()
    df = df.convert_dtypes()

    # Normalize types
    for col in df.columns:
        if df[col].dtype.name == 'boolean':
            df[col] = df[col].astype(int)
        elif "Int" in str(df[col].dtype):
            df[col] = df[col].astype(int)
        elif "Float" in str(df[col].dtype):
            df[col] = df[col].astype(float)

    return df
