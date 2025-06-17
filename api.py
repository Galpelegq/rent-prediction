from flask import Flask, request, render_template
import pandas as pd
import joblib
from assets_data_prep import prepare_data

app = Flask(__name__)
model = joblib.load("trained_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form

    input_data = {
        'property_type': form_data['property_type'],
        'neighborhood': form_data['neighborhood'],
        'room_num': float(form_data['room_num']),
        'floor': float(form_data['floor']),
        'total_floors': float(form_data['total_floors']),
        'area': float(form_data['area']),
        'garden_area': float(form_data['garden_area']),
        'num_of_images': int(form_data['num_of_images']),
        'distance_from_center': float(form_data['distance_from_center']),
        'monthly_arnona': float(form_data['monthly_arnona']),
        'building_tax': float(form_data['building_tax']),
        'num_of_payments': int(form_data['num_of_payments']),
        'description': form_data['description'],
        'has_balcony': int('has_balcony' in form_data),
        'elevator': int('elevator' in form_data),
        'has_safe_room': int('has_safe_room' in form_data),
        'has_parking': int('has_parking' in form_data),
        'has_storage': int('has_storage' in form_data),
        'has_bars': int('has_bars' in form_data),
        'ac': int('ac' in form_data),
        'is_renovated': int('is_renovated' in form_data),
        'is_furnished': int('is_furnished' in form_data),
        'handicap': int('handicap' in form_data)
    }

    df_user = pd.DataFrame([input_data])
    df_prep = prepare_data(df_user, dataset_type="test")

    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in df_prep.columns:
            df_prep[col] = 0
    df_prep = df_prep[expected_cols]

    prediction = model.predict(df_prep)[0]
    return render_template('index.html', prediction_text=f"הערכת שכר דירה: ₪{prediction:,.0f}")

@app.route('/clear', methods=['GET'])
def clear_form():
    return render_template('index.html')  # טופס ריק

if __name__ == '__main__':
    app.run(debug=True)
