# File: integrated_async_offline.py
# Purpose: Offline system with Keras MobileNetV3 fallback + Price prediction

import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import datetime, base64, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
import lightgbm as lgb

from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

warnings.filterwarnings("ignore")

# --- CONFIG ---
MODEL_FILE = "scalable_price_predictor_pipeline_v3.pkl"

# --- Price Prediction Model ---
GRADE_TO_SCORE_MAP = {'A': 9.5, 'B': 7.5, 'C': 5.5}
PRODUCT_NAME_TO_CODE_MAP = {
    "Tomato (Local)": "Product_01", "Basmati Rice": "Product_02", "Chili (Dried)": "Product_03",
    "Onion (Nasik)": "Product_04", "Cotton (Long Staple)": "Product_05", "Ginger (Fresh)": "Product_06",
    "Moong Dal": "Product_07", "Potatoes (Red)": "Product_08", "Mustard Oil Seed": "Product_09",
    "Millet (Jowar)": "Product_10"
}

def generate_region_map(n=500):
    states = ["Maharashtra", "Odisha", "Gujarat", "Karnataka", "Punjab", "Rajasthan", "Telangana", "Delhi-NCR"]
    region_map = {"Cuttack Market": "Region-385_Odisha"}
    for i in range(n):
        region_map[f"Market_{np.random.randint(10,999)}-{np.random.choice(states).split('-')[0][:3]}"] = f"Region-{i:03d}_{np.random.choice(states)}"
    return region_map

REGION_NAME_TO_CODE_MAP = generate_region_map()

def create_mock_data(n_samples=100000):
    product_codes = list(PRODUCT_NAME_TO_CODE_MAP.values())
    region_codes = list(REGION_NAME_TO_CODE_MAP.values())
    mock_grades = np.random.choice(list(GRADE_TO_SCORE_MAP.keys()), n_samples, p=[0.4,0.4,0.2])
    quality_scores = np.clip([GRADE_TO_SCORE_MAP[g]+np.random.normal(0,0.5) for g in mock_grades],5,10)
    df = pd.DataFrame({
        'Product_Name': np.random.choice(product_codes, n_samples),
        'Harvest_Month': np.random.randint(1,13,n_samples),
        'Region_Code': np.random.choice(region_codes, n_samples),
        'Quality_Score': quality_scores,
        'Pest_Damage_Ratio': np.clip(np.random.lognormal(-2.5,1,n_samples),0,0.1),
        'Market_Demand_Index': np.clip(np.random.normal(1.2,0.4,n_samples),0.5,2.5),
        'Storage_Cost_Index': np.clip(np.random.normal(0.2,0.1,n_samples),0.05,0.5),
        'Current_Temperature_C': np.clip(np.random.normal(30,8,n_samples),10,45),
        'Last_7_Days_Rainfall_MM': np.clip(np.random.lognormal(2.5,1,n_samples),0,300),
    })
    price_noise = np.random.normal(0,5,n_samples)
    price_by_product = df['Product_Name'].apply(lambda x:{'Product_01':20,'Product_02':60,'Product_09':70}.get(x,45))
    df['Base_Price_INR'] = np.clip((price_by_product + df['Quality_Score']*3 - df['Pest_Damage_Ratio']*50 + df['Market_Demand_Index']*10 + df['Storage_Cost_Index']*20 - df['Current_Temperature_C']*0.1 - df['Last_7_Days_Rainfall_MM']*0.05 + price_noise).round(2),10,150)
    return df

def train_model():
    print("Training Price Prediction Model...")
    df = create_mock_data()
    X = df.drop('Base_Price_INR',axis=1)
    y = df['Base_Price_INR']
    cat_features = ['Product_Name','Harvest_Month','Region_Code']
    num_features = ['Quality_Score','Pest_Damage_Ratio','Market_Demand_Index','Storage_Cost_Index','Current_Temperature_C','Last_7_Days_Rainfall_MM']
    pre = ColumnTransformer(transformers=[('cat',TargetEncoder(min_samples_leaf=20,smoothing=10),cat_features),
                                          ('num',StandardScaler(),num_features)], remainder='passthrough')
    pipeline = Pipeline([('pre',pre), ('reg',lgb.LGBMRegressor(random_state=42,n_estimators=500,learning_rate=0.05,n_jobs=-1))])
    pipeline.fit(X,y)
    print("Model Ready.")
    return pipeline

model_pipeline = train_model()

# --- Offline Keras MobileNetV3 Fallback ---
fallback_model = MobileNetV3Small(weights='imagenet')

def call_fallback_model_from_base64(base64_str):
    img_data = base64.b64decode(base64_str)
    with open("temp_img.jpg","wb") as f:
        f.write(img_data)
    img = image.load_img("temp_img.jpg", target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = fallback_model.predict(x)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1].lower()
    if label in ['tomato','onion','chili','ginger','potato']:
        grade = 'A'
    else:
        grade = 'B'
    return {"quality_grade": grade, "justification": f"Predicted by Keras MobileNetV3: {decoded[1]}"}

# --- Flask API ---
app = Flask(__name__)

REQUIRED_FIELDS = ['Product_Name','Region_Name','Harvest_Month',
                   'Market_Demand_Index','Storage_Cost_Index','Pest_Damage_Ratio',
                   'Current_Temperature_C','Last_7_Days_Rainfall_MM']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    missing=[f for f in REQUIRED_FIELDS if f not in data]
    if missing: 
        return jsonify({"error":"Missing features","missing":missing}),400

    try:
        # Use Base64 string if provided, else read from file
        if 'encoded_image_base64' in data:
            base64_img = data['encoded_image_base64']
        elif 'encoded_image_file' in data:
            with open(data['encoded_image_file'], 'r') as f:
                base64_img = f.read().strip()
        else:
            base64_img = None

        if base64_img:
            grade_output = call_fallback_model_from_base64(base64_img)
            grade = grade_output.get('quality_grade','B')
            justification = grade_output.get('justification','No justification')
        else:
            grade = 'B'
            justification = 'No image provided'

        quality_score = GRADE_TO_SCORE_MAP.get(grade,7.5)
        product_code = PRODUCT_NAME_TO_CODE_MAP.get(data['Product_Name'])
        region_code = REGION_NAME_TO_CODE_MAP.get(data['Region_Name'])
        if not product_code or not region_code:
            return jsonify({"error":"Product or Region not found"}),400

        model_input = pd.DataFrame([{
            'Product_Name':product_code,
            'Harvest_Month':int(data['Harvest_Month']),
            'Region_Code':region_code,
            'Quality_Score':quality_score,
            'Pest_Damage_Ratio':float(data['Pest_Damage_Ratio']),
            'Market_Demand_Index':float(data['Market_Demand_Index']),
            'Storage_Cost_Index':float(data['Storage_Cost_Index']),
            'Current_Temperature_C':float(data['Current_Temperature_C']),
            'Last_7_Days_Rainfall_MM':float(data['Last_7_Days_Rainfall_MM'])
        }])
        predicted_price = round(float(model_pipeline.predict(model_input)[0]),2)

        return jsonify({
            "status":"success",
            "timestamp":datetime.datetime.now().isoformat(),
            "product_name":data['Product_Name'],
            "region_name":data['Region_Name'],
            "quality_grade":grade,
            "justification":justification,
            "predicted_base_price_inr_per_kg":predicted_price
        }),200

    except Exception as e:
        return jsonify({"error":"Prediction failed","details":str(e)}),500

# --- ENTRY POINT FOR RENDER/Heroku ---
if __name__=="__main__":
    print("--- Starting Integrated System with Offline Keras MobileNetV3 ---")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
