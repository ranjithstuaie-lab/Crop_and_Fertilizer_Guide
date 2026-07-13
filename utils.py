import joblib
import numpy as np

# -----------------------
# Crop Recommendation
# -----------------------

crop_model = joblib.load("random_forest_final_model.joblib")
crop_scaler = joblib.load("standard_scaler.joblib")
crop_encoder = joblib.load("label_encoder.joblib")

# -----------------------
# Fertilizer Recommendation
# -----------------------

fertilizer_model = joblib.load("best_fertilizer_recommender_model.joblib")
fertilizer_scaler = joblib.load("scaler.joblib")

encoders = joblib.load("label_encoders.joblib")

le_crop = encoders["le_crop"]
le_fertilizer = encoders["le_fertilizer"]


def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    sample = np.array([[

        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall

    ]])

    sample = crop_scaler.transform(sample)

    prediction = crop_model.predict(sample)

    probability = crop_model.predict_proba(sample)[0]

    crop = crop_encoder.inverse_transform(prediction)[0]

    return crop, probability


def predict_fertilizer(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    crop
):

    crop_encoded = le_crop.transform([crop])[0]

    # Numeric features (7) — scale these with the fertilizer scaler
    numeric = np.array([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]])

    # scaler was fitted on numeric features only (7); transform them
    numeric_scaled = fertilizer_scaler.transform(numeric)

    # Append the encoded crop (do not scale the encoded label)
    sample = np.hstack([numeric_scaled, np.array([[crop_encoded]])])

    prediction = fertilizer_model.predict(sample)

    fertilizer = le_fertilizer.inverse_transform(prediction)[0]

    return fertilizer