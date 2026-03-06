from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model, scaler, encoder
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    brand_encoded = encoder.transform(df[["brand"]])
    brand_df = pd.DataFrame(
        brand_encoded,
        columns=encoder.get_feature_names_out(["brand"])
    )

    df = df.drop("brand", axis=1)
    df = pd.concat([df, brand_df], axis=1)

    df_scaled = scaler.transform(df)

    price = model.predict(df_scaled)[0]

    return jsonify({"price": round(float(price), 2)})

if __name__ == "__main__":
    app.run(debug=True)
