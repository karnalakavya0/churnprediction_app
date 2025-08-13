from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # if you saved it

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    return jsonify({"churn_prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
