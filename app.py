from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained pipeline
model = joblib.load("model.pkl")

# Mapping label -> sentiment
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    # Make prediction using pipeline
    prediction_label = model.predict([review])[0]
    prediction = label_map[prediction_label]

    print("Prediction result:", prediction)  # For debugging

    return render_template("index.html", review=review, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

