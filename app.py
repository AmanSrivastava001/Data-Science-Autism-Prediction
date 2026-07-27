from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask import render_template

app = Flask(__name__)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ---------------------------------------------------------------------------
# These mappings reproduce the cleanup that was done in the training
# notebook BEFORE the LabelEncoders were fit. The encoders.pkl file only
# knows about the *cleaned* categories below, so any raw value coming from
# the HTML form must be normalised the same way before we call
# encoder.transform(), otherwise sklearn raises:
#   ValueError: y contains previously unseen labels: 'Parent'
# ---------------------------------------------------------------------------
ETHNICITY_MAP = {
    "?": "Others",
    "others": "Others",
}

RELATION_MAP = {
    "?": "Others",
    "Relative": "Others",
    "Parent": "Others",
    "Health care professional": "Others",
}

COUNTRY_MAP = {
    "Viet Nam": "Vietnam",
    "AmericanSamoa": "United States",
    "Hong Kong": "China",
}


def safe_transform(encoder, value):
    """Transform a single value with a fitted LabelEncoder.

    Falls back to 'Others' (or the first known class if 'Others' isn't
    available) when the value was never seen during training, so the app
    never crashes on an unexpected category.
    """
    classes = list(encoder.classes_)
    if value not in classes:
        value = "Others" if "Others" in classes else classes[0]
    return encoder.transform([value])[0]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-form", methods=["POST"])
def predict_form():

    data = {
        "A1_Score": int(request.form["A1_Score"]),
        "A2_Score": int(request.form["A2_Score"]),
        "A3_Score": int(request.form["A3_Score"]),
        "A4_Score": int(request.form["A4_Score"]),
        "A5_Score": int(request.form["A5_Score"]),
        "A6_Score": int(request.form["A6_Score"]),
        "A7_Score": int(request.form["A7_Score"]),
        "A8_Score": int(request.form["A8_Score"]),
        "A9_Score": int(request.form["A9_Score"]),
        "A10_Score": int(request.form["A10_Score"]),
        "age": int(request.form["age"]),
        "gender": request.form["gender"],
        "ethnicity": request.form["ethnicity"],
        "jaundice": request.form["jaundice"],
        "austim": request.form["austim"],
        "contry_of_res": request.form["contry_of_res"],
        "used_app_before": request.form["used_app_before"],
        "result": float(request.form["result"]),
        "relation": request.form["relation"],
    }

    # Normalise categories to match what the encoders were trained on
    data["ethnicity"] = ETHNICITY_MAP.get(data["ethnicity"], data["ethnicity"])
    data["relation"] = RELATION_MAP.get(data["relation"], data["relation"])
    data["contry_of_res"] = COUNTRY_MAP.get(data["contry_of_res"], data["contry_of_res"])

    df = pd.DataFrame([data])

    for col in encoders:
        if col in df.columns:
            df[col] = df[col].apply(lambda v, enc=encoders[col]: safe_transform(enc, v))

    prediction = model.predict(df)

    result = (
        "⚠️ Autism Detected"
        if prediction[0] == 1
        else "✅ No Autism Detected"
    )

    return render_template(
        "index.html",
        prediction=result,
    )


if __name__ == "__main__":
    app.run(debug=True)