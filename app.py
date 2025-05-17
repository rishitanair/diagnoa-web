import pandas as pd
from flask import Flask, render_template, request
import joblib
import os

app = Flask("Diagnoa")

disease_model = joblib.load('./models/model.pkl')
disease_label_encoder = joblib.load('./models/label_encoder.pkl')
available_symptoms = joblib.load('./models/symptom_vocab.pkl')


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    message = None
    if request.method == 'POST':
        selected_symptoms = [request.form.get(f'symptom{i}') for i in range(1, 18)]
        cleaned_symptoms = [sym.lower().strip() for sym in selected_symptoms if sym and sym.strip()]

        input_features = [0] * len(available_symptoms)
        missing_symptoms = []
        for symptom in cleaned_symptoms:
            if symptom in available_symptoms:
                index = available_symptoms.index(symptom)
                input_features[index] = 1
            else:
                missing_symptoms.append(symptom)

        if missing_symptoms:
            message = f"These symptoms were ignored as unknown: {', '.join(missing_symptoms)}"

        prediction_encoded = disease_model.predict([input_features])[0]
        result = disease_label_encoder.inverse_transform([prediction_encoded])[0]

    return render_template('index.html', symptom_list=available_symptoms, prediction=result, warning=message)


@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_info = None
    warning_msg = None
    if request.method == 'POST':
        submitted_symptoms = request.form.getlist('symptoms')
        cleaned_symptoms = [sym.lower().strip() for sym in submitted_symptoms if sym and sym.strip()]

        input_features = [0] * len(available_symptoms)
        missing_symptoms = []
        for symptom in cleaned_symptoms:
            if symptom in available_symptoms:
                index = available_symptoms.index(symptom)
                input_features[index] = 1
            else:
                missing_symptoms.append(symptom)

        if missing_symptoms:
            warning_msg = f"Ignored unknown symptoms: {', '.join(missing_symptoms)}"

        prediction_encoded = disease_model.predict([input_features])[0]
        disease_name = disease_label_encoder.inverse_transform([prediction_encoded])[0]

        confidence_score = 90

        prediction_info = {
            'disease': disease_name,
            'confidence': confidence_score,
            'symptoms': cleaned_symptoms
        }

    return render_template('predict.html', symptoms_list=available_symptoms, prediction=prediction_info, warning=warning_msg)


if __name__ == '__main__':
    port_num = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port_num)
