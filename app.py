from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained models
rf_clf = joblib.load('models/random_forest_classifier.joblib')
rf_reg = joblib.load('models/random_forest_regressor.joblib')

# Load the dataset
file_path = 'data/combined.csv'
df = pd.read_csv(file_path)

# Extract county data with latitude and longitude
county_data = df.drop_duplicates(subset='county').sort_values(by="county").reset_index(drop=True)
county_data_dict = county_data.set_index('county').T.to_dict()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/viz1')
def viz1():
    return render_template('viz1.html')

@app.route('/viz2')
def viz2():
    return render_template('viz2.html')

@app.route('/viz3')
def viz3():
    return render_template('viz3.html')

@app.route('/heatmap_data')
def get_heatmap_data():
    # Serve the heatmap_data.json file
    return app.send_static_file('heatmap_data.json')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    county = int(request.form['county'])
    temp = float(request.form['temp'])
    humid = float(request.form['humid'])
    precip = float(request.form['precip'])

    # Create a DataFrame for the new input
    X_new = pd.DataFrame({
        'county': [county],
        'q_avgtempF': [temp],
        'q_avghumid': [humid],
        'q_sumprecip': [precip]
    })

    # Predict wildfire occurrence
    wildfire_occurrence = rf_clf.predict(X_new)

    # Predict acres burned if wildfire occurs
    if wildfire_occurrence[0] == 1:
        acres_burned = rf_reg.predict(X_new)[0]
    else:
        acres_burned = 0

    # Get county name and coordinates
    county_name = county_data['county'].unique()[county]
    lat = county_data_dict[county_name]['lat']
    lon = county_data_dict[county_name]['long']

    result = {
        'wildfire_occurrence': int(wildfire_occurrence[0]),
        'acres_burned': acres_burned,
        'county': county_name,
        'lat': lat,
        'lon': lon
    }

    response = jsonify(result)
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response

if __name__ == "__main__":
    app.run(debug=True)
