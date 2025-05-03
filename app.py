from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load trained model (assumes it includes preprocessing)
with open('file.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input
        input_data = {
            'SALE PRICE': float(request.form['sale_price']),
            'LAND SQUARE FEET': float(request.form['land_sqft']),
            'GROSS SQUARE FEET': float(request.form['gross_sqft']),
            'YEAR BUILT': int(request.form['year_built']),
            'BUILDING CLASS CATEGORY': request.form['building_class'],
            'BOROUGH': request.form['borough'],
            'NEIGHBORHOOD': request.form['neighborhood'],
            'SALE YEAR': int(request.form['sale_year']),
            'SALE MONTH': int(request.form['sale_month'])
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(df)[0]

        return render_template('index.html', prediction=f"Predicted Tax Class: {prediction}")
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
