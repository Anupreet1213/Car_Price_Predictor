from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)

cors=CORS(app)
model = pickle.load(open("car_prediction\LinearRegressionModel.pkl", 'rb'))
car=pd.read_csv('car_prediction\Cleaned_Car.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name','company','year','kms_driven','fuel_type']))
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)