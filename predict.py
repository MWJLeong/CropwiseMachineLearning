import joblib
import pandas as pd
from data import find_yield_data

CROP_TYPES = ['Barley', 'Corn', 'Potato', 'Rice', 'Soybean', 'Sugarcane', 'Sunflower', 'Wheat']

models = joblib.load('model/regressors.pkl')

data = find_yield_data()

def predict(humidity, temperature, nutrient):
    crop_data = []

    for crop in CROP_TYPES:
        input_data = pd.DataFrame([{
            "Humidity": humidity,
            "Temperature": temperature,
        }])

        prediction = models[crop].predict(input_data)[0].item() * data[crop][nutrient]

        crop_data.append((crop, prediction))

    return sorted(crop_data, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    while True:
        humidity = float(input('Humidity: '))
        temperature = float(input('Temperature: '))
        nutrient = input('Nutrient: ')
        crop_data = predict(humidity, temperature, nutrient)
        print(crop_data)
