import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from data import find_yield_data

NUTRIENTS = ['Protein', 'Oil', 'Carbohydrates', 'Calories', 'EFA']
CROP_TYPES = ['Barley', 'Corn', 'Potato', 'Rice', 'Soybean', 'Sugarcane', 'Sunflower', 'Wheat']

try:
    model = joblib.load('model/regressors.pkl')
except FileNotFoundError:
    raise SystemExit("Error: Model not found")

FIXED_TEMP = 20
FIXED_HUMIDITY = 70

humidity_range = np.linspace(20, 90, 50)
temperature_range = np.linspace(10, 35, 50)

data = find_yield_data()

def predict_yield_over_humidity(nutrient, crop):
    X = pd.DataFrame({
        'Humidity': humidity_range,
        'Temperature': [FIXED_TEMP] * len(humidity_range),
    })
    return model[crop].predict(X) * data[crop][nutrient]


def predict_yield_over_temperature(nutrient, crop):
    X = pd.DataFrame({
        'Humidity': [FIXED_HUMIDITY] * len(temperature_range),
        'Temperature': temperature_range,
    })
    return model[crop].predict(X) * data[crop][nutrient]


for nutrient in NUTRIENTS:
    for crop in CROP_TYPES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        yield_hum = predict_yield_over_humidity(nutrient, crop)
        axes[0].scatter(humidity_range, yield_hum, c='green', label='Yield')
        axes[0].set_title(f'{crop} - {nutrient} vs Humidity')
        axes[0].set_xlabel('Humidity (%)')
        axes[0].set_ylabel('Yield (kg/ha)')

        yield_temp = predict_yield_over_temperature(nutrient, crop)
        axes[1].scatter(temperature_range, yield_temp, c='blue', label='Yield')
        axes[1].set_title(f'{crop} - {nutrient} vs Temperature')
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].set_ylabel('Yield (kg/ha)')

        plt.tight_layout()
        plt.savefig(f'plots/{nutrient}_{crop}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
