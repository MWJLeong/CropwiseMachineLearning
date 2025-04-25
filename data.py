import pandas as pd
from bs4 import BeautifulSoup
import requests

DATA_PATH = 'data/crop_yield_dataset.csv'
YIELD_URL = 'https://www.gardeningplaces.com/articles/nutrition-per-hectare1.htm'

def import_data():
    df = pd.read_csv(DATA_PATH)
    return df


def find_yield_data():
    response = requests.get(YIELD_URL)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table')
    table = tables[2]
    rows = table.find_all('tr')[1:]

    crop_name_mapping = {
        'maize (shelled corn)': 'Corn',
        'rice - paddy': 'Rice',
        'soybeans': 'Soybean',
        'sugarcane': 'Sugarcane',
        'sunflower seeds': 'Sunflower',
        'wheat': 'Wheat',
        'barley': 'Barley',
        'potato': 'Potato'
    }

    data = {}

    for row in rows:
        try:
            cols = row.find_all('td')
            crop_name = cols[0].text.strip()

            if crop_name in crop_name_mapping:
                crop_name = crop_name_mapping[crop_name]

            produce_kg_per_ha = float(cols[1].text.strip().replace(',', ''))
            protein_pc = float(cols[2].text.strip().replace('%', '')) / 100
            oil_pc = float(cols[4].text.strip().replace('%', '')) / 100
            efa_pc = float(cols[6].text.strip().replace('%', '')) / 100
            carb_pc = float(cols[8].text.strip().replace('%', '')) / 100
            kcal_per_sqm = float(cols[10].text.strip().replace(',', ''))
            kcal_per_tonne = kcal_per_sqm * 10000 / (produce_kg_per_ha / 1000)

            if crop_name in ['Barley', 'Corn', 'Potato', 'Rice', 'Soybean', 'Sugarcane', 'Sunflower', 'Wheat']:
                data[crop_name] = {
                    'Protein': protein_pc,
                    'Oil': oil_pc,
                    'Carbohydrates': carb_pc,
                    'EFA': efa_pc,
                    'Calories': kcal_per_tonne
                }
        except:
            pass
    return data


def clean_data(df):
    df.drop(['Date', 'Soil_Quality', 'N', 'P', 'K'], axis=1, inplace=True)
    df = df[~df['Crop_Type'].isin(['Cotton', 'Tomato'])]
    return df


# def add_nutritional_values(df):
#     data = find_yield_data()

#     for index, row in df.iterrows():
#         crop = row['Crop_Type']
#         if crop in data:
#             yield_per_ha = row['Crop_Yield']
            
#             protein_tonne = data[crop]['Protein'] * yield_per_ha
#             oil_tonne = data[crop]['Oil'] * yield_per_ha
#             carb_tonne = data[crop]['Carbohydrates'] * yield_per_ha
#             efa_tonne = data[crop]['EFA'] * yield_per_ha
#             kcal = data[crop]['Calories'] * yield_per_ha

#             df.at[index, 'Protein'] = protein_tonne
#             df.at[index, 'Oil'] = oil_tonne
#             df.at[index, 'Carbohydrates'] = carb_tonne
#             df.at[index, 'EFA'] = efa_tonne
#             df.at[index, 'Calories'] = kcal
#         else:
#             df.at[index, 'Protein'] = 0
#             df.at[index, 'Oil'] = 0
#             df.at[index, 'Carbohydrates'] = 0
#             df.at[index, 'EFA'] = 0
#             df.at[index, 'Calories'] = 0
#     return df


def get_dataset():
    df = import_data()
    df = clean_data(df)
    return df


if __name__ == '__main__':
    print(get_dataset())