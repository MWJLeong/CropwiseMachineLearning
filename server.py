from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def landing():
    return 'ML Crop Prediction\n'


@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    data = request.json
    humidity = data.get('humidity', 0.0)
    temperature = data.get('temperature', 0.0)
    nutrient = data.get('nutrient', '')

    crops = predict(humidity, temperature, nutrient)
    return jsonify({'yields': crops})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000)