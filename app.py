from flask import Flask, request, jsonify, render_template
from index import StockPredictor
import numpy as np

app = Flask(__name__)
predictor = StockPredictor('TSLA', '1m')
predictor.load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_new = predictor.preprocess_new_data(data)
    predictions = predictor.predict(X_new)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)