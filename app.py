import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import json

app = Flask(__name__)

model = pickle.load(open('testcase_classifier.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    input_data_as_numpy_array = np.asarray(query_df)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input_data_scaled = scaler.transform(input_data_reshaped)
    prediction = model.predict(input_data_scaled)
    print(prediction)
    if (prediction[0] == 1):
        print('Priority is High')
        pred_val = "High"
    elif (prediction[0] == 2):
        print('Priority is Medium')
        pred_val = "Medium"
    else:
        print('Priority is Low')
        pred_val = "Low"
    return pred_val


if __name__ == "__main__":
    app.run(debug=True, port=4000)








    