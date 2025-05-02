from flask import Flask, request, render_template
import pickle
import numpy as np
# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
        model = pickle.load(file)
app = Flask(__name__)

@app.route('/')
def home():
        return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
                float_features=[float(x) for x in request.form.values()]
                final_features = [np.array(float_features)]
                prediction = model.predict(final_features)
                float_prediction = float(prediction)
                final_prediction = round(float_prediction,3)
                return render_template('index.html',prediction_text='{} Rs.'.format(final_prediction))

if __name__== "__main__":
    app.run(debug=True)
