# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

import pandas as pd



# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'spam-sms-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        #text = denoise_text(message)
        #text = remove_special_characters(text)
        #text = simple_stemmer(text)
        #text = remove_stopwords(text)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)