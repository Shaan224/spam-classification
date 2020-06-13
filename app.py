# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.tokenize.toktok import ToktokTokenizer

#Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
def remove_between_brackets(text):
    return re.sub('[\!"#$%&\'()*+,-./:;<=>?@\][\\\^_`{|}~]'," ",text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_between_brackets(text)
    return text




#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z\s]'
    text=re.sub(pattern,'',text)
    return text


#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text


#set stopwords to english
stopword_list.extend(['aa','aah','aaniye','abj','ag','aaooooright','aathilove','aathiwhere','ab','abbey','u','r','k','p','n','c'])

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token.lower() for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text



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
        text = denoise_text(message)
        text = remove_special_characters(text)
        text = simple_stemmer(text)
        text = remove_stopwords(text)
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)