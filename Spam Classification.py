

#Load the libraries
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


from sklearn.naive_bayes import MultinomialNB


import os





#importing the data
data = pd.read_csv("SMSSpamCollection.csv",delimiter="\t",header=None) #Reading data
data.columns = ['Label','Text'] #Changing Column names

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
#Apply function on review column
data['Cleaned']=data['Text'].apply(denoise_text)



#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
data['Cleaned']=data['Cleaned'].apply(remove_special_characters)

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
data['Cleaned']=data['Cleaned'].apply(simple_stemmer)

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
#Apply function on review column
data['Cleaned']=data['Cleaned'].apply(remove_stopwords)







Encoder =LabelEncoder()
data['LabelEncoded']=Encoder.fit_transform(data['Label'])





# Creating the Bag of Words model

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(data['Cleaned']).toarray()





# Extracting dependent variable from the dataset
y = data['LabelEncoded']



# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))







X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.3,random_state=1234)



# Importing the required Libraries


# Building the Naive Bayes Model
clf_train = MultinomialNB(alpha=0.2)

# Modelling Up on Train & Predicting up on Test
clf_train.fit(X_train, y_train)


# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-sms-model.pkl'
pickle.dump(clf_train, open(filename, 'wb'))




def predict_text(text):
    text = denoise_text(text)
    text = remove_special_characters(text)
    text = simple_stemmer(text)
    final_text = remove_stopwords(text)
    text_vectorizer = cv.transform([final_text]).toarray()
    prediction =  clf_train.predict(text_vectorizer)
    prediction = int(prediction[0])
    if prediction == 1:
        print("This is a SPAM Message")
    else:
        print("This is a normal message")
    return prediction


