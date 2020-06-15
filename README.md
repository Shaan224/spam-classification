# Spam Classification for Text Messages

## Introduction

In this repo I have built a classification model to classify a text message as a **"Spam message"** or a **"Normal Message"** using **Natural Language Processing techniques** and **Text Classification**.

Dataset used - [**Kaggle Spam Classification for Text Messages**](https://www.kaggle.com/team-ai/spam-text-message-classification)

I have performed text cleaning and built a Multinomial Naive Bayes model giving the following accuracies, recall and precision:


| Spam | Accuracy | Recall | Precision |
|:-:|---|---|---|
| Train  | 0.99 | 0.97 | 0.96 |
| Test  | 0.98 | 0.95 | 0.93 |

After building the model I deployed this classifier on **Heroku** using **Flask**

Deployed Model - [**Heroku**](https://spam-classification-api.herokuapp.com/)

## Tools And Technologies

* NLTK libraries - Is used for text wrangling and preprocessing of the reviews. Tokenization, Lemmatization and Stopword removal.
* Matplotlib and Plotly - For visualization and creating interactive plots.
* Sklearn Countvectorizer - To create Term-Document Matrix
* Pandas and Numpy - For Data Manipulation and Text Cleaning
* Flask -  For deploying the machine learning model

## Files in this Repository

* static	
  * Not Spam.gif	- Gif file for result.html
  * Spam.gif	- Gif file for result.html
  * mind.ico	- Icon file for the webpage
  * styles.css - Style for formatting of html codes
* templates	
  * homepage.html	- Main html webpage
  * ornaments_texture1136.jpg	- html background image
  * result.html - Result html webpage
* Procfile -  For Heroku web building
* SMSSpamCollection.csv - Datafile	
* Spam Classification.ipynb - Training Jupyter Notebook
* Spam Classification.py	- Training file
* app.py - app file for Flask	
* cv-transform.pkl - Pickled Countvectorizer
* requirements.txt	- Intsalling dependencies and creating a vertual environment
* spam-sms-model.pkl - Pickled Model file
