import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from nltk.tokenize import word_tokenize
import math
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import sqlite3

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """
    Function to load database created in process_data.py file.
    
    Input:
    database_filepath - path where database is stored
    
    Output:
    X - values used to learn model
    y - target variable values
    column_names - name of the columns in database
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM datatable', conn)
    #fill empty spaces with 0
    df = df.fillna(0)
    
    #change values bigger than 1 to 1
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    X = df['message']
    y = df.iloc[:,4:]
    column_names = y.columns.tolist()
    return X, y, column_names

def re_emails(text, replace_for="EMAIL"):
    """
    Function to replace email addresses with the provided word.
    
    Input:
    text - string where we will be replacing words
    replace for - sentence used for replacement
    
    Output:
    Changed text with replaced emails by word "EMAIL".
    """
    
    pattern = r'\b[A-Za-z0-9._%+-][email protected][A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(pattern, replace_for, text) 

def re_urls(text, replace_for="URL"):
    """
    Function to replace urls with the provided word.
    
    Input:
    text - string where we will be replacing words
    replace for - sentence used for replacement
    
    Output:
    Changed text with replaced urls by word "URL".
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[[email protected]&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, replace_for)
    return text 

def re_digits(text, replace_for='DIGITS'): 
    """
    Function which replace all digits by word: "DIGITS"
    
    Input:
    text - string where we will be replacing words
    replace for - sentence used for replacement
    
    Output:
    Changed text with replaced digits by word "DIGITS".
    """
    return re.sub(r'\d+', replace_for, text) 

def re_hashtags(text, replace_for=' '): 
    """
    Function which remove hashtags
    
    Input:
    text - string where we will be removing hashtags
    replace for - character used for replacement
    
    Output:
    Changed text without hashtags
    """
    return re.sub(r'#', replace_for, text) 

def re_user_mentioned(text, replace_for='MENTIONED_USER'): 
    """
    Function which replace mentions ( words starting with @ signs)
    
    Input:
    text - string where we will be replacing words
    replace for - sentence used for replacement
    
    Output:
    Changed text with replaced mentions by word "MENTIONED_USER".
    """
    return re.sub(r'@(\S+)', replace_for, text)

def preprocessing(doc):
    """
    Preprocessing functions which remove/replace unwanted signs/words from strings in database
    
    Input: 
    doc - text which should be changed
    
    Output:
    doc - changed text, without unwanted words
    """
    doc = re_emails(doc)
    doc = re_urls(doc)
    doc = re_digits(doc)
    doc = re_hashtags(doc)
    doc = re_user_mentioned(doc)
    return doc

def remove_stopwords(text):
    """
    Function which remove english stopwords
    
    Input:
    text - string where we will be removing hashtags
    
    Output:
    Changed text without stopwords
    """
    stop = set(stopwords.words('english'))
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

class AdditionalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Class where additional feature will be created
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
       """
       Function which counts how many '@' signs we have in text
       Input:
       X - dataframe where text is stored
       
       Output:
       additional_features - array with number of '@' signs in every row
       """
        additional_features = np.array([text.count('@') for text in X])
        return additional_features.reshape(-1, 1)
    
def tokenize(text):    
    """
    Function which change text to tokens
    
    Input:
    text - string which will be changed to tokens
    
    Output:
    words - tokens
    """
    text_processed = preprocessing(text)
    text_stopwords = remove_stopwords(text_processed)
    tokens = word_tokenize(text_stopwords)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    words=[word.lower() for word in clean_tokens if word.isalpha()]
    return words


def build_model():
    """
    Creating ML Pipeline 
    Specifing GridSearch parameters
    Builind model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())])),

            ('txt_len', AdditionalFeatureExtractor())
        ])),

        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    #specify parameters    
    parameters = { 
    'classifier__estimator__n_estimators': [10,20,50],
    'classifier__estimator__max_depth' : [2,3,5,8],
    'classifier__estimator__random_state': [0],
    'classifier__estimator__min_samples_split': [2, 5],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=1, verbose=2) 
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluating ML model
    
    Input:
    model - trained ML model
    X_test - test dataframe
    Y_test - test target variable
    category_names - column names for test dataframe
    
    Output:
    class_report2 - classificatin report for created model
    """
    predicted_gs = model.predict(X_test)
    class_report2 = classification_report(Y_test, predicted_gs, target_names=category_names)
    return class_report2


def save_model(model, model_filepath):
    """
    Saving model to .pickle file
    
    Input:
    model - trained ML model
    model_filepath - path where model should be saved
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()