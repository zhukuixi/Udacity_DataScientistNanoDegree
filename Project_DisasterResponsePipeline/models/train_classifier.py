# import libraries
import sys
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
import pickle
    
# NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re 
import nltk
nltk.download(['stopwords','punkt', 'wordnet', 'averaged_perceptron_tagger'])

# DS
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine.connect())                       
    X = df.message
    Y = df.iloc[:,4:]
    return X, Y, Y.columns.values

def tokenize(text):
    # normalize
    re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize
    words = word_tokenize(text)
    
    # stop word removal
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatization 
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return tokens


def build_model():
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('rf',MultiOutputClassifier(RandomForestClassifier()))
                   ])        
    
    parameters = {
        'rf__estimator__n_estimators' : [50, 100,150]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], Y_pred[:, index]))



def save_model(model, model_filepath):   
    filename = f'{model_filepath}'
    pickle.dump(model, open(filename, 'wb'))


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