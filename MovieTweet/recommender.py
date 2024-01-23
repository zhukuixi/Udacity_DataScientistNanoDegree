import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import progressbar
from recKnowledge import recKnowledge
from recCF import recCF

import sys # can use sys to take command line arguments


class Recommender():
    '''
    This class reads movie and reviews data provided by user and will predict the rating for
    a user-movie pair and do recommend a list of movies to users.
    '''

    def __init__(self):
        self.recommender_knowledge = recKnowledge()
        self.recommender_CF = recCF()

    def fit(self,movies,reviews ):
        """
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        """
        self.recommender_knowledge.fit(movies,reviews)
        self.recommender_CF.fit(movies,reviews,'SVD')

    def predict_rating(self, user_id,movie_id):
        """
        makes predictions of a rating for a user on a movie-user combo
        """
        return self.recommender_CF.predict(user_id,movie_id)

    def make_recs(self,user_id,style='knowledge'):
        """
        given a user id or a movie that an individual likes
        make recommendations
        """
        if style == 'knowledge':
            return self.recommender_knowledge.predict(user_id)
        elif style == 'CF':
            return self.recommender_CF.predict(user_id)


if __name__ == '__main__':
    # read in data
    test_dataSize = 1000
    movies = pd.read_csv('./data/original_movies.dat',
                        delimiter='::',
                        header=None,
                        names=['movie_id', 'movie', 'genre'],
                        dtype={'movie_id': object}, engine='python')
    reviews = pd.read_csv('./data/original_ratings.dat',
                          delimiter='::',
                          header=None,
                          names=['user_id', 'movie_id', 'rating', 'timestamp'],
                          dtype={'movie_id': object, 'user_id': object, 'timestamp': object},
                          engine='python')
    reviews['user_id'] = reviews['user_id'].astype('int')
    # Reduce the size reviews dataset
    reviews = reviews.loc[:test_dataSize, :]

    # preprocessing
    movies['date'] = movies['movie'].str[-5:-1]
    dummy_time = pd.get_dummies(movies['date'].str[:2] + "00's")
    movies_new = pd.concat([movies, dummy_time], axis=1)
    total_genres = set()

    for gen in movies_new['genre'].dropna().str.split("|"):
        for g in gen:
            total_genres.add(g)

    def getCategory(x, g):
        if pd.isna(x):
            return 0
        return 1 if g in x else 0

    for g in total_genres:
        movies_new[g] = movies_new['genre'].map(lambda x: getCategory(x, g))

    reviews['date'] = reviews['timestamp'].apply(lambda x: datetime.fromtimestamp(int(x)))
    reviews_new = reviews
    print(reviews_new.head())

    rec = Recommender()
    rec.fit(movies_new,reviews_new)
    print(rec.predict_rating(2,'0358273'))
    print(rec.make_recs(2,'knowledge'))
    print(rec.make_recs(2,'CF'))
