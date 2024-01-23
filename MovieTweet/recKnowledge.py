import pandas as pd
import numpy as np

class recKnowledge:
    def __init__(self):
        self.ranked_movies = None

    def fit(self, movies, reviews):
        """
        This function generated a dataset consisted of movies sorted by highest avg rating, more reviews,
        latest review date. All movies in the output should at least have 4 ratings.

        INPUT
        movies - (Pandas dataframe) the movies dataframe
        reviews - (Pandas dataframe) the reviews dataframe

        OUTPUT
        ranked_movies - (Pandas dataframe) a dataframe with movies that are sorted by highest avg rating, more reviews,
                      then time, and must have more than 4 ratings
        """
        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        self.ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with only 5 or more reviews
        self.ranked_movies = self.ranked_movies[self.ranked_movies['num_ratings'] > 4]

        return self.ranked_movies

    def predict(self,user_id, n_top=10, years=None, genres=None):
        """
        This function select the top movie names from the movie ranking dataframe generated
        by function create_ranked_df. It allows filtering on movie years and genres.

        INPUT:
        user_id - the user_id (str) of the individual you are making recommendations for
        n_top - (int) an integer of the number recommendations you want back
        ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time
        years - (list) a list of strings with years of movies
        genres - (list) a list of strings with genres of movies

        OUTPUT:
        top_movies - (list) a list of the n_top recommended movies by movie title in order best to worst
        """

        # Step 1: filter movies based on year and genre
        n = self.ranked_movies.shape[0]
        filter_years = self.ranked_movies['date'].isin(years) if years is not None else np.array([True]*n)
        filter_genres = self.ranked_movies[genres].sum(axis=1) > 0 if genres is not None else np.array([True]*n)
        row_filter = filter_years & filter_genres

        # Step 2: create top movies list
        recommendations = list(self.ranked_movies.loc[row_filter, 'movie'][:n_top])
        return recommendations
