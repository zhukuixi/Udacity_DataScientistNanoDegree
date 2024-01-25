import pandas as pd
import numpy as np

class recCF():
    """
    This Recommender uses FunkSVD to make predictions of exact ratings.
    And uses either Collaborative Filtering or a Knowledge Based recommendation (highest ranked) to make recommendations for users.
    """

    def __init__(self):
        self.movies = None
        self.reviews = None
        self.user_by_movie = None
        self.movies_seen = {}
        self.movies_to_analyze = {}
        self.user_distance = None
        # Matrix for SVD
        self.user_mat = None
        self.movie_mat = None

    def fit(self,movies,reviews,latent_features=12,learning_rate=0.0001,iters=100):
        """
        INPUT:
        movies - (Pandas dataframe)
        reviews - (Pandas dataframe)
        latent_features - (int) number of hidden feature of SVD
        learning_rate - (float) learning rate of gradient descent of SVD algorithm
        iters - (int) iteration of gradient descent of SVD algorithm

        OUTPUT: None
        """

        self.movies = movies
        self.reviews = reviews[['user_id', 'movie_id', 'rating']]
        self.reviews['user_id'].astype('int')
        self.user_by_movie = self.reviews.pivot_table(index='user_id', columns='movie_id', values='rating')
        # initialize movies_seen
        self.create_userMovie_dict()
        # initialize movies_to_analyze by filtering movies_seen
        self.create_movies_to_analyze()
        # compute user distance
        self.getUserDistance()

        # Matrix for SVD
        self.user_mat = None
        self.movie_mat = None
        self.FunkSVD(latent_features, learning_rate, iters)

    def predict(self,user_id,movie_id=None,num_recs=5):
        """
        INPUT:
        user_id - (int) user's id
        movie_id - (str) movie's id
        num_recs - (int) number of recommended movies

        OUTPUT:
        recommendations - (list) a list of movie recommendation
        """

        # Predict the rating of given user_id and movie_id
        if movie_id != None:
            user_index = np.where(self.user_by_movie.index==user_id)[0]
            movie_index = np.where(self.user_by_movie.columns==movie_id)[0]
            pred = self.user_mat[user_index,:].dot(self.movie_mat[:,movie_index])[0][0]
            return pred
        # Recommend the top popular movie to user_id based on his/her cloeset neighbor's rating
        else:
            recommendations = None
            try:
                seen = set(self.movies_to_analyze[user_id])
                neighbors = self.find_closest_neighbors(user_id)
                ans = set()
                for neigh in neighbors:
                    for m in self.movie_names(self.movies_liked(neigh)):
                        if m not in seen:
                            ans.add(m)
                    if len(ans) > num_recs:
                        break
                recommendations = list(ans)[:num_recs]
            except:
                print('Cannot make recommendation. \
                       This user needed to have at least 2 ratings to get movie recommendation.')

            return recommendations

    def movie_names(self,movie_ids):
        """
        This function return the movie names of the input movie_ids

        INPUT:
        movie_ids - (list) a list of movie_ids

        OUTPUT:
        movies - (list) a list of movie names associated with the movie_ids
        """

        movie_lst = list(self.movies.query('movie_id.isin(@movie_ids)')['movie'])
        return movie_lst

    def getUserDistance(self):
        """
        This function compute the euclidean distance between users

        OUTPUT: None
        """
        unique_user = list(self.movies_to_analyze.keys())
        df_dist = []
        for i in range(len(unique_user) - 1):
            for j in range(i + 1, len(unique_user)):
                user1, user2 = unique_user[i], unique_user[j]
                corr = self.compute_euclidean_dist(user1, user2)
                df_dist.append([user1, user2, corr])
                df_dist.append([user2, user1, corr])

        self.user_distance = pd.DataFrame(df_dist, columns=['user1', 'user2', 'eucl_dist'])

    def create_userMovie_dict(self):
        """
        This function generates a dictionary where each key is a user_id and the value is an array of movie_ids

        INPUT: None

        OUTPUT: None
        """

        for i in range(self.user_by_movie.shape[0]):
            user_id = int(self.user_by_movie.index[i])
            movie_id = list(self.user_by_movie.iloc[i].loc[pd.isna(self.user_by_movie.iloc[i]) == False].index)
            self.movies_seen[user_id] = movie_id

    def movies_watched(self,user_id):
        """
        This function return a list of movie seen by user_id

        INPUT:
        user_id - (int) the user_id of an individual as int

        OUTPUT:
        movies - (list) an array of movies the user has watched
        """

        movies = self.movies_seen[user_id]
        return movies

    def create_movies_to_analyze(self, lower_bound=2):
        """
        This function generates a dictionary where each key is a user_id and the value is an array of movie_ids

        INPUT:
        movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
        lower_bound - (an int) a user must have more movies seen than the lower bound to be added to the movies_to_analyze dictionary

        OUTPUT: None
        """

        user_to_add = [k for k, v in self.movies_seen.items() if len(v) >= lower_bound]
        for u in user_to_add:
            self.movies_to_analyze[u] = self.movies_seen[u]


    def compute_euclidean_dist(self,user1, user2):
        """
        INPUT:
        user1 - (int) user_id
        user2 - (int) user_id

        OUTPUT:
        the euclidean distance between user1 and user2
        """

        movie1 = self.movies_to_analyze[user1]
        movie2 = self.movies_to_analyze[user2]
        common_movie = list(set(movie1).intersection(set(movie2)))
        data = self.user_by_movie.loc[[user1, user2], common_movie].transpose()
        dist = np.sqrt(sum([(data.iloc[i, 0] - data.iloc[i, 1]) ** 2 for i in range(data.shape[0])]))
        return dist

    def find_closest_neighbors(self,user):
        """
        INPUT:
            user - (int) the user_id of the individual you want to find the closest users
        OUTPUT:
            closest_neighbors - an array of the id's of the users sorted from closest to farthest away
        """

        closest_neighbors = list(self.user_distance.query('user1==@user').sort_values('eucl_dist')['user2'])[1:]
        return closest_neighbors

    def movies_liked(self,user_id, min_rating=7):
        """
        INPUT:
        user_id - the user_id of an individual as int
        min_rating - the minimum rating considered while still a movie is still a "like" and not a "dislike"

        OUTPUT:
        movies_liked - an array of movies the user has watched and liked
        """
        # Implement your code here
        movies_liked = list(self.reviews[['user_id', 'movie_id', 'rating']].query('user_id==@user_id and rating>=@min_rating')['movie_id'])

        return movies_liked

    def FunkSVD(self, latent_features=12, learning_rate=0.0001, iters=100):
        """
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:None
        """

        # Set up useful values to be used through the rest of the function
        rating_matrix = np.array(self.user_by_movie)
        n_users = rating_matrix.shape[0]
        n_movies = rating_matrix.shape[1]

        # u
        self.user_mat = np.random.rand(n_users, latent_features)
        # vt
        self.movie_mat = np.random.rand(latent_features, n_movies)

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        row_ind, col_ind = np.where(~np.isnan(rating_matrix))
        num_ratings = len(row_ind)

        for i in range(iters):
            sse = 0
            for j in range(num_ratings):

                row, col = row_ind[j], col_ind[j]
                u_i = self.user_mat[row, :]
                v_i = self.movie_mat[:, col]
                predict_value = u_i.dot(v_i)
                true_value = rating_matrix[row, col]
                error = true_value - predict_value
                sse += error ** 2
                # update
                self.user_mat[row, :] += (learning_rate * 2 * error * v_i)
                self.movie_mat[:, col] += (learning_rate * 2 * error * u_i)

            print("%d \t\t %f" % (i + 1, sse / num_ratings))




