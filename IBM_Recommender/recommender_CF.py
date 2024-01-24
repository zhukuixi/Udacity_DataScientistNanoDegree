# create the user-article matrix with 1's and 0's
import pandas as pd
import numpy as np


class RecommenderCF:

    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.user_item = None
        self.create_user_item_matrix()

    def email_mapper(self):
        coded_dict = dict()
        cter = 1
        email_encoded = []

        for val in self.df['email']:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter += 1
            email_encoded.append(coded_dict[val])

        return email_encoded

    def create_user_item_matrix(self):
        '''
        INPUT:
        df - pandas dataframe with article_id, title, user_id columns

        Description:
        Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with
        an article and a 0 otherwise
        '''

        email_encoded = self.email_mapper()
        del self.df['email']
        self.df['user_id'] = email_encoded

        self.df['article_id'] = self.df['article_id'].astype(str)
        self.user_item = self.df.pivot_table(index='user_id', columns='article_id', aggfunc='size')
        self.user_item.mask(self.user_item > 0, 1, inplace=True)
        self.user_item.fillna(0, inplace=True)

    def find_similar_users(self,user_id):
        """
        INPUT:
        user_id - (int) a user_id
        user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first

        Description:
        Computes the similarity of every pair of users based on the dot product
        Returns an ordered

        """

        # compute similarity of each user to the provided user
        matrix_user_item = np.array(self.user_item)
        index = np.where(self.user_item.index == user_id)[0][0]
        user_vector = matrix_user_item[index, :]
        user_similairty = matrix_user_item.dot(user_vector)
        # sort by similarity
        user_similarity_sorted = sorted(zip(user_similairty, self.user_item.index), key=lambda x: x[0], reverse=True)
        # create list of just the ids
        most_similar_users = [id for _, id in user_similarity_sorted if id != user_id]

        return most_similar_users

    def get_article_names(self,article_ids):
        """
        INPUT:
        article_ids - (list) a list of article ids
        df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the title column)
        """

        article_names = list(self.df.loc[self.df['article_id'].isin(article_ids), 'title'].drop_duplicates())
        return article_names

    def get_user_articles(self,user_id):
        """
        INPUT:
        user_id - (int) a user id
        user_item - (pandas dataframe) matrix of users by articles:
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
        article_names - (list) a list of article names associated with the list of article ids
                        (this is identified by the doc_full_name column in df_content)

        Description:
        Provides a list of the article_ids and article titles that have been seen by a user
        """

        user_row = self.user_item.loc[user_id, :]
        index = np.where(user_row > 0)[0]
        article_ids = list(self.user_item.columns[index])
        article_names = self.get_article_names(article_ids)

        return article_ids, article_names

    def get_top_sorted_users(self,user_id):
        """
        INPUT:
        user_id - (int)
        df - (pandas dataframe) df as defined at the top of the notebook
        user_item - (pandas dataframe) matrix of users by articles:
                1's when a user has interacted with an article, 0 otherwise


        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        neighbor_id - is a neighbor user_id
                        similarity - measure of the similarity of each user to the provided user_id
                        num_interactions - the number of articles viewed by the user - if a u

        Other Details - sort the neighbors_df by the similarity and then by number of interactions where
                        highest of each is higher in the dataframe

        """

        # compute similarity of each user to the provided user
        mapping_user_interaction = dict(self.df.groupby('user_id').size())
        user_interaction = self.user_item.index.map(mapping_user_interaction)
        matrix_user_item = np.array(self.user_item)
        index = np.where(self.user_item.index == user_id)[0][0]
        user_vector = matrix_user_item[index, :]
        user_similarity = matrix_user_item.dot(user_vector)
        neighbors_df = pd.DataFrame({'neighbor_id': self.user_item.index,
                                     'similarity': user_similarity,
                                     'num_interactions': user_interaction
                                     }).sort_values(['similarity', 'num_interactions'], ascending=False)
        neighbors_df = neighbors_df.query('neighbor_id!=@user_id')

        return neighbors_df

    def user_user_recs_part2(self, user_id, m=10):
        """
        INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title

        Description:
        Loops through the users based on closeness to the input user_id
        For each user - finds articles the user hasn't seen before and provides them as recs
        Does this until m recommendations are found

        Notes:
        * Choose the users that have the most total article interactions
        before choosing those with fewer article interactions.

        * Choose articles with the articles with the most total interactions
        before choosing those with fewer total interactions.

        """
        user_seen_id, user_seen_title = self.get_user_articles(user_id)
        similar_users = self.get_top_sorted_users(user_id)['neighbor_id']
        mapping_article_interaction = dict(self.df.groupby('article_id').size())

        recs = []
        for neighbor in similar_users:
            neighbor_seen_id, neighbor_seen_title = self.get_user_articles(neighbor)
            neighbor_seen_id.sort(key=lambda x: mapping_article_interaction[x], reverse=True)
            recommended_id = set(neighbor_seen_id) - set(user_seen_id)
            recs.extend(recommended_id)
            if len(recs) >= m:
                break

        recs = recs[:m]
        rec_names = self.get_article_names(recs)
        return rec_names
