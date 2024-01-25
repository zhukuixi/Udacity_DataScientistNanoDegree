

import pandas as pd

class RecommenderRanked():
    def __init__(self,df_path):
        self.df = pd.read_csv(df_path)
        del self.df['Unnamed: 0']

    def get_top_articles(self, n=5):
        """
        INPUT:
        n - (int) the number of top articles to return
        df - (pandas dataframe) a dataframe describe user item interaction

        OUTPUT:
        top_articles - (list) A list of the top 'n' article titles

        """
        top_articles = list(self.df.groupby(['article_id', 'title']).size().reset_index(name='read_cnt') \
                                                                    .sort_values('read_cnt', ascending=False)['title'])[:n]

        return top_articles  # Return the top article titles from df (not df_content)

