from recommender_CF import RecommenderCF
from recommender_content_based import RecommenderContent
from recommender_ranked_based import RecommenderRanked
import pickle


class Recommender:
    def __init__(self,df_path,df_content_path):
        self.rec_CF = RecommenderCF(df_path)
        self.rec_Ranked = RecommenderRanked(df_path)
        self.rec_Content = RecommenderContent(df_content_path)

if __name__ == '__main__':
    rec = Recommender(df_path='./data/user-item-interactions.csv',df_content_path='./data/articles_community.csv')
    with open('recommend.pkl', 'wb') as file:
        pickle.dump(rec, file)

