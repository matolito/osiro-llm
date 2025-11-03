import random
import pandas as pd

from osiro_llm.recommenders.baselines import BaseRecommender


class RandomRecommender(BaseRecommender):
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        pass

    def recommend(self, user_id: int, n: int, all_movie_ids: list):
        return random.sample(all_movie_ids, n)
