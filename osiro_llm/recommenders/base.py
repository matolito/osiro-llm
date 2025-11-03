from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        pass

    @abstractmethod
    def recommend(self, user_id: int, n: int, all_movie_ids: list):
        pass