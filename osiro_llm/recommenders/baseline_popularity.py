import pandas as pd

from osiro_llm.recommenders.baselines import BaseRecommender


class PopularityRecommender(BaseRecommender):
    def __init__(self):
        self.item_popularity = None
        self.ratings_df = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.item_popularity = (
            ratings_df.groupby("MovieID").size().sort_values(ascending=False)
        )
        self.ratings_df = ratings_df

    def recommend(self, user_id: int, n: int, all_movie_ids: list):
        if self.item_popularity is None or self.ratings_df is None:
            raise RuntimeError("The recommender has not been fitted yet.")

        # Get movies user has already rated
        user_rated_movies = set(
            self.ratings_df[self.ratings_df["UserID"] == user_id]["MovieID"]
        )

        # Recommend most popular movies the user hasn't rated
        recommendations = [
            item for item in self.item_popularity.index if item not in user_rated_movies
        ]
        return recommendations[:n]
