import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from osiro_llm.recommenders.baselines import BaseRecommender


class UserCFRecommender(BaseRecommender):
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix = None
        self.user_similarity = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        # Create user-item matrix
        self.user_item_matrix = pd.pivot_table(
            ratings_df, values="Rating", index="UserID", columns="MovieID"
        ).fillna(0)

        # Calculate user similarity
        user_similarity_matrix = cosine_similarity(self.user_item_matrix)

        # To avoid recommending items to the user themselves, set diagonal to 0
        np.fill_diagonal(user_similarity_matrix, 0)

        self.user_similarity = pd.DataFrame(
            user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def recommend(self, user_id: int, n: int, all_movie_ids: list):
        if self.user_item_matrix is None:
            raise RuntimeError("The recommender has not been fitted yet.")

        if user_id not in self.user_item_matrix.index:
            # Cold start user. Return empty list.
            return []

        # 1. Get top k similar users (neighbors)
        similar_users = (
            self.user_similarity[user_id].sort_values(ascending=False).head(self.k)
        )

        # 2. Get items rated by neighbors
        neighbor_ratings = self.user_item_matrix.loc[similar_users.index]

        # 3. Predict scores (weighted average of ratings from similar users)
        # Numerator: sum(similarity * rating)
        numerator = neighbor_ratings.mul(similar_users, axis=0).sum(axis=0)

        # Denominator: sum of similarities for users who have rated the item
        rated_by_neighbors = neighbor_ratings.copy()
        rated_by_neighbors[rated_by_neighbors > 0] = 1
        denominator = rated_by_neighbors.mul(similar_users, axis=0).sum(axis=0)

        # Avoid division by zero by adding a small epsilon
        denominator[denominator == 0] = 1e-10

        predicted_scores = numerator / denominator

        # 4. Filter out already seen items and sort
        user_rated_movies = self.user_item_matrix.loc[user_id]
        user_rated_movies = user_rated_movies[user_rated_movies > 0].index

        predicted_scores = predicted_scores.drop(user_rated_movies, errors="ignore")
        recommendations = predicted_scores.sort_values(ascending=False).head(n)

        return recommendations.index.tolist()
