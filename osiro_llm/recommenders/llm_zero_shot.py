import pandas as pd

from osiro_llm.recommenders.baselines import BaseRecommender
from osiro_llm.llm.google import GoogleLLMWrapper
from osiro_llm.llm.prompts import ZERO_SHOT_PROMPT


class LLMZeroShotRecommender(BaseRecommender):
    def __init__(self, llm_wrapper: GoogleLLMWrapper, rating_threshold=4):
        self.llm_wrapper = llm_wrapper
        self.rating_threshold = rating_threshold
        self.ratings_df = None
        self.movies_df = None
        self.movie_id_to_title = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.movie_id_to_title = self.movies_df.set_index("MovieID")["Title"].to_dict()

    def recommend(self, user_id: int, n: int, all_movie_ids: list):
        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError("The recommender has not been fitted yet.")

        # Get user's liked movies
        user_ratings = self.ratings_df[self.ratings_df["UserID"] == user_id]
        liked_movies = user_ratings[user_ratings["Rating"] >= self.rating_threshold]
        liked_movie_titles = [self.movie_id_to_title[mid] for mid in liked_movies["MovieID"]]

        # Get candidate movies
        user_rated_movies = set(user_ratings["MovieID"])
        candidate_movie_ids = [mid for mid in all_movie_ids if mid not in user_rated_movies]
        candidate_movie_titles = [self.movie_id_to_title[mid] for mid in candidate_movie_ids]
        
        # For simplicity, let's limit the number of candidates sent to the LLM
        candidate_movie_titles = candidate_movie_titles[:100]

        prompt = ZERO_SHOT_PROMPT.format(
            liked_movies="|".join(liked_movie_titles),
            candidate_movies="|".join(candidate_movie_titles),
            n=n
        )

        response = self.llm_wrapper.generate_content(prompt)
        recommended_titles = [title.strip() for title in response.split("|")]

        # Convert titles back to MovieIDs
        title_to_movie_id = {v: k for k, v in self.movie_id_to_title.items()}
        recommended_ids = [title_to_movie_id[title] for title in recommended_titles if title in title_to_movie_id]

        return recommended_ids[:n]
