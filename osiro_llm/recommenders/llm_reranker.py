import pandas as pd

from osiro_llm.recommenders.baselines import BaseRecommender
from osiro_llm.llm.google import GoogleLLMWrapper
from osiro_llm.llm.prompts import RERANKING_PROMPT


class LLMReranker(BaseRecommender):
    def __init__(self, base_recommender: BaseRecommender, llm_wrapper: GoogleLLMWrapper, rating_threshold=4):
        self.base_recommender = base_recommender
        self.llm_wrapper = llm_wrapper
        self.rating_threshold = rating_threshold
        self.ratings_df = None
        self.movies_df = None
        self.movie_id_to_title = None

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.movie_id_to_title = self.movies_df.set_index("MovieID")["Title"].to_dict()
        self.base_recommender.fit(ratings_df, movies_df)

    def recommend(self, user_id: int, n: int, all_movie_ids: list):
        if self.ratings_df is None or self.movies_df is None:
            raise RuntimeError("The recommender has not been fitted yet.")

        # Get initial candidates from the base recommender
        candidate_ids = self.base_recommender.recommend(user_id, n * 5, all_movie_ids) # Get more candidates to re-rank
        candidate_titles = [self.movie_id_to_title[mid] for mid in candidate_ids if mid in self.movie_id_to_title]

        # Get user's liked movies
        user_ratings = self.ratings_df[self.ratings_df["UserID"] == user_id]
        liked_movies = user_ratings[user_ratings["Rating"] >= self.rating_threshold]
        liked_movie_titles = [self.movie_id_to_title[mid] for mid in liked_movies["MovieID"]]

        prompt = RERANKING_PROMPT.format(
            liked_movies="|".join(liked_movie_titles),
            candidate_movies="|".join(candidate_titles),
            n=n
        )

        response = self.llm_wrapper.generate_content(prompt)
        reranked_titles = [title.strip() for title in response.split("|")]

        # Convert titles back to MovieIDs
        title_to_movie_id = {v: k for k, v in self.movie_id_to_title.items()}
        reranked_ids = [title_to_movie_id[title] for title in reranked_titles if title in title_to_movie_id]

        return reranked_ids[:n]
