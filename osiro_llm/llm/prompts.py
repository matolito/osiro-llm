ZERO_SHOT_PROMPT = """
You are a movie recommendation expert.
A user has watched and liked the following movies:
{liked_movies}

Here is a list of movies they have not seen yet:
{candidate_movies}

Based on the user's preferences, which of the candidate movies would you recommend?
Please return a list of {n} movie titles, and nothing else.
The list should be formatted as a pipe-separated list.
Example: Movie Title 1 (YYYY) | Movie Title 2 (YYYY) | Movie Title 3 (YYYY)
"""

RERANKING_PROMPT = """
You are a movie re-ranking expert.
A user has watched and liked the following movies:
{liked_movies}

A standard recommendation algorithm has suggested the following movies for the user, in this order:
{candidate_movies}

Please re-rank this list of movies to better match the user's tastes.
Return the re-ranked list of {n} movie titles, and nothing else.
The list should be formatted as a pipe-separated list.
Example: Movie Title 1 (YYYY) | Movie Title 2 (YYYY) | Movie Title 3 (YYYY)
"""
