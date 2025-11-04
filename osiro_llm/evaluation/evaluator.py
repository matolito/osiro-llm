import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

from osiro_llm.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


class Evaluator:
    def __init__(self, models: dict, k=10, rating_threshold=4, llm_delay=1):
        self.models = models
        self.k = k
        self.rating_threshold = rating_threshold
        self.llm_delay = llm_delay

    def evaluate(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, movies_df: pd.DataFrame
    ):

        # 2. Create the ground truth map (test items)
        test_user_items = (
            test_df[test_df["Rating"] >= self.rating_threshold]
            .groupby("UserID")["MovieID"]
            .apply(set)
            .to_dict()
        )

        # 2b. Create the seen items map (train items)
        train_user_items = train_df.groupby("UserID")["MovieID"].apply(set).to_dict()

        # Get the list of users we need to evaluate (those with items in the test set).
        test_user_ids = list(test_user_items.keys())

        # Convert to a set for fast O(1) lookups and filtering
        all_movie_ids_set = set(movies_df["MovieID"].unique())
        results = {}

        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")

            # 3. Fit the model on the training portion of the ratings.
            model.fit(train_df, movies_df)

            metrics = {"precision@k": [], "recall@k": [], "ndcg@k": []}

            for user_id in tqdm(
                test_user_ids[:30], desc=f"Predicting for {model_name}"
            ):
                if "LLM" in model_name:
                    time.sleep(self.llm_delay)

                # Get the ground truth list of relevant items from our map.
                relevant_items = test_user_items.get(user_id, set())
                if not relevant_items:
                    continue

                # 4. Get candidate items by filtering seen items.
                # Get items this user saw in the training set
                seen_items = train_user_items.get(user_id, set())

                # Create a list of candidates: all movies MINUS seen movies
                candidate_items = list(all_movie_ids_set - seen_items)

                # 5. Get recommendations ONLY from candidate items.
                recommendations = model.recommend(user_id, self.k, candidate_items)

                # Calculate metrics
                metrics["precision@k"].append(
                    precision_at_k(recommendations, relevant_items, self.k)
                )
                metrics["recall@k"].append(
                    recall_at_k(recommendations, relevant_items, self.k)
                )
                metrics["ndcg@k"].append(
                    ndcg_at_k(recommendations, relevant_items, self.k)
                )

            # Average metrics
            avg_metrics = {k: sum(v) / len(v) if v else 0 for k, v in metrics.items()}
            results[model_name] = avg_metrics
            print(f"Results for {model_name}: {avg_metrics}")

        return results
