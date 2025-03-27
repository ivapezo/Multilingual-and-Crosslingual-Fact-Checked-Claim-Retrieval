from collections import defaultdict
from heapq import nlargest
import json
import math
from pathlib import Path
from typing import Dict, List
from itertools import product as itertools_product
import logging

import pandas as pd

from helpers import calculate_success_at_10

def load_predictions(file_path: Path) -> Dict[str, List[int]]:
    """
    Load predictions from a JSON file.

    Args:
        file_path (Path): Path to the predictions file.

    Returns:
        Dict[str, List[int]]: Predictions loaded from the file.
    """
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return {}
    
    with open(file_path, "r") as file:
        predictions = json.load(file)
    return predictions

def aggregate_predictions(
    predictions_list: List[Dict[str, List[int]]], 
    method: str = "rrf", 
    weights: List[float] = None,
    top_n: int = 10,
) -> Dict[str, List[int]]:
    """
    Aggregate predictions from multiple sources using different ranking fusion methods.

    Args:
        predictions_list (List[Dict[str, List[int]]]): List of prediction dictionaries.
        method (str): Aggregation method ("average", "exp_decay", or "rrf").
        weights (List[float], optional): Weights for each prediction list. Defaults to equal weighting.
        top_n (int): Number of top fact-checks to return.

    Returns:
        Dict[str, List[int]]: Aggregated top-N predictions per post_id.
    """
    if not predictions_list:
        return {}

    weights = weights if weights else [1] * len(predictions_list)
    if len(weights) != len(predictions_list):
        print(len(predictions_list))
        raise ValueError("Number of weights must match the number of prediction lists.")
    
    offset=50
    lambda_decay=0.1
    aggregated = defaultdict(lambda: defaultdict(float))

    # Aggregate scores from all sources
    for predictions, weight in zip(predictions_list, weights):
        for post_id, fact_check_ids in predictions.items():
            for rank, fact_check_id in enumerate(fact_check_ids):
                if method == "average":
                    score = weight*(1 / (rank + 1))
                elif method == "exp_decay":
                    score = weight * math.exp(-lambda_decay * rank)
                elif method == "rrf":
                    score = weight / (rank + offset)
                else:
                    raise ValueError(f"Invalid aggregation method: {method}")
                aggregated[post_id][fact_check_id] += score

    # Sort and extract top-N results
    final_predictions = {
        #post_id: [int(fact_check_id) for fact_check_id, _ in nlargest(top_n, fact_check_scores.items(), key=lambda x: x[1])]
        post_id: [fact_check_id for fact_check_id, _ in nlargest(top_n, fact_check_scores.items(), key=lambda x: x[1])]
        for post_id, fact_check_scores in aggregated.items()
    }

    return final_predictions


def evaluate_predictions(
    posts: pd.DataFrame,
    retrieved_documents: Dict[str, List[int]],
    mapping_df: pd.DataFrame,
    top_n: int,
    lang: str,
    stage: str,
) -> float:
    """
    Evaluate predictions using Success@10.

    Args:
        posts (pd.DataFrame): DataFrame of posts.
        retrieved_documents (Dict[str, List[int]]): Predictions to evaluate.
        mapping_df (pd.DataFrame): Mapping of relevant documents.
        top_n (int): N results to evaluate.
        lang (str): Language code.
        stage (str): Evaluation stage.

    Returns:
        float: Average Success@10 score.
    """

    # Map relevant fact-check IDs to posts
    posts = posts.merge(
        mapping_df.groupby("post_id")["fact_check_id"].agg(list).reset_index(), 
        on="post_id", 
        how="left"
    ).rename(columns={"fact_check_id": "relevant_docs"})
    
    # Fill missing values with empty lists
    posts["relevant_docs"] = posts["relevant_docs"].apply(lambda x: x if isinstance(x, list) else [])

    def compute_row_success(row):
        if row["relevant_docs"] and str(row["post_id"]) in retrieved_documents:
            try:
                return calculate_success_at_10(
                    retrieved_documents[str(row["post_id"])],
                    row["relevant_docs"],
                    top_n,
                )
            except Exception as e:
                logging.error(f"Error during Success@10 calculation for Post ID: {row['post_id']}. Error: {e}")
        return 0

    posts["success_at_k"] = posts.apply(compute_row_success, axis=1)

    # Compute average S@k
    evaluation_queries_len = (posts["success_at_k"] > 0).sum()
    avg_success_at_10 = posts["success_at_k"].sum() / evaluation_queries_len if evaluation_queries_len > 0 else 0

    logging.info(f"Language: {lang}, Stage: {stage}, Avg S@10: {avg_success_at_10}")
    return avg_success_at_10
    

def search_optimal_aggregation_settings(
    predictions_list: List[Dict[str, List[int]]],
    language: str,
    evaluation_fn,
    methods: List[str] = ["average", "exp_decay", "rrf"],
    weight_ranges: List[List[float]] = None,
    top_n: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Search for optimal aggregation settings for a specific language.
    """
    if weight_ranges is None:
        num_rerankers = len(predictions_list)
        weight_ranges = [[1.0, 0.5, 1.0]] * num_rerankers  # Default weight options for each reranker

    best_score = -float('inf')
    best_settings = {"method": None, "weights": None, "score": None}

    for method, weights in itertools_product(methods, itertools_product(*weight_ranges)):
        aggregated_predictions = aggregate_predictions(
            predictions_list=predictions_list,
            method=method,
            weights=list(weights),
            top_n=top_n
        )
        score = evaluation_fn(aggregated_predictions, language)

        if score > best_score:
            best_score = score
            best_settings = {
                "method": method,
                "weights": list(weights),
                "score": score,
            }

    return best_settings