import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
import torch 
import torch.nn.functional as F
from tqdm import tqdm
import time  # Import time module for measuring execution time
import mlflow

from helpers import calculate_success_at_10, load_config, preprocess_dataframe

mlflow.set_experiment("retrieval_reranking_experiments")

def get_retriever(config: Dict, device:str):
    """
    Creates a single retriever based on the configuration.

    Args:
        config: The configuration for the retriever.

    Returns:
        A retriever instance.
    """
    from retrievers import BiEncoderRetriever, CrossEncoderRetriever, Retriever
    retriever_model = config["retriever_model"]
    model_name = config["model_name"]
    
    if model_name == "CrossEncoderRetriever":
        return CrossEncoderRetriever(retriever_model, model_name, device)
    elif model_name=="BiEncoderRetriever":
        return BiEncoderRetriever(retriever_model, model_name)
    else:
        return Retriever(retriever_model, model_name, device)
    
def get_reranker(config: Dict, devices: list):
    """
    Creates a reranker based on the configuration.

    Args:
        config: The configuration for the reranker.
        device: The device to run the model on.

    Returns:
        A reranker instance.
    """
    from rerankers import CrossEncoderGritLMReranker, BiEncoderReranker, CrossEncoderReranker, Reranker

    reranker_model = config["reranker_model"]
    model_name = config["model_name"]

    if model_name == "BiEncoderReranker":
        return BiEncoderReranker(reranker_model, model_name)
    elif model_name=="CrossEncoderReranker":
        return CrossEncoderReranker(reranker_model, model_name, config["max_length"])
    elif model_name=="CrossEncoderGritLMReranker":
        return CrossEncoderGritLMReranker(reranker_model, model_name)
    else:
        return Reranker(reranker_model, model_name, devices[0])

def retrieve(posts: pd.DataFrame, fact_checks: pd.DataFrame, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor, n: int = 10, batch_size: int = 32):
    """
    Optimized retrieval function with batch processing and memory management.
    
    Args:
        posts: DataFrame containing posts
        fact_checks: DataFrame containing fact checks
        query_embeddings: Tensor of query embeddings
        document_embeddings: Tensor of document embeddings
        n: Number of documents to retrieve
        batch_size: Size of batches for processing
    """
    retrieved_results = {}
    num_queries = len(posts)
    
    # Process queries in batches
    for i in tqdm(range(0, num_queries, batch_size), desc="Processing queries"):
        batch_end = min(i + batch_size, num_queries)
        batch_post_ids = posts['post_id'].iloc[i:batch_end]
        batch_query_embeddings = query_embeddings[i:batch_end]
        
        # Ensure tensors are on the correct device
        if not isinstance(batch_query_embeddings, torch.Tensor):
            batch_query_embeddings = torch.tensor(batch_query_embeddings)
        if not isinstance(document_embeddings, torch.Tensor):
            document_embeddings = torch.tensor(document_embeddings)
            
        # Calculate similarities for the batch
        with torch.no_grad():
            # Reshape query embeddings for batch processing
            batch_query_embeddings = batch_query_embeddings.unsqueeze(1)
            
            # Calculate cosine similarity for the entire batch at once
            similarity_scores = F.cosine_similarity(batch_query_embeddings, document_embeddings, dim=-1)
            
            # Get top-N indices for each query in the batch
            actual_n = min(n, similarity_scores.size(1))
            top_n_indices = similarity_scores.topk(actual_n).indices
            
            # Store results for each query in the batch
            for j, post_id in enumerate(batch_post_ids):
                top_n_fact_check_ids = fact_checks['fact_check_id'].iloc[top_n_indices[j]].tolist()
                retrieved_results[str(post_id)] = top_n_fact_check_ids
        
        # Clear GPU cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return retrieved_results

def rerank(posts: pd.DataFrame, fact_checks: pd.DataFrame, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor, top_retrieved: Dict[str, List[int]], n: int = 10):
    reranked_results = {}
    with torch.no_grad():
        for i, post_id in enumerate(posts['post_id']):
            if str(post_id) in list(top_retrieved.keys()):
                # Get the embedding for the current query
                query_embedding = query_embeddings[i].unsqueeze(0)  # Shape: (1, D)
                # Get the fact_check_ids relevant to the current query from top_retrieved
                relevant_fact_check_ids = top_retrieved.get(str(post_id))
                # Find indices in fact_checks DataFrame for these fact_check_ids
                relevant_docs_mask = fact_checks['fact_check_id'].isin(relevant_fact_check_ids)
                relevant_docs_indices = fact_checks[relevant_docs_mask].index
                # Filter document embeddings to only include the relevant documents for this query
                relevant_docs_embeddings = document_embeddings[relevant_docs_indices]
                query_embedding = torch.tensor(query_embedding)
                relevant_docs_embeddings = torch.tensor(relevant_docs_embeddings)
                # Calculate similarity for the current query against relevant document embeddings
                similarity_scores = F.cosine_similarity(query_embedding, relevant_docs_embeddings, dim=-1)
                # Get top-N document indices based on similarity scores
                actual_n = min(n, similarity_scores.size(0))  # Get the smaller value between n and the tensor size
                top_n_indices = similarity_scores.topk(actual_n).indices.tolist()
                top_n_fact_check_ids = fact_checks['fact_check_id'].iloc[relevant_docs_indices[top_n_indices]].tolist()
                # Store the top-N fact_check_ids for the current post_id
                reranked_results[str(post_id)] = top_n_fact_check_ids
    return reranked_results


def process_language_stage(
    model,
    lang: str, 
    stage: str, 
    experiment_config: Dict, 
    model_config: Dict, 
    task_description: str,
    is_reranker: bool = False
) -> Dict:
    """
    Processes a single language and stage, retrieves or reranks documents, evaluates, and saves predictions.

    Args:
        model (Union[Retriever, Reranker]): The retriever or reranker instance.
        lang (str): Language code.
        stage (str): Stage name.
        experiment_config (Dict): Configuration dictionary for experiments.
        model_config (Dict): Configuration dictionary for the model.
        task_description (str): Description of the retrieval or reranking task.
        is_reranker(bool): Whether the model is a reranker (default: False).

    Returns:
        Dict: A dictionary containing language, stage, and average Success@10.
    """

    lang_stage_path = Path(experiment_config["data_dir"]) / lang / stage
    posts_path = lang_stage_path / "posts.csv"
    fact_checks_path = lang_stage_path / "fact_checks.csv"
    mapping_path = lang_stage_path / "mapping.csv"
    predictions_path = lang_stage_path / model_config.get("top_predictions") if is_reranker else None
    
    if not (fact_checks_path.exists() and posts_path.exists() and (not is_reranker or predictions_path.exists())):
        logging.warning(f"Data missing for {lang} - {stage}. Skipping.")
        return {"language": lang, "stage": stage, "avg_success_at_10": None}

    try:
        posts = pd.read_csv(posts_path)
        fact_checks = pd.read_csv(fact_checks_path)
        mapping_df = pd.read_csv(mapping_path)
        top_retrieved = load_config(predictions_path) if is_reranker else None
        logging.info(f"Loaded data for Language: {lang}, Stage: {stage}")
    except Exception as e:
        logging.error(f"Error reading CSV files for Language: {lang}, Stage: {stage}. Error: {e}")
        return {"language": lang, "stage": stage, "avg_success_at_10": None}

    try:
        fact_checks = preprocess_dataframe(fact_checks, ["claim", "title", "instances"], version=model_config["version"])
        posts = preprocess_dataframe(posts, ["ocr", "text", "instances", "verdicts"], version=model_config["version"])
        logging.info(f"Preprocessed data for Language: {lang}, Stage: {stage}")
    except Exception as e:
        logging.error(f"Error during preprocessing for Language: {lang}, Stage: {stage}. Error: {e}")
        return {"language": lang, "stage": stage, "avg_success_at_10": None}
    
    # Combine relevant columns into a single 'text' column
    fact_checks["text"] = fact_checks.apply(
        lambda row: (
            f"claim: {row['claim']} "
            f"title:{row['title']} instances:{row['instances']} "
        ),
        axis=1
    )
    posts["text"] = posts.apply(
        lambda row: (
            f"ocr: {row['ocr']} "
            f"text: {row['text']} instances:{row['instances']} "
            f"verdict: {row['verdicts']}"
        ),
        axis=1
    )
    # Remove empty texts
    posts = posts[posts['text'].str.strip() != ""]
    fact_checks = fact_checks[fact_checks['text'].str.strip() != ""]

    print(posts["text"][0])
    print(fact_checks["text"][0])

    model_name = model_config["model_name"]
    logging.info(f"Retrieving documents for Language: {lang}, Stage: {stage} using {model_name}")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model", model_config["model_name"])
        mlflow.log_param("batch_size", model_config.get("batch_size", 32))
        mlflow.log_param("language", lang)
        mlflow.log_param("stage", stage)
        mlflow.log_param("top_n", model_config.get("top_n", 10))
        mlflow.log_param("max_length", model_config.get("max_length", 512))
        mlflow.log_param("task_description", task_description)

        try:
            process_function = model.rerank_batch_with_ids if is_reranker else model.retrieve_batch_with_ids
            process_kwargs = {
                "fact_checks": fact_checks,
                "posts": posts,
                "task_description": task_description,
                "n": model_config.get("top_n", 10),
                "max_length": model_config.get("max_length", 512)
            }
            
            if is_reranker:
                process_kwargs["top_retrieved"] = top_retrieved

            # Measure execution time for the retrieval/reranking process
            start_time = time.time()
            processed_documents = process_function(**process_kwargs)
            end_time = time.time()

            # Calculate and log the time taken
            total_time = end_time - start_time
            num_queries = len(posts)
            time_per_query = total_time / num_queries if num_queries > 0 else 0
            logging.info(f"Processing for Language: {lang}, Stage: {stage} took {total_time:.2f} seconds.")
            logging.info(f"Average time per query: {time_per_query:.2f} seconds.")

            logging.info(f"Processed documents for Language: {lang}, Stage: {stage}")
        except torch.cuda.OutOfMemoryError as e:
            logging.error(f"CUDA OOM during retrieval for Language: {lang}, Stage: {stage}. Error: {e}")
            torch.cuda.empty_cache()
            return {"language": lang, "stage": stage, "avg_success_at_10": None}
        except Exception as e:
            logging.error(f"Unexpected error during retrieval for Language: {lang}, Stage: {stage}. Error: {e}")
            torch.cuda.empty_cache()
            return {"language": lang, "stage": stage, "avg_success_at_10": None}
        
        # Initialize a new column in the DataFrame to store relevant documents
        posts['relevant_docs'] = None

        # Map relevant documents from mapping_df
        for idx, row in posts.iterrows():
            relevant_docs = mapping_df[mapping_df['post_id'] == row['post_id']]['fact_check_id'].tolist()
            if relevant_docs:
                posts.at[idx, 'relevant_docs'] = relevant_docs

        # Evaluate Success@10
        total_success_at_10 = 0
        evaluation_queries_len = 0
        for idx, row in posts.iterrows():
            relevant_docs = row["relevant_docs"]
            post_id = row["post_id"]
            if relevant_docs and str(post_id) in processed_documents:
                evaluation_queries_len += 1
                try:
                    total_success_at_10 += calculate_success_at_10(
                        processed_documents[str(post_id)], 
                        relevant_docs, 
                        model_config.get("top_n", 10)
                    )
                except Exception as e:
                    logging.error(f"Error during Success@10 calculation for Post ID: {post_id}. Error: {e}")
        
        avg_success_at_10 = total_success_at_10 / evaluation_queries_len if evaluation_queries_len > 0 else 0
        logging.info(f"Language: {lang}, Stage: {stage}, Avg S@10: {avg_success_at_10}")

        # Log metrics
        mlflow.log_metric("success_at_10", avg_success_at_10)
        mlflow.log_metric("latency_per_query", time_per_query)

        # Set a tag that we can use to remind ourselves what this model was for
        mlflow.set_logged_model_tags(
            model_config["retriever_model"], {"Training Info": "Retrieval/Reranking for Language: {lang}, Stage: {stage}"}
        )

        # Save the processed documents
        try:
            save_predictions_dir = model_config.get("save_predictions")
            output_dir = lang_stage_path / f'predictions_{"reranker" if is_reranker else "retriever"}_{model_config["reranker_model"] if is_reranker else model_config["retriever_model"]}'
            #save_predictions(processed_documents, output_dir, save_predictions_dir)#_orig_new')
            logging.info(f"Retrieved documents saved to {output_dir}/{save_predictions_dir}")#_orig_new")
        except Exception as e:
            logging.error(f"Error saving predictions for Language: {lang}, Stage: {stage}. Error: {e}")

    # Clear GPU cache after processing
    torch.cuda.empty_cache()
    return {"language": lang, "stage": stage, "avg_success_at_10": avg_success_at_10}


def process_documents(
    model,
    tasks: List[Dict], 
    experiment_config: Dict, 
    model_config: Dict, 
    task_description: str,
    is_reranker: bool = False,
) -> List[Dict]:
    """
    Retrieves or reranks documents for multiple language-stage tasks using a single model.

    Args:
        model (Union[Retriever, Reranker]): The retriever or reranker instance.
        tasks (List[Dict]): List of tasks containing language and stage.
        experiment_config (Dict): Configuration dictionary for experiments.
        model_config (Dict): Configuration dictionary for the model.
        task_description (str): Description of the retrieval task.

    Returns:
        List[Dict]: List of results containing language, stage, and avg_success_at_10.
    """
    results = []
    total_tasks = len(tasks)
    task_type = "reranking" if is_reranker else "retrieval"
    logging.info(f"Starting {task_type} for {total_tasks} tasks.")

    for idx, task in enumerate(tasks, 1):
        lang = task['language']
        stage = task['stage']
        logging.info(f"Processing task {idx}/{total_tasks}: Language={lang}, Stage={stage}")
        result = process_language_stage(
            model, 
            lang, 
            stage, 
            experiment_config, 
            model_config, 
            task_description,
            is_reranker
        )
        results.append(result)
    
    logging.info(f"Completed all {task_type} tasks.")
    return results

def log_results(results: List[Dict]):
    overall_s10 = 0
    for res in results:
        lang = res.get("language")
        stage = res.get("stage")
        avg_s10 = res.get("avg_success_at_10")
        overall_s10 += avg_s10
        if avg_s10 is not None:
            logging.info(f"Language: {lang}, Stage: {stage}, Avg S@10: {avg_s10}")
    overall_s10 = overall_s10/len(results)
    logging.info(f"Average S@10: {overall_s10}")
