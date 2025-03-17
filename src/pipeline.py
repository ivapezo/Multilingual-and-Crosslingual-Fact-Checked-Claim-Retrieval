import logging
import pandas as pd
from pathlib import Path
from BM25 import create_bm25_index, retrieve_top_n
from models_utils import get_retriever, get_reranker, process_documents
from ensemblers_utils import aggregate_predictions, evaluate_predictions, load_predictions
from helpers import load_config, save_predictions, check_gpu_availability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    # Load configuration files
    base_path = Path("Task7/config")
    pipeline_config = load_config(base_path / "pipeline_config.json")
    experiment_config = load_config(base_path / "experiment_config.json")

    # Check GPU availability
    available_devices = check_gpu_availability(min_free_gb=2.0)
    logging.info(f"Available devices: {available_devices}")

    if not available_devices:
        logging.error("No available GPUs with sufficient free memory. Exiting.")
        exit(1)

    # Prepare tasks (languages & stages)
    languages = experiment_config.get("languages", [])
    stages = experiment_config.get("stages", [])
    tasks = [{"language": lang, "stage": stage} for lang in languages for stage in stages]

    # Step 1: Run Retrieval Models (Neural & BM25)
    retrieval_results = []

    # Run Neural Retrieval Models
    for retriever_config in pipeline_config["retrievers"]:
        selected_device = available_devices[0]  # Assign the first available device
        retriever = get_retriever(retriever_config, selected_device)

        task_description = retriever_config.get("task_description", "")
        results = process_documents(
            model=retriever,
            tasks=tasks,
            experiment_config=experiment_config,
            model_config=retriever_config,
            task_description=task_description,
        )

        retrieval_results.append(results)

    # Run BM25 Retrieval
    bm25_config = pipeline_config.get("bm25", {})
    top_n = bm25_config.get("top_n", 10)
    k1 = bm25_config.get("k1", 1.5)
    b = bm25_config.get("b", 0.75)

    bm25_predictions = {}

    for lang in languages:
        for stage in stages:
            lang_stage_path = Path(f"Task7/data/{lang}/{stage}")
            fact_checks_path = lang_stage_path / 'fact_checks.csv'
            posts_path = lang_stage_path / 'posts.csv'

            if not (fact_checks_path.exists() and posts_path.exists()):
                logging.warning(f"Data missing for {lang} - {stage}. Skipping BM25 retrieval.")
                continue

            fact_checks = pd.read_csv(fact_checks_path)
            posts = pd.read_csv(posts_path)

            documents = fact_checks['claim'].fillna('') + " " + fact_checks['title'].fillna('')
            bm25 = create_bm25_index(documents.tolist(), k1=k1, b=b)

            for _, row in posts.iterrows():
                query = row["ocr"] + " " + row["text"]
                retrieved_docs = retrieve_top_n(bm25, query, top_n)
                retrieved_doc_ids = [int(fact_checks.iloc[doc_id]["fact_check_id"]) for doc_id in retrieved_docs]
                bm25_predictions[int(row["post_id"])] = retrieved_doc_ids

    # Save BM25 predictions
    save_predictions(bm25_predictions, Path("Task7/data/bm25_predictions.json"), "bm25_predictions.json")

    # Step 2: Ensemble Retrieval Results (Neural + BM25)
    ensembler_config = pipeline_config.get("retriever_ensembler", {})
    retrieved_predictions = [
        load_predictions(Path(f"Task7/data/{lang}/{stage}/{ensembler_config['predictions_dirs'][i]}"))
        for i in range(len(ensembler_config["predictions_dirs"]))
    ]
    retrieved_predictions.append(bm25_predictions)  # Include BM25 results

    aggregated_retrieved = aggregate_predictions(
        predictions_list=retrieved_predictions,
        weights=ensembler_config.get("weights", [1.0] * len(retrieved_predictions)),
        top_n=ensembler_config.get("top_n", 10),
    )

    # Save aggregated retrieval results
    save_predictions(aggregated_retrieved, Path("Task7/data/aggregated_retrieval.json"), "aggregated_retrieval.json")

    # Step 3: Run Rerankers
    reranked_results = []
    for reranker_config in pipeline_config["rerankers"]:
        selected_device = available_devices[0]  # Assign the first available device
        reranker = get_reranker(reranker_config, selected_device)

        task_description = reranker_config.get("task_description", "")
        results = process_documents(
            model=reranker,
            tasks=tasks,
            experiment_config=experiment_config,
            model_config=reranker_config,
            task_description=task_description,
            is_reranker=True,
        )

        reranked_results.append(results)

    # Step 4: Ensemble Reranked Results
    ensembler_config = pipeline_config.get("reranker_ensembler", {})
    reranked_predictions = [
        load_predictions(Path(f"Task7/data/{lang}/{stage}/{ensembler_config['predictions_dirs'][i]}"))
        for i in range(len(ensembler_config["predictions_dirs"]))
    ]
    
    aggregated_reranked = aggregate_predictions(
        predictions_list=reranked_predictions,
        weights=ensembler_config.get("weights", [1.0] * len(reranked_predictions)),
        top_n=ensembler_config.get("top_n", 10),
    )

    # Save final reranked results
    save_predictions(aggregated_reranked, Path("Task7/data/final_reranked.json"), "final_reranked.json")

    # Step 5: Evaluate Final Results
    final_results = []
    total_success = 0

    for lang in languages:
        for stage in stages:
            posts_path = Path(f"Task7/data/{lang}/{stage}/posts.csv")
            mapping_path = Path(f"Task7/data/{lang}/{stage}/mapping.csv")

            if not posts_path.exists() or not mapping_path.exists():
                logging.warning(f"Data files missing for Language: {lang}, Stage: {stage}. Skipping.")
                continue

            posts = pd.read_csv(posts_path)
            mapping_df = pd.read_csv(mapping_path)

            avg_s_at_10 = evaluate_predictions(
                posts=posts,
                retrieved_documents=aggregated_reranked,
                mapping_df=mapping_df,
                top_n=ensembler_config.get("top_n", 10),
                lang=lang,
                stage=stage,
            )

            final_results.append({"language": lang, "stage": stage, "avg_success_at_10": avg_s_at_10})
            total_success += avg_s_at_10

    # Step 6: Log Final Results
    for result in final_results:
        logging.info(f"Language: {result['language']}, Stage: {result['stage']}, Avg S@10: {result['avg_success_at_10']}")

    if final_results:
        logging.info(f"Final Avg S@10: {total_success / len(final_results)}")
    else:
        logging.warning("No valid tasks processed. Final Avg S@10 cannot be computed.")
