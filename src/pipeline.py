import logging
import os
import pandas as pd
from pathlib import Path
from BM25 import create_bm25_index, retrieve_top_n
from models_utils import get_retriever, get_reranker, process_documents
from ensemblers_utils import aggregate_predictions, evaluate_predictions, load_predictions
from helpers import load_config, save_predictions, check_gpu_availability
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data_files(lang: str, stage: str, base_path: str = "Task7/data/clef-2021") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load and validate data files for a given language and stage."""
    lang_stage_path = Path(f"{base_path}/{lang}/{stage}")
    fact_checks_path = lang_stage_path / 'fact_checks.csv'
    posts_path = lang_stage_path / 'posts.csv'

    if not (fact_checks_path.exists() and posts_path.exists()):
        logging.warning(f"Data missing for {lang} - {stage}. Skipping.")
        return None, None

    return pd.read_csv(fact_checks_path), pd.read_csv(posts_path)

def process_model_predictions(
    model_config: Dict[str, Any],
    tasks: List[Dict[str, str]],
    experiment_config: Dict[str, Any],
    available_devices: List[str],
    is_reranker: bool = False
) -> List[Dict[str, Any]]:
    """Process predictions for a model (retriever or reranker)."""
    results = []
    for config in model_config:
        selected_device = available_devices[0]
        model = get_reranker(config, selected_device) if is_reranker else get_retriever(config, selected_device)
        
        task_description = config.get("task_description", "")
        result = process_documents(
            model=model,
            tasks=tasks,
            experiment_config=experiment_config,
            model_config=config,
            task_description=task_description,
            is_reranker=is_reranker,
        )
        results.append(result)
    return results

def load_and_aggregate_predictions(
    languages: List[str],
    stages: List[str],
    ensembler_config: Dict[str, Any],
    additional_predictions: Dict = None
) -> Dict[str, List[int]]:
    """Load predictions for all language/stage combinations and aggregate them."""
    predictions = []
    
    for lang in languages:
        for stage in stages:
            for i in range(len(ensembler_config["predictions_dirs"])):
                pred = load_predictions(Path(f"Task7/data/clef-2021/{lang}/{stage}/{ensembler_config['predictions_dirs'][i]}"))
                predictions.append(pred)
    
    if additional_predictions:
        predictions.append(additional_predictions)

    return aggregate_predictions(
        predictions_list=predictions,
        weights=ensembler_config.get("weights", [1.0] * len(predictions)),
        top_n=ensembler_config.get("top_n", 10),
    )

def evaluate_results(
    languages: List[str],
    stages: List[str],
    aggregated_predictions: Dict[str, List[int]],
    ensembler_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Evaluate results for all language/stage combinations."""
    final_results = []
    total_success = 0

    for lang in languages:
        for stage in stages:
            fact_checks, posts = load_data_files(lang, stage)
            if fact_checks is None or posts is None:
                continue

            mapping_path = Path(f"Task7/data/clef-2021/{lang}/{stage}/mapping.csv")
            if not mapping_path.exists():
                logging.warning(f"Mapping file missing for {lang} - {stage}. Skipping.")
                continue

            mapping_df = pd.read_csv(mapping_path)
            avg_s_at_10 = evaluate_predictions(
                posts=posts,
                retrieved_documents=aggregated_predictions,
                mapping_df=mapping_df,
                top_n=ensembler_config.get("top_n", 10),
                lang=lang,
                stage=stage,
            )

            final_results.append({"language": lang, "stage": stage, "avg_success_at_10": avg_s_at_10})
            total_success += avg_s_at_10

    return final_results, total_success

if __name__ == "__main__":
    try:
        # Load configuration files
        base_path = Path(__file__).parent.parent / "config"
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
        retrieval_results = process_model_predictions(
            pipeline_config["retrievers"],
            tasks,
            experiment_config,
            available_devices
        )

        # Run BM25 Retrieval
        bm25_config = pipeline_config.get("bm25", {})
        top_n = bm25_config.get("top_n", 10)
        k1 = bm25_config.get("k1", 1.5)
        b = bm25_config.get("b", 0.75)

        bm25_predictions = {}
        for lang in languages:
            for stage in stages:
                fact_checks, posts = load_data_files(lang, stage)
                if fact_checks is None or posts is None:
                    continue

                documents = fact_checks['claim'].fillna('') + " " + fact_checks['title'].fillna('')
                bm25 = create_bm25_index(documents.tolist(), k1=k1, b=b)

                for _, row in posts.iterrows():
                    query = str(row["ocr"]) + " " + str(row["text"])
                    retrieved_docs = retrieve_top_n(bm25, query, documents.tolist(), top_n)
                    retrieved_doc_ids = [int(fact_checks.iloc[doc_id]["fact_check_id"]) for doc_id in retrieved_docs]
                    bm25_predictions[int(row["post_id"])] = retrieved_doc_ids

        # Save BM25 predictions
        save_predictions(bm25_predictions, Path("Task7/data/bm25_predictions.json"), "bm25_predictions.json")

        # Step 2: Ensemble Retrieval Results (Neural + BM25)
        ensembler_config = pipeline_config.get("retriever_ensembler", {})
        aggregated_retrieved = load_and_aggregate_predictions(
            languages,
            stages,
            ensembler_config,
            bm25_predictions
        )

        # Save aggregated retrieval results
        save_predictions(aggregated_retrieved, Path("Task7/data/aggregated_retrieval.json"), "aggregated_retrieval.json")
        
        # Step 3: Run Rerankers
        reranked_results = process_model_predictions(
            pipeline_config["rerankers"],
            tasks,
            experiment_config,
            available_devices,
            is_reranker=True
        )

        # Step 4: Ensemble Reranked Results
        ensembler_config = pipeline_config.get("reranker_ensembler", {})
        aggregated_reranked = load_and_aggregate_predictions(
            languages,
            stages,
            ensembler_config
        )

        # Save final reranked results
        save_predictions(aggregated_reranked, Path("Task7/data/clef-2021/eng/train/final_reranked"), "final_reranked.json")

        # Step 5: Evaluate Final Results
        final_results, total_success = evaluate_results(
            languages,
            stages,
            aggregated_reranked,
            ensembler_config
        )

        # Step 6: Log Final Results
        for result in final_results:
            logging.info(f"Language: {result['language']}, Stage: {result['stage']}, Avg S@10: {result['avg_success_at_10']}")

        if final_results:
            logging.info(f"Final Avg S@10: {total_success / len(final_results)}")
        else:
            logging.warning("No valid tasks processed. Final Avg S@10 cannot be computed.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise
