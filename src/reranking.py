"""
Script to rerank the top-k results from the initial retrieval model using a reranker.
"""

import os
import logging
from pathlib import Path
from models_utils import get_reranker, log_results, process_documents
from helpers import (
    check_gpu_availability,
    load_config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    base_path = Path("Task7/config")
    pipeline_config = load_config(base_path / "pipeline_config.json")
    experiment_config = load_config(base_path / "experiment_config.json")

    # Check GPU availability
    available_devices = check_gpu_availability(min_free_gb=2.0)
    logging.info(f"Available devices: {available_devices}")

    if not available_devices:
        logging.error("No available GPUs with sufficient free memory. Exiting.")
        exit(1)

    logging.info(f"Selected devices for reranking: {available_devices}")

    # Prepare list of tasks (language and stage)
    tasks = []
    for lang in experiment_config.get("languages", []):
        for stage in experiment_config.get("stages", []):
            tasks.append({"language": lang, "stage": stage})

    logging.info(f"Prepared {len(tasks)} tasks for reranking.")

    # Iterate through the defined rerankers
    for reranker_config in pipeline_config["rerankers"]:
        reranker = get_reranker(reranker_config, available_devices)
        task_description = reranker_config.get("task_description", None)
        logging.info(f"Starting retrieval using {reranker_config['model_name']} on {available_devices}")
        results = process_documents(
            model=reranker,
            tasks=tasks,
            experiment_config=experiment_config,
            model_config=reranker_config,
            task_description=task_description,
            is_reranker=True
        )

        # Log overall results
        log_results(results)