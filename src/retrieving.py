"""
Script to retrieve the top-k results using the retriever model using a single retriever.
"""
from pathlib import Path
import logging
from models_utils import get_retriever, log_results, process_documents
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

if __name__ == "__main__":
    base_path = Path("Task7/config")
    pipeline_config = load_config(base_path / "pipeline_config.json")
    experiment_config = load_config(base_path / "experiment_config.json")
    
    # Check GPU availability
    available_devices = check_gpu_availability(min_free_gb=2.0)  # Adjust min_free_gb as needed
    logging.info(f"Available devices: {available_devices}")
    
    if available_devices:
        selected_device = available_devices[0]
        logging.info(f"Selected device for retriever: {selected_device}")
    else:
        logging.error("No available GPUs with sufficient free memory. Exiting.")
        exit(1)

    # Prepare list of tasks (language and stage)
    tasks = []
    for lang in experiment_config.get("languages", []):
        for stage in experiment_config.get("stages", []):
            tasks.append({"language": lang, "stage": stage})

    logging.info(f"Prepared {len(tasks)} tasks for retrieval.")

     # Iterate through the defined rerankers
    for retriever_config in pipeline_config["retrievers"]:
        retriever = get_retriever(retriever_config, selected_device)
        task_description = retriever_config.get("task_description", None)
        logging.info(f"Starting retrieval using {retriever_config['model_name']} on {selected_device}")
        results = process_documents(
            model=retriever, 
            tasks=tasks, 
            experiment_config=experiment_config, 
            model_config=retriever_config, 
            task_description=task_description
        )

        # Log overall results
        log_results(results)