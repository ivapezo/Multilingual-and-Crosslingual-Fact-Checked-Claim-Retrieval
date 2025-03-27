from pathlib import Path
import pandas as pd
import logging
from ensemblers_utils import aggregate_predictions, evaluate_predictions, load_predictions
from helpers import load_config, save_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

if __name__ == "__main__":
    # Load configuration
    base_path = Path("Task7/config")
    experiment_config = load_config(base_path / "experiment_config.json")
    pipeline_config = load_config(base_path / "pipeline_config.json")
    mode =  pipeline_config.get("ensembler_mode") 
    ensembler_config = pipeline_config.get(f"{mode}_ensembler", {})

    # Iterate through languages and stages
    languages = experiment_config.get("languages", [])
    stages = experiment_config.get("stages", [])
    task_results = []
    average = 0

    for lang in languages:
        for stage in stages:

            posts_path = Path(f"Task7/data/{lang}/{stage}/posts.csv")
            mapping_path = Path(f"Task7/data/{lang}/{stage}/mapping.csv")

            if not (posts_path.exists()):
                logging.warning(f"Data files missing for Language: {lang}, Stage: {stage}. Skipping.")
                continue

            # Load predictions
            predictions = [
                load_predictions(Path(f"Task7/data/{lang}/{stage}/{ensembler_config['predictions_dirs'][i]}"))
                for i in range(len(ensembler_config["predictions_dirs"]))
            ]

            # Aggregate predictions
            aggregated_predictions = aggregate_predictions(predictions,
                                                            weights=[1.0, 0.5, 1.0], 
                                                            top_n=ensembler_config.get("top_n", 100))

            # Save aggregated predictions
            try:
                dir = ensembler_config["output_dir"]
                file = ensembler_config["output_file"]
                output_dir = Path(f'Task7/data/{lang}/{stage}/{dir}')
                save_predictions(aggregated_predictions, output_dir, ensembler_config.get("output_file"))
                logging.info(f"Retrieved documents saved to {output_dir}/{file}.json")
            except Exception as e:
                logging.error(f"Error saving predictions for Language: {lang}, Stage: {stage}. Error: {e}")

            # Evaluate predictions
            posts = pd.read_csv(posts_path)
            mapping_df = pd.read_csv(mapping_path)

            avg_s_at_10 = evaluate_predictions(
                posts=posts,
                retrieved_documents=aggregated_predictions,
                mapping_df=mapping_df,
                lang=lang,
                stage=stage,
            )

            task_results.append({"language": lang, "stage": stage, "avg_success_at_10": avg_s_at_10})
            average += avg_s_at_10


    for result in task_results:
        logging.info(f"Language: {result['language']}, Stage: {result['stage']}, Avg S@10: {result['avg_success_at_10']}")
    
    logging.info(f"Avg S@10: {average/len(task_results)}")
