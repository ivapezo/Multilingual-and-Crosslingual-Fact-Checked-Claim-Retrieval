import json
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
import logging
from helpers import calculate_success_at_10, load_config, preprocess_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def create_bm25_index(docs, k1=1.5, b=0.75):
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    return bm25

def retrieve_top_n(bm25, query, top_n):
    preprocessed_query = query.split()
    scores = bm25.get_scores(preprocessed_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [index for index in top_indices]

def main():
    experiment_config = load_config("Task7/config/experiment_config.json")
    bm25_config = load_config("Task7/config/model_config_bm25.json")
    top_n = bm25_config["top_n"]
    output_dir = Path(experiment_config["data_dir"])
    k1 = bm25_config["model_params"]["k1"]
    b = bm25_config["model_params"]["b"]
    version = bm25_config["version"]

    for lang in experiment_config["languages"]:
        for stage in experiment_config["stages"]:
            lang_stage_path = output_dir / lang / stage
            fact_checks_path = lang_stage_path / 'fact_checks.csv'
            posts_path = lang_stage_path / 'posts.csv'
            if stage == "train":
                mapping_path = lang_stage_path / 'mapping.csv'
                mapping_df = pd.read_csv(mapping_path)

            if not (fact_checks_path.exists() and posts_path.exists()):
                print(f"Data missing for {lang} - {stage}. Skipping.")
                continue

            fact_checks = pd.read_csv(fact_checks_path)
            posts = pd.read_csv(posts_path)

            fact_checks = preprocess_dataframe(fact_checks, ["claim", "title", "instances"], version, lang, preprocess=True)
            logging.info(f"Preprocessed fact checks for Language: {lang}, Stage: {stage}")
            posts = preprocess_dataframe(posts, ["ocr", "text", "instances", "verdicts"], version, lang, preprocess=True)
            logging.info(f"Preprocessed posts for Language: {lang}, Stage: {stage}")

            documents = [row["claim"] + " " + row["title"] + " " + row["instances"] for _, row in fact_checks.iterrows()]

            evaluation_queries = []
            for _, row in posts.iterrows():
                if stage == "train":
                    relevant_docs = mapping_df[mapping_df['post_id'].eq(row['post_id'])]['fact_check_id'].tolist()
                else:
                    relevant_docs = []
                evaluation_queries.append({"query": row["ocr"] + " " + row["text"] + " " + row["instances"] + " " + row["verdicts"],
                                            "relevant_docs": relevant_docs,
                                            "post_id": row["post_id"]
                                        })

            bm25 = create_bm25_index(documents, k1=k1, b=b)
            logging.info(f"BM25 index created...")

            total_success_at_10 = 0
            predictions = {}
            for eval_query in evaluation_queries:
                query = eval_query["query"]
                post_id = eval_query["post_id"]
                if stage == "train":
                    relevant_docs = eval_query["relevant_docs"]

                retrieved_docs = retrieve_top_n(bm25, query, top_n)
                #retrieved_docs = [int(fact_checks.iloc[doc_id]["fact_check_id"]) for doc_id in retrieved_docs]
                #predictions[int(post_id)] = retrieved_docs
                retrieved_docs = [fact_checks.iloc[doc_id]["fact_check_id"] for doc_id in retrieved_docs]
                predictions[post_id] = retrieved_docs

                if stage == "train":
                    total_success_at_10 += calculate_success_at_10(retrieved_docs, relevant_docs, top_n)
            
            if stage == "train":
                avg_success_at_10 = total_success_at_10 / len(evaluation_queries)
                print(f"Language: {lang}, Stage: {stage}, Avg S@10: {avg_success_at_10}")

            # Save predictions to a JSON file
            predictions_output_path = lang_stage_path / f'{bm25_config["save_predictions"]}_{top_n}_eng'
            predictions_output_path.mkdir(parents=True, exist_ok=True)
            with open(predictions_output_path / f'predictions.json', 'w') as f:
                json.dump(predictions, f, indent=4)
            print(f"Predictions saved to {predictions_output_path}")

if __name__ == "__main__":
    main()
