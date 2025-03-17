"""
Module for helper functions
"""
import ast
import datetime
import json
from pathlib import Path
from typing import Dict
import pandas as pd
import torch
import logging
import os
from preprocessing_pipeline import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"#"2, 3, 4, 5" #"2 3" #"1"  # 1 = 5, 0 = 4, 2 = 0 (GeForce prvi)

def check_gpu_availability(min_free_gb: float = 1.0) -> list:
    """
    Checks the availability of GPUs and returns a list of available GPU devices 
    that have at least `min_free_gb` of free memory. If no GPUs meet the criteria,
    defaults to using the CPU.

    Args:
        min_free_gb (float, optional): Minimum required free memory per GPU in GB.
                                       Defaults to 1.0 GB.

    Returns:
        list: A list of torch.device objects representing available GPUs or CPU.
    """

    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()
    available_devices = []

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of GPUs detected: {num_gpus}")

        gpu_info = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024 ** 3)  # in GB
            allocated_mem = torch.cuda.memory_allocated(i) / (1024 ** 3)
            free_mem = total_mem - allocated_mem
            gpu_info.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_gb': total_mem,
                'allocated_gb': allocated_mem,
                'free_gb': free_mem
            })

        # Sort GPUs by free memory in descending order
        gpu_info_sorted = sorted(gpu_info, key=lambda x: x['free_gb'], reverse=True)
        logging.info("GPU Status:")
        for gpu in gpu_info_sorted:
            logging.info(
                f"  - Device {gpu['id']}: {gpu['name']}, "
                f"Total: {gpu['total_gb']:.2f} GB, "
                f"Allocated: {gpu['allocated_gb']:.2f} GB, "
                f"Free: {gpu['free_gb']:.2f} GB"
            )

        # Select GPUs with enough free memory
        for gpu in gpu_info_sorted:
            if gpu['free_gb'] >= min_free_gb:
                device = torch.device(f"cuda:{gpu['id']}")
                available_devices.append(device)
                logging.info(
                    f"Selected Device {gpu['id']}: {gpu['name']} "
                    f"with {gpu['free_gb']:.2f} GB free memory."
                )

        if not available_devices:
            logging.warning(f"No GPUs have at least {min_free_gb} GB free memory. Defaulting to CPU.")
            available_devices.append(torch.device("cpu"))
    else:
        logging.info("No GPU detected. Using CPU.")
        available_devices.append(torch.device("cpu"))

    return available_devices

def save_predictions(predictions, output_file_path: Path, file_name: str):
    """
    Save predictions to the specified output file in json format.
    """
    output_file_path.mkdir(parents=True, exist_ok=True)
    with open(output_file_path / f'{file_name}.json', 'w') as f:
        json.dump(predictions, f, indent=4)

def extract_texts_and_language(row):
    """
    Extracts the original text, English text, and language from the given row.
    """

    # Handle NaN or empty values
    if pd.isna(row) or row.strip() == "[]":
        return pd.Series(["", "", ""])

    try:
        parsed_data = ast.literal_eval(row.strip())
    except (SyntaxError, ValueError):
        return pd.Series(["", "", ""])  # Handle invalid JSON-like input gracefully
    
    # Ensure parsed_data is always a list for uniform processing
    parsed_data = parsed_data if isinstance(parsed_data, list) else [parsed_data]

    # Extract original text, English text, and language
    original_texts = [entry[0] if entry and len(entry) > 0 else "" for entry in parsed_data]
    english_texts = [entry[1] if entry and len(entry) > 1 else "" for entry in parsed_data]
    languages = [next(iter(entry[2][0]), "") if entry and len(entry) > 2 and entry[2] else "" for entry in parsed_data]

    return pd.Series(["\n".join(original_texts), "\n".join(english_texts), languages[0]])  # Return first language found

def load_config(filepath: str) -> Dict:
    """
    Loads the configuration from the specified file.

    Args:
        filepath: The path to the configuration file.

    Returns:
        The configuration as a dictionary.
    """
    with open(filepath) as f:
        return json.load(f)
    
def calculate_success_at_10(retrieved_docs, relevant_docs, top_n):
    if not retrieved_docs:  # Handle None or empty list
        return 0
    return int(any(doc_id in relevant_docs for doc_id in retrieved_docs[:top_n]))

def preprocess_dataframe(df, text_columns, version, lang="eng", preprocess=False):
    for col in text_columns:
        if col == "instances":
            df[col] = df[col].apply(lambda x: extract_instances(x))
        elif col == "verdicts":
            df[col] = df[col].apply(lambda x: str(x).strip("['']"))
        else:
            df[col] = df[col].apply(lambda x: extract_texts_and_language(x)[version])
        if preprocess:
            df[col] = df[col].apply(lambda x: preprocessing(x, lang))
    return df

def extract_instances(row):
    """
    Extracts the timestamp and the URL from the given row and returns them in a formatted string.
    Handles invalid timestamps and missing data, and labels claims based on the fact-check status.
    """
    # Handle missing or empty input
    if pd.isna(row) or row.strip() == "[]":
        logging.warning("Empty or missing data encountered.")
        return ""

    # Attempt to parse row safely
    try:
        parsed_data = ast.literal_eval(row)
        if not isinstance(parsed_data, list):
            logging.warning("Invalid data format encountered.")
            return "Invalid data format"
    except (SyntaxError, ValueError) as e:
        logging.error(f"Parsing error: {e}")
        return "Parsing error"
    
     # Function to convert timestamp to date
    def format_timestamp(unix_timestamp):
        if isinstance(unix_timestamp, (int, float)):
            try:
                return datetime.datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d")
            except (OSError, ValueError) as e:
                logging.error(f"Invalid timestamp encountered: {e}")
                return "Invalid timestamp"
        logging.warning("Missing or non-numeric timestamp.")
        return "No timestamp"

    # Process timestamps
    formatted_dates = [format_timestamp(entry[0]) for entry in parsed_data if len(entry) >= 1]

    # Join formatted dates with ' and '
    result = " and ".join(formatted_dates) if formatted_dates else "No valid timestamps"
    
    logging.info(f"Extracted instances: {result}")
    return result