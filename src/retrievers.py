import abc
import torch
import numpy as np
import pandas as pd
from torch import nn
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from models_utils import encode, retrieve, get_detailed_instruct

class Retriever(abc.ABC):
    """
    Abstract base class for retrieval strategies.
    """
    def __init__(self, model_name: str, cache_dir: str = "/newstorage5/ipezo/huggingface_cache"):
        """
        Initialize the retriever with model and tokenizer.

        Args:
            model_name (str): Name identifier for the retriever.
            cache_dir (str, optional): Directory to cache models. Defaults to "/newstorage5/ipezo/huggingface_cache".
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()  # Device handling is managed in subclasses
        self.model.eval()
    
    @abc.abstractmethod
    def _load_model(self):
        """
        Abstract method to load the model.
        Must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def _load_tokenizer(self):
        """
        Abstract method to load the tokenizer.
        Must be implemented by subclasses.
        """
        pass

    def _encode(self, combined_texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of combined texts (queries + documents) into embeddings using the model and average pooling.
        """
        return encode(self, combined_texts)
    
    def retrieve_batch_with_ids(
        self, 
        fact_checks: pd.DataFrame,
        posts: pd.DataFrame,
        task_description: str = None,
        n: int = 10,
        max_length: int = 512,
        batch_size: int = 32  # Reduced batch size
    ) -> Dict[str, List[int]]:
        """
        Retrieves documents for multiple queries with or without task instructions using a cross-encoder.

        Args:
            fact_checks (pd.DataFrame): DataFrame containing fact-checks with 'fact_check_id' and 'text'.
            posts (pd.DataFrame): DataFrame containing posts with 'post_id' and 'text'.
            task_description (str, optional): Description of the retrieval task.
            n (int, optional): Number of top similar fact-checks to retrieve. Defaults to 10.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.

        Returns:
            Dict[str, List[int]]: Mapping from post_id to list of relevant fact_check_ids.
        """
        if task_description:
            queries_with_instructions = [get_detailed_instruct(task_description, query) for query in posts['text']]
            queries = queries_with_instructions
        else:
            # Queries without instructions
            queries = [query for query in posts['text']]
            
        # Combine queries and documents for batch tokenization
        documents_texts = fact_checks['text'].tolist()
        combined_texts = queries + documents_texts
        
        # Encode combined texts in a single batch to ensure aligned embeddings
        embeddings = self._encode(combined_texts, batch_size=batch_size, max_length=max_length)
        
        # Separate query and document embeddings
        query_embeddings = embeddings[:len(queries)]
        document_embeddings = embeddings[len(queries):]
        
        return retrieve(posts, fact_checks, query_embeddings, document_embeddings, n)

class BiEncoderRetriever:
    def __init__(self, retriever_model: str, model_name: str):
        """
        Initialize the retriever with a pre-trained model and tokenizer.
        """
        self.retriever_model = retriever_model
        super().__init__(model_name)
    
    def _load_model(self):
        """
        Load the model on a specific GPU and manage computation on other GPUs.
        """
        model = SentenceTransformer("BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16})
        model = nn.DataParallel(model, device_ids=[0])  # Use available GPU IDs
        return model

    def _load_tokenizer(self):
        """
        Load the tokenizer for the transformer model.
        """
        return AutoTokenizer.from_pretrained(self.retriever_model, trust_remote_code=True, use_fast=False)
    
    def tokenize(self, texts):
        self.tokenizer = self._load_tokenizer()
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt") 

    def retrieve_batch_with_ids(
        self,
        fact_checks: pd.DataFrame,
        posts: pd.DataFrame,
        task_description: str = None,
        n: int = 10,
        max_length: int = 512,
        batch_size: int = 4,
    ) -> Dict[str, List[int]]:
        """
        Reranks documents for multiple queries with or without task instructions using a bi-encoder.

        Args:
            fact_checks (pd.DataFrame): DataFrame containing fact-checks with 'fact_check_id' and 'text'.
            posts (pd.DataFrame): DataFrame containing posts with 'post_id' and 'text'.
            task_description (str, optional): Description of the retrieval task.
            n (int, optional): Number of top similar fact-checks to retrieve. Defaults to 10.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.

        Returns:
            Dict[str, List[int]]: Mapping from post_id to list of relevant fact_check_ids.
        """
        documents_texts = fact_checks['text'].tolist()

        query_prefix = f"<instruct>: {task_description}\n<query>: " if task_description else ""
        queries = [query_prefix + query for query in posts['text']]

        query_embeddings = self.model.module.encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
        document_embeddings = self.model.module.encode(documents_texts, batch_size=batch_size, instruction="", max_length=max_length, num_workers=32, return_numpy=True)
        if isinstance(document_embeddings, np.ndarray):
            document_embeddings = torch.tensor(document_embeddings)

        return retrieve(posts, fact_checks, query_embeddings, document_embeddings, n)


class CrossEncoderRetriever(Retriever):
    """
    Retrieves the documents using a multilingual transformer model with instruction input.
    """
    def __init__(self, retriever_model: str, model_name: str, device: str):
        """
        Initialize TransformerRetriever with a specific transformer model.

        Args:
            retriever_model (str): The model identifier from Hugging Face.
            model_name (str): Name identifier for the retriever.
            device (str): Device to run the model on ('cuda:0', 'cuda:1', etc.).
        """
        self.device = device
        self.retriever_model = retriever_model
        super().__init__(model_name)
    
    def _load_model(self):
        """
        Load a transformer model and move it to the specified device.
        """
        model = AutoModel.from_pretrained(self.retriever_model, cache_dir=self.cache_dir)
        model.to(self.device)
        return model
    
    def _load_tokenizer(self):
        """
        Load a tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.retriever_model, cache_dir=self.cache_dir)
        
    def load_pretrained(self, model_path: str):
        """
        Return a finetuned model wrapped with a TransformerRetriever.
        """
        model = SentenceTransformer(model_path)
        model.to(self.device)
        return self
