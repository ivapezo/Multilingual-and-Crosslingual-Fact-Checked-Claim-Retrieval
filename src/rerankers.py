import abc
import torch
import pandas as pd
from torch import nn
from gritlm import GritLM
from typing import List, Dict
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModel, AutoTokenizer
from models_utils import rerank
from encoder_utils import encode, gritlm_instruction, get_detailed_instruct, last_token_pool

class Reranker(abc.ABC):
    """
    Abstract base class for reranking strategies.
    """
    def __init__(self, model_name: str, cache_dir: str = "/newstorage5/ipezo/huggingface_cache"):
        """
        Initialize the reranker with model and tokenizer.

        Args:
            model_name (str): Name identifier for the reranker.
            cache_dir (str, optional): Directory to cache models. Defaults to "/newstorage5/ipezo/huggingface_cache".
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
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

    def _encode(self, combined_texts: List[str], max_length: int = 512, batch_size: int = 16):
        """
        Encodes a list of combined texts (queries + documents) into embeddings using the model and average pooling.
        """
        return encode(self, combined_texts)
    
    def _encode_texts(self, texts: List[str], instruction: str = "", max_length: int = 512, batch_size: int = 16) -> torch.Tensor:
        """
        Process texts with an optional instruction and encode them.
        Subclasses can override this to modify the instruction handling.
        """
        processed_texts = [instruction + text if instruction else text for text in texts]
        return self._encode(processed_texts, batch_size=batch_size, max_length=max_length)

    
    def rerank_batch_with_ids(
        self, 
        fact_checks: pd.DataFrame,
        posts: pd.DataFrame,
        top_retrieved: Dict[str, List[int]],
        task_description: str = None,
        n: int = 10,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> Dict[str, List[int]]:
        """
        Reranks documents for multiple queries with or without task instructions.

        Args:
            fact_checks (pd.DataFrame): DataFrame containing fact-checks with 'fact_check_id' and 'text'.
            posts (pd.DataFrame): DataFrame containing posts with 'post_id' and 'text'.
            top_retrieved (Dict[str, List[int]]): Dictionary with a list of top candidate fact-checks for a given claim.
            task_description (str, optional): Description of the retrieval task.
            n (int, optional): Number of top similar fact-checks to retrieve. Defaults to 10.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.

        Returns:
            Dict[str, List[int]]: Mapping from post_id to list of relevant fact_check_ids.
        """
        if task_description:
            # Construct instruction-based queries
            queries = [get_detailed_instruct(task_description, query) for query in posts['text']]
        else:
            # Queries without instructions
            queries = [query for query in posts['text']]

        # Combine queries and documents for batch tokenization
        documents = fact_checks['text'].tolist()

         # Encode queries and documents together
        combined_texts = queries + documents
        embeddings = self._encode_texts(combined_texts, instruction="", batch_size=batch_size, max_length=max_length)
        
        # Separate query and document embeddings
        query_embeddings = embeddings[:len(queries)]
        document_embeddings = embeddings[len(queries):]
        
        return rerank(posts, fact_checks, query_embeddings, document_embeddings, top_retrieved, n)

class BiEncoderReranker(Reranker):
    def __init__(self, reranker_model: str, model_name: str):
        """
        Initialize the reranker with a pre-trained model and tokenizer.
        """
        self.reranker_model = reranker_model
        super().__init__(model_name)
    
    def _load_model(self):
        """
        Load a transformer model and use DataParallel to utilize multiple GPUs.
        """
        model = AutoModel.from_pretrained(self.reranker_model,
                                          device_map='auto', 
                                          trust_remote_code=True, 
                                          torch_dtype=torch.float16,  # Using mixed precision
                                          cache_dir=self.cache_dir)
        model = nn.DataParallel(model, device_ids=[0, 1])  # Using available GPU IDs
        return model

    def _load_tokenizer(self):
        """
        Load the tokenizer for the transformer model.
        """
        return AutoTokenizer.from_pretrained(self.reranker_model, trust_remote_code=True, use_fast=False)
    
    def rerank_batch_with_ids(
        self, 
        fact_checks: pd.DataFrame,
        posts: pd.DataFrame,
        top_retrieved: Dict[str, List[int]],
        task_description: str = None,
        n: int = 10,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> Dict[str, List[int]]:
        """
        Reranks documents for multiple queries with or without task instructions.

        Args:
            fact_checks (pd.DataFrame): DataFrame containing fact-checks with 'fact_check_id' and 'text'.
            posts (pd.DataFrame): DataFrame containing posts with 'post_id' and 'text'.
            top_retrieved (Dict[str, List[int]]): Dictionary with a list of top candidate fact-checks for a given claim.
            task_description (str, optional): Description of the retrieval task.
            n (int, optional): Number of top similar fact-checks to retrieve. Defaults to 10.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.

        Returns:
            Dict[str, List[int]]: Mapping from post_id to list of relevant fact_check_ids.
        """

        query_prefix = "Instruct: " + task_description + "\nQuery: "

        queries = [query for query in posts['text']]
            
        # Combine queries and documents for batch tokenization
        documents = fact_checks['text'].tolist()

        # Use the underlying module's _do_encode function
        query_embeddings = torch.from_numpy(self.model.module._do_encode(
            queries, batch_size=batch_size, instruction=query_prefix, 
            max_length=max_length, num_workers=32, return_numpy=True
        ))
        document_embeddings = torch.from_numpy(self.model.module._do_encode(
            documents, batch_size=batch_size, instruction="",
            max_length=max_length, num_workers=32, return_numpy=True
        ))

        return rerank(posts, fact_checks, query_embeddings, document_embeddings, top_retrieved, n)

class CrossEncoderGritLMReranker(Reranker):
    """
    Reranker implementation using the GritLM model.
    """

    def __init__(self, reranker_model: str, model_name: str):
        """
        Initialize the reranker with a pre-trained model and tokenizer.
        """
        self.model_name = model_name
        self.reranker_model = reranker_model
        self.model = self._load_model()
        self.model.eval()

    def _load_model(self):
        """
        Load the GritLM model.

        Returns:
            GritLM: Loaded GritLM model.
        """
        return GritLM(self.reranker_model, torch_dtype="auto", device_map="auto")

    def _load_tokenizer(self):
        """
        GritLM does not require an explicit tokenizer.

        Returns:
            None
        """
        return None

    def _encode_texts(self, texts: List[str], instruction: str, max_length: int = 512, batch_size: int = 16) -> torch.Tensor:
        """
        Encode texts into embeddings using the GritLM model.

        Args:
            texts (List[str]): Input texts to encode.
            instruction (str): Instruction text for encoding.
            batch_size (int): Batch size for encoding.

        Returns:
            torch.Tensor: Normalized embeddings.
        """
        embeddings = []
        formatted_instruction = gritlm_instruction(instruction)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(batch_texts, instruction=formatted_instruction, max_length=max_length)
                embeddings.append(torch.tensor(batch_embeddings))

        return F.normalize(torch.cat(embeddings, dim=0), p=2, dim=1)

class CrossEncoderReranker(Reranker):
    """
    Optimized reranker using Qwen transformer model with task instructions.
    """
    def __init__(self, reranker_model: str, model_name: str, max_length: int = 2048):
        """
        Initialize the reranker with model and tokenizer.

        Args:
            model_name (str): Hugging Face model name.
            max_length (int): Max sequence length for tokenization.
            device (str): Device for computation.
        """
        self.reranker_model = reranker_model
        self.max_length = max_length
        super().__init__(model_name)

    def _load_model(self):
        """
        Load the model (implemented from the base class).
        """
        model = AutoModel.from_pretrained(self.reranker_model,
                                        device_map='auto', 
                                        torch_dtype=torch.float16,
                                        cache_dir=self.cache_dir,
                                        trust_remote_code=True)
        return model

    def _load_tokenizer(self):
        """
        Load the tokenizer (implemented from the base class).
        """
        return AutoTokenizer.from_pretrained(self.reranker_model, device_map='auto', trust_remote_code=True)

    def _encode_texts(self, texts: List[str], instruction: str = "", max_length: int = 512, batch_size: int = 16) -> torch.Tensor:
        """
        Encode texts into embeddings using last token pooling.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                batch_dict = {k: v.to(f"cuda:{0}") for k, v in batch_dict.items()}  # Move to primary GPU

                # Forward pass
                try:
                    outputs = self.model(**batch_dict)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("Reducing batch size due to OOM")
                        return self._encode_texts(texts, instruction, max_length, batch_size // 2)  # Retry with reduced batch size
                    raise

                # Pool and normalize embeddings
                pooled = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                embeddings.append(normalized.cpu())  # Move to CPU to free GPU memory

                # Clear GPU cache
                torch.cuda.empty_cache()

        return torch.cat(embeddings, dim=0)
