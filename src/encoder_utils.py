import torch
from typing import List
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Constructs an instruction-based query for multilingual reranking.
    """
    return f"Instruct: {task_description}\nQuery: {query}"

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average pools the hidden states based on the attention mask.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9) # prevent division by zero

def gritlm_instruction(instruction: str) -> str:
    """
    Formats the instruction for GritLM input.
    """
    return f"<|user|>\n{instruction}\n<|embed|>\n" if instruction else "<|embed|>\n"

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform last token pooling on the model's hidden states.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def encode(self, combined_texts: List[str], max_length: int = 512, batch_size: int = 32) -> torch.Tensor:
    """
    Encodes a list of combined texts (queries + documents) into embeddings using the model and average pooling.
    """
    tokenized_texts = self.tokenizer(combined_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    dataset = TensorDataset(tokenized_texts['input_ids'], tokenized_texts['attention_mask'])

    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset)
    else:
        sampler = None  # Use default sampler in single-GPU mode

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_embeddings = average_pool(outputs.last_hidden_state, attention_mask)
            embeddings.append(pooled_embeddings.cpu())
    return torch.cat(embeddings, dim=0)
