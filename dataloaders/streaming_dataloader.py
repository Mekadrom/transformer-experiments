from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Iterator

import logging
import random
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingCausalLMDataset(IterableDataset):
    """
    A PyTorch IterableDataset for efficient streaming of very large text datasets.
    Designed to handle datasets like FineWeb that are too large to fit in memory.
    
    This dataset:
    1. Streams data without loading the entire dataset into memory
    2. Tokenizes on-the-fly in chunks
    3. Creates overlapping sequences of specified length for causal language modeling
    4. Handles multiple workers efficiently
    """
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str,
        dataset_config_name: Optional[str] = None,
        split: str = "train",
        sequence_length: int = 1024,
        text_column: str = "text",
        chunk_size: int = 1000,  # Number of examples to process at once
        buffer_size: int = 10000,  # Buffer size for shuffling
        cache_dir: Optional[str] = None,
        streaming: bool = True,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            dataset_name: Name of the Hugging Face dataset
            dataset_config_name: Configuration name for the dataset
            split: Dataset split to use
            sequence_length: The length of sequences to return
            text_column: Column name containing the text data
            chunk_size: Number of examples to process at once
            buffer_size: Size of buffer for shuffling streaming data
            cache_dir: Directory to cache the dataset
            streaming: Whether to stream the dataset or load it entirely
            shuffle: Whether to shuffle the data
            max_samples: Maximum number of samples to process (for debugging)
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.sequence_length = sequence_length
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.cache_dir = cache_dir
        self.streaming = streaming
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.seed = seed
        
        # Set padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset in streaming mode
        logger.info(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir
        )
        
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)
        
        if self.max_samples is not None:
            self.dataset = self.dataset.take(self.max_samples)
    
    def state_dict(self) -> Dict:
        """Return a dictionary with the current state for later resumption."""
        return {"seed": self.seed}
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load a state dictionary to resume from a saved state."""
        self.seed = state_dict["seed"]
        if self.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)
    
    def process_chunk(self, chunk: List[Dict]) -> Tuple[List[int], List[bool]]:
        """
        Process a chunk of examples.
        
        Args:
            chunk: A list of dataset examples
            
        Returns:
            Tuple of (all_token_ids, all_is_doc_start)
        """
        all_token_ids = []
        all_is_doc_start = []  # Track document boundaries
        
        for example in chunk:
            text = example[self.text_column]
            if isinstance(text, str) and text.strip():  # Skip empty texts
                # Mark document start
                if all_token_ids:  # Not the first document overall
                    all_is_doc_start.append(True)
                    
                # Tokenize and add to our buffer
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                all_token_ids.extend(token_ids)
                
                # Mark all other tokens as not document start
                all_is_doc_start.extend([False] * (len(token_ids) - (1 if all_token_ids else 0)))
        
        return all_token_ids, all_is_doc_start
    
    def create_samples(
        self, 
        token_ids: List[int], 
        is_doc_start: List[bool],
        respect_document_boundaries: bool = True
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Create samples from a chunk of tokenized text.
        
        Args:
            token_ids: List of token IDs
            is_doc_start: List of booleans indicating document starts
            respect_document_boundaries: Whether to avoid creating sequences that cross document boundaries
            
        Yields:
            Dictionary containing input_ids and attention_mask tensors
        """
        i = 0
        while i + 1 < len(token_ids):  # Need at least 2 tokens for input and target
            # If we're respecting document boundaries and this is the start of a document,
            # but we're not at the beginning, skip to the start of this document
            if respect_document_boundaries and i > 0 and is_doc_start[i]:
                i += 1
                continue
                
            # Calculate end index for this sequence
            end_idx = i + self.sequence_length
            
            # Check if sequence would cross a document boundary
            if respect_document_boundaries and end_idx < len(is_doc_start):
                # Find the next document start within our potential sequence
                next_doc_start = is_doc_start[i:end_idx].index(True) + i if True in is_doc_start[i:end_idx] else None
                
                if next_doc_start is not None:
                    # Adjust end_idx to not cross document boundary
                    end_idx = next_doc_start
            
            # Make sure we have enough tokens for a meaningful sequence
            if end_idx - i < 2:  # Need at least 2 tokens
                i = end_idx
                continue
                
            # Cap to available tokens
            end_idx = min(end_idx, len(token_ids))
            
            # Get the sequence for input and label
            seq = token_ids[i:end_idx]
            
            # For causal LM: input is all tokens except the last, target is all tokens except the first
            input_seq = seq[:-1]
            target_seq = seq[1:]
            
            # Handle case where we couldn't get enough tokens
            if len(input_seq) < self.sequence_length - 1:
                # If too short, pad to full length
                pad_length = self.sequence_length - 1 - len(input_seq)
                input_ids = input_seq + [self.tokenizer.pad_token_id] * pad_length
                labels = target_seq + [self.tokenizer.pad_token_id] * pad_length
                attention_mask = [0] * len(input_seq) + [1] * pad_length
            else:
                input_ids = input_seq
                labels = target_seq
                attention_mask = [0] * len(input_ids)
            
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
            
            # Move to next starting position - can overlap or stride
            # For full coverage without overlap: i = end_idx - 1
            # For some overlap: i += self.sequence_length // 2
            # For one token stride (most overlap): i += 1
            i += self.sequence_length // 4  # Using 75% overlap by default
    
    def get_stream(self, worker_info=None) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Get a stream of samples.
        
        Args:
            worker_info: Information about the worker in multi-process data loading
            
        Yields:
            Dictionary containing input_ids, attention_mask, and labels tensors
        """
        # Initialize buffer for carrying over tokens between chunks
        carry_over_tokens = []
        carry_over_is_doc_start = []
        
        # Handle multi-process data loading
        if worker_info is not None:
            # Split the dataset among workers
            self.dataset = self.dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id
            )
            # Use different seed for each worker
            if self.shuffle:
                self.dataset = self.dataset.shuffle(
                    seed=self.seed + worker_info.id,
                    buffer_size=self.buffer_size
                )
        
        # Process the dataset in chunks
        chunk = []
        for example in self.dataset:
            chunk.append(example)
            
            if len(chunk) >= self.chunk_size:
                # Process the current chunk
                token_ids, is_doc_start = self.process_chunk(chunk)
                
                # Combine with carry-over from previous chunk
                if carry_over_tokens:
                    token_ids = carry_over_tokens + token_ids
                    is_doc_start = carry_over_is_doc_start + is_doc_start
                
                # Create and yield samples
                yield from self.create_samples(token_ids, is_doc_start)
                
                # Save the last sequence_length-1 tokens for the next chunk
                # This ensures we can create samples that cross chunk boundaries
                carry_over_tokens = token_ids[-(self.sequence_length-1):]
                carry_over_is_doc_start = is_doc_start[-(self.sequence_length-1):]
                
                # Reset chunk
                chunk = []
        
        # Process any remaining examples
        if chunk:
            token_ids, is_doc_start = self.process_chunk(chunk)
            
            if carry_over_tokens:
                token_ids = carry_over_tokens + token_ids
                is_doc_start = carry_over_is_doc_start + is_doc_start
            
            yield from self.create_samples(token_ids, is_doc_start)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Create an iterator of samples.
        
        Returns:
            Iterator of dictionaries containing input_ids, attention_mask, and labels tensors
        """
        worker_info = torch.utils.data.get_worker_info()
        return self.get_stream(worker_info)


def create_causal_lm_dataloaders(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_config_name: Optional[str] = None,
    batch_size: int = 8,
    sequence_length: int = 1024,
    text_column: str = "text",
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
    train_shuffle: bool = True,
    eval_shuffle: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    streaming: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 1000,
    buffer_size: int = 10000,
    max_samples: Optional[Dict[str, int]] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and test sets.
    
    Args:
        dataset_path: Path to the Hugging Face dataset
        dataset_name: Name of the Hugging Face dataset
        tokenizer: The tokenizer to use
        dataset_config_name: Configuration name for the dataset
        batch_size: Batch size for training and evaluation
        sequence_length: Length of sequences to use
        text_column: Column name containing the text data
        train_split: Name of the training split
        val_split: Name of the validation split
        test_split: Name of the test split
        train_shuffle: Whether to shuffle the training data
        eval_shuffle: Whether to shuffle the evaluation data
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of batches to prefetch per worker
        pin_memory: Whether to pin memory in data loading
        seed: Random seed
        streaming: Whether to stream the dataset
        cache_dir: Directory to cache the dataset
        chunk_size: Number of examples to process at once
        buffer_size: Size of buffer for shuffling
        respect_document_boundaries: Whether to avoid creating sequences that cross document boundaries
        max_samples: Maximum number of samples to use from each split (for debugging)
        
    Returns:
        Dictionary of DataLoaders for train, validation, and test sets
    """
    logger.info(f"Creating dataloaders for {dataset_name}")
    
    # Set default max_samples
    if max_samples is None:
        max_samples = {"train": None, "validation": None, "test": None}
    
    # Determine available splits
    available_splits = []
    try:
        info = load_dataset(dataset_name, dataset_config_name, split="train", streaming=True)
        available_splits.append("train")
    except (ValueError, FileNotFoundError):
        logger.warning(f"Train split not available for {dataset_name}")
    
    try:
        info = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
        available_splits.append("validation")
    except (ValueError, FileNotFoundError):
        logger.warning(f"Validation split not available for {dataset_name}")
    
    try:
        info = load_dataset(dataset_name, dataset_config_name, split="test", streaming=True)
        available_splits.append("test")
    except (ValueError, FileNotFoundError):
        logger.warning(f"Test split not available for {dataset_name}")
    
    logger.info(f"Available splits: {available_splits}")
    
    dataloaders = {}
    
    # Create train dataloader
    if "train" in available_splits:
        train_dataset = StreamingCausalLMDataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=train_split,
            sequence_length=sequence_length,
            text_column=text_column,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
            cache_dir=cache_dir,
            streaming=streaming,
            shuffle=train_shuffle,
            max_samples=max_samples["train"],
            seed=seed
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory
        )
        
        dataloaders["train"] = train_dataloader
    
    # Create validation dataloader
    if "validation" in available_splits:
        val_dataset = StreamingCausalLMDataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=val_split,
            sequence_length=sequence_length,
            text_column=text_column,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
            cache_dir=cache_dir,
            streaming=streaming,
            shuffle=eval_shuffle,
            max_samples=max_samples["validation"],
            seed=seed
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=max(1, num_workers // 2),
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory
        )
        
        dataloaders["validation"] = val_dataloader
    
    # Create test dataloader
    if "test" in available_splits:
        test_dataset = StreamingCausalLMDataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=test_split,
            sequence_length=sequence_length,
            text_column=text_column,
            chunk_size=chunk_size,
            buffer_size=buffer_size,
            cache_dir=cache_dir,
            streaming=streaming,
            shuffle=eval_shuffle,
            max_samples=max_samples["test"],
            seed=seed
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=max(1, num_workers // 2),
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory
        )
        
        dataloaders["test"] = test_dataloader
    
    return dataloaders


# Example usage
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Example with WikiText-103 for testing
    dataloaders = create_causal_lm_dataloader(
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config_name="wikitext-103-raw-v1",
        batch_size=4,
        sequence_length=512,
        num_workers=2,
        max_samples={"train": 1000, "validation": 100, "test": 100}  # Limit for testing
    )
    
    # Test train dataloader
    if "train" in dataloaders:
        train_dataloader = dataloaders["train"]
        logger.info("Testing train dataloader...")
        batch = next(iter(train_dataloader))
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input shape: {batch['input_ids'].shape}")
        logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")

        logger.info(f"Example input: {tokenizer.decode(random.choice(batch['input_ids']), skip_special_tokens=True)}")
        logger.info(f"Example target: {tokenizer.decode(random.choice(batch['labels']), skip_special_tokens=True)}")
        logger.info(f"Example attention mask: {random.choice(batch['attention_mask'])}")
    
    # Test validation dataloader
    if "validation" in dataloaders:
        val_dataloader = dataloaders["validation"]
        logger.info("Testing validation dataloader...")
        batch = next(iter(val_dataloader))
        logger.info(f"Input shape: {batch['input_ids'].shape}")

    # Example with a much larger dataset like FineWeb
    # This would typically be run on a cluster with significant compute resources
    """
    dataloaders = get_streaming_dataloaders(
        dataset_name="allenai/fineweb",
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        batch_size=16,
        sequence_length=2048,
        num_workers=8,
        streaming=True,
        cache_dir="/path/to/cache",
        chunk_size=5000,
        buffer_size=50000
    )
    """
