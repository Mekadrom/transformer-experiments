import torch
import torch.nn as nn
from typing import List, Tuple

class MultiGPUCausalWrapper:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, gpu_ids: List[int], batches_per_step: int, sync_steps: int):
        pass

class MultiGPUTranslationWrapper:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, gpu_ids: List[int], batches_per_step: int, sync_steps: int):
        """
        Wrapper for distributed training of translation models across multiple GPUs.
        
        Args:
            model: The transformer model to distribute
            optimizer: The optimizer being used
            gpu_ids: List of GPU IDs to use
            batches_per_step: Number of batches to accumulate gradients over before performing an optimization step
            sync_steps: Number of steps to wait before syncing across devices
        """
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)
        self.batches_per_step = batches_per_step
        self.sync_steps = sync_steps
        
        # Create model copies for each GPU
        self.models = []
        for gpu_id in gpu_ids:
            device = f'cuda:{gpu_id}'
            model_copy = type(model)(*model.args, **model.kwargs).to(device)
            model_copy.load_state_dict(model.state_dict())
            self.models.append(model_copy)
            
        self.optimizer = optimizer
        self.step = 0
        self.current_accumulation_step = 0
        
    def split_batch(self, 
                   src_seqs: torch.Tensor,
                   tgt_seqs: torch.Tensor,
                   src_key_padding_mask: torch.Tensor,
                   tgt_key_padding_mask: torch.Tensor) -> List[Tuple[torch.Tensor, ...]]:
        """
        Split a batch across available GPUs.
        """
        batch_size = src_seqs.size(0)
        split_size = batch_size // self.num_gpus
        
        # Handle cases where batch_size < num_gpus
        if split_size == 0:
            split_size = 1
            actual_gpus = min(batch_size, self.num_gpus)
        else:
            actual_gpus = self.num_gpus
            
        splits = []
        for i in range(actual_gpus):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < actual_gpus - 1 else batch_size
            
            device = f'cuda:{self.gpu_ids[i]}'
            splits.append((
                src_seqs[start_idx:end_idx].to(device),
                tgt_seqs[start_idx:end_idx].to(device),
                src_key_padding_mask[start_idx:end_idx].to(device),
                tgt_key_padding_mask[start_idx:end_idx].to(device)
            ))
            
        return splits

    def forward_pass(self, 
                     src_seqs: torch.Tensor,
                     tgt_seqs: torch.Tensor,
                     src_key_padding_mask: torch.Tensor,
                     tgt_key_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        splits = self.split_batch(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)
        
        total_translation_loss = 0
        total_moe_loss = 0
        total_encoder_moe_loss = 0
        total_decoder_moe_loss = 0
        
        # Process each split on its designated GPU
        for i, (input_ids, labels, src_key_padding_mask, tgt_key_padding_mask) in enumerate(splits):
            outputs = self.models[i](
                input_ids=input_ids, labels=labels, attention_mask=src_key_padding_mask, decoder_attention_mask=tgt_key_padding_mask
            )

            loss = outputs.loss.to(input_ids.device)
            translation_loss = outputs.prediction_loss.to(input_ids.device)
            moe_loss = outputs.moe_loss.to(input_ids.device)
            enc_moe_vars = outputs.encoder_gating_variances.to(input_ids.device)
            dec_moe_vars = outputs.decoder_gating_variances.to(input_ids.device)
            
            # Scale losses
            scaled_loss = (loss) / (self.batches_per_step * self.num_gpus)
            scaled_loss.backward()
            
            # Accumulate losses for reporting
            total_translation_loss += translation_loss.item()
            total_moe_loss += moe_loss.item()
            total_encoder_moe_loss += enc_moe_vars.item()
            total_decoder_moe_loss += dec_moe_vars.item()
        
        self.step += 1
        self.current_accumulation_step += 1

        if self.step >= self.batches_per_step:
            self.step = 0
            if hasattr(self, 'clip_grad_norm') and self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.models[0].parameters(), self.clip_grad_norm)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.current_accumulation_step >= self.sync_steps:
            self.current_accumulation_step = 0
            self._sync_models()
        
        return (
            torch.tensor(total_translation_loss),
            torch.tensor(total_translation_loss + total_moe_loss),
            torch.tensor(total_moe_loss),
            torch.tensor(total_encoder_moe_loss),
            torch.tensor(total_decoder_moe_loss)
        )
    
    def _sync_models(self):
        """
        Synchronize model parameters across all GPUs using the first GPU's model as source of truth.
        """
        main_state = self.models[0].state_dict()
        for model in self.models[1:]:
            model.load_state_dict(main_state)
            
    def save_checkpoint(self, step: int, prefix: str = ""):
        """
        Save a checkpoint using the first GPU's model (since they're synced).
        """
        return {
            'step': step,
            'model': self.models[0],
            'optimizer': self.optimizer
        }
