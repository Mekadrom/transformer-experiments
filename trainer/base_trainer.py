from megatransformer import megatransformer, multihead_attn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Tuple

import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import time
import utils

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class BaseTrainer:
    def __init__(self, args):
        self.args = args

        self.run_name = args.run_name
        self.d_model = args.d_model
        self.warmup_steps = args.warmup_steps
        self.print_frequency = args.print_frequency
        if hasattr(args, 'batches_per_step'):
            self.batches_per_step: int = args.batches_per_step

        self.run_dir = os.path.join('runs', self.run_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        with open(os.path.join(self.run_dir, "args.txt"), 'w') as f:
            f.write(str(args))

        self.summary_writer = SummaryWriter(log_dir=self.run_dir)

        self.tokenizer_run_dir = os.path.join('runs', args.tokenizer_run_name)

        self.src_tokenizer, self.tgt_tokenizer = self.load_tokenizers(identifier=self.tokenizer_run_dir)

        self.model, self.optimizer = self.load_model_and_optimizer(self.run_dir, checkpoint_model_name=args.model_checkpoint if hasattr(args, 'model_checkpoint') else 'transformer_checkpoint.pth.tar')

        if hasattr(self.model, 'encoder'):
            if isinstance(self.model.encoder.embed_tokens, nn.Embedding):
                print(self.model.encoder.embed_tokens.weight.device)
            else:
                print(self.model.encoder.embed_tokens.embedding.weight.device)
        
        decoder = self.model.decoder if hasattr(self.model, 'decoder') else self.model
        if isinstance(decoder.embed_tokens, nn.Embedding):
            print(decoder.embed_tokens.weight.device)
        else:
            print(decoder.embed_tokens.embedding.weight.device)

        self.precision = torch.float32
        if hasattr(args, 'precision') and args.precision is not None:
            if args.precision == 'bf16':
                self.precision = torch.bfloat16
            elif args.precision == 'fp16':
                self.precision = torch.float16

        self.model = self.model.to(self.precision)

        if bool(args.compile_model):
            print("Using compiled model")
            torch._dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)
            self.compiled_model = torch.compile(self.model)
        else:
            self.compiled_model = self.model

        if bool(args.early_stop):
            self.early_stopping = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
        else:
            self.early_stopping = None

        utils.print_model(self.model)
        print(f"Optimizer: {self.optimizer}")

        if args.save_initial_checkpoint:
            utils.save_checkpoint(-1, self.model, self.optimizer, f"runs/{args.run_name}/")

        self.target_sequence_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda source_sequences, target_sequences: (target_sequences)

        self.step = 0
        self.start_step = int(self.args.start_step)
        self.n_steps = int(self.args.n_steps)

        print(f"Loading data...")
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

        if hasattr(self, 'batch_size'):
            print(f"batch_size: {self.batch_size}")
        print(f"n_steps: {self.n_steps}")

        self.lr_scheduler = None
        if hasattr(args, 'lr_scheduler') and args.lr_scheduler is not None:
            if args.lr_scheduler == 'cosine':
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.n_steps)
            elif args.lr_scheduler == 'cosine_warmup':
                self.lr_scheduler = utils.create_warmup_cosine_scheduler(self.optimizer, self.warmup_steps, self.n_steps, min_lr=0.)
            elif args.lr_scheduler == 'noam':
                pass
            elif args.lr_scheduler == 'constant':
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
            else:
                raise ValueError(f"Invalid lr_scheduler: {args.lr_scheduler}")

    def load_tokenizers(self, identifier):
        raise NotImplementedError

    def load_model_and_optimizer(self, run_dir, checkpoint_model_name='transformer_checkpoint.pth.tar') -> Tuple[nn.Module, torch.optim.Optimizer]:
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError

    def train(self, model_name_prefix=''):
        print(f"Training for {self.n_steps} steps...")
        while self.step < self.n_steps:
            self.training(self.compiled_model)

            val_loss_avg = self.validation(self.model)

            utils.save_checkpoint(self.step, self.model, self.optimizer, prefix=f"{self.run_dir}/{model_name_prefix}")

            if self.early_stopping is not None:
                if self.early_stopping(val_loss_avg):
                    print("Early stopping")
                    utils.average_checkpoints(self.step, self.optimizer, self.run_dir, self.args.early_stop_checkpoint_window, model_name_prefix='step')

                    print(f"Training complete. Evaluating one last time...")
                    self.validation(self.model)
                    break

        print(f"Training complete. Averaging checkpoints...")
        utils.average_checkpoints(self.step, self.optimizer, self.run_dir, model_name_prefix='step')

        print(f"Training complete. Evaluating one last time...")
        self.validation(self.model)

    def training(self, model: megatransformer.MegaTransformer):
        raise NotImplementedError
    
    def validation(self, model: megatransformer.MegaTransformer):
        raise NotImplementedError
    
    def evaluate(self, src, tgt):
        raise NotImplementedError

    def viz_model(self, step, model, src, **kwargs):
        raise NotImplementedError
