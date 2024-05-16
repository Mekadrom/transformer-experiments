from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, args, base_run_dir):
        self.args = args

        self.run_name = args.run_name
        self.d_model = args.d_model
        self.n_steps = args.n_steps
        self.warmup_steps = args.warmup_steps
        self.device = args.device
        self.print_frequency = args.print_frequency
        if hasattr(args, 'batches_per_step'):
            self.batches_per_step = args.batches_per_step

        self.run_dir = os.path.join(base_run_dir, 'runs', self.run_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.summary_writer = SummaryWriter(log_dir=self.run_dir)

        self.bpe_run_dir = os.path.join(base_run_dir, 'runs', args.tokenizer_run_name)

        self.src_bpe_model, self.tgt_bpe_model = utils.load_tokenizers(self.bpe_run_dir)

        if 'train_vae' in args and args.train_vae:
            self.tgt_bpe_model = self.src_bpe_model

        self.model, self.optimizer = self.load_model_and_optimizer()
        self.model = self.model.to(args.device)

        if args.torch_compile_model:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)
            self.compiled_model = torch.compile(self.model)
        else:
            self.compiled_model = self.model

        self.criterion = self.get_criteria()

        if args.early_stop:
            self.early_stopping = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
        else:
            self.early_stopping = None

        utils.print_model(self.model)
        print(f"Optimizer: {self.optimizer}")
        print(f"Criterion: {self.criterion}")

        if args.save_initial_checkpoint:
            utils.save_checkpoint(-1, self.model, self.optimizer, f"runs/{args.run_name}/")

        self.sacrebleu_epochs = []
        self.target_sequence_transform = lambda source_sequences, source_sequence_lengths, target_sequences, target_sequence_lengths: (target_sequences, target_sequence_lengths)

    def get_criteria(self):
        raise NotImplementedError

    def load_model_and_optimizer(self):
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError

    def train(self, model_name_prefix=''):
        self.train_loader, self.val_loader, self.test_loader = self.load_data()

        self.steps = 0
        self.start_epoch = self.args.start_epoch
        self.epochs = (self.args.n_steps // (self.train_loader.n_batches // self.args.batches_per_step)) + 1

        print(f"Training for {self.epochs} epochs...")
        start = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            self.steps = (epoch * self.train_loader.n_batches // self.batches_per_step)

            self.train_loader.create_batches()
            self.train_epoch(self.compiled_model, epoch=epoch)

            self.val_loader.create_batches()
            val_loss_avg = self.validate_epoch(self.model)

            utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/{model_name_prefix}")

            if self.early_stopping is not None:
                if self.early_stopping(val_loss_avg):
                    print("Early stopping")
                    utils.average_checkpoints(self.epochs, self.optimizer, self.run_dir, self.args.early_stop_num_latest_checkpoints_to_avg, model_name_prefix='step')

                    print(f"Training complete. Evaluating one last time...")
                    self.val_loader.create_batches()
                    self.validate_epoch(self.model)
                    break

        time_taken = time.time() - start

        # recalculate steps to make sure validation data is updated with correct steps
        self.steps = (self.epochs * self.train_loader.n_batches // self.batches_per_step)

        print(f"Training complete. Averaging checkpoints...")
        utils.average_checkpoints(self.epochs, self.optimizer, self.run_dir, model_name_prefix='step')

        print(f"Training complete. Evaluating one last time...")
        self.val_loader.create_batches()
        self.validate_epoch(self.model)

        print(f"Training complete. Scoring with sacrebleu...")
        return utils.sacrebleu_evaluate(self.args, self.run_dir, self.src_bpe_model, self.tgt_bpe_model, self.model, device=self.device, sacrebleu_in_python=True, test_loader=self.test_loader).score, time_taken, utils.count_parameters(self.model)

    def train_epoch(self, model, epoch):
        raise NotImplementedError
    
    def validate_epoch(self, model):
        raise NotImplementedError
    
    def evaluate(self, src, tgt):
        raise NotImplementedError
    
    def viz_model(self, step, model, src, tgt=None):
        raise NotImplementedError
    
    def viz_attn_weights(self, stack_name, layer_num, n_head, activation_weights, attendee_tokens, attending_tokens):
        fig, ax = plt.subplots(figsize=(10, 10))
        s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
        s.set(xlabel="Attending Tokens", ylabel="Attended Tokens", title=f"{stack_name}-Attn Layer {layer_num} Head {n_head} Weights")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf