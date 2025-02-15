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
        self.n_steps = args.n_steps
        self.warmup_steps = args.warmup_steps
        self.encoder_device = args.encoder_device
        self.decoder_device = args.decoder_device
        self.print_frequency = args.print_frequency
        if hasattr(args, 'batches_per_step'):
            self.batches_per_step: int = args.batches_per_step

        self.run_dir = os.path.join('runs', self.run_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.summary_writer = SummaryWriter(log_dir=self.run_dir)

        self.tokenizer_run_dir = os.path.join('runs', args.tokenizer_run_name)

        self.src_tokenizer, self.tgt_tokenizer = self.load_tokenizers(identifier=self.tokenizer_run_dir)

        self.model, self.optimizer = self.load_model_and_optimizer(self.run_dir)

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

        self.sacrebleu_epochs = []
        self.target_sequence_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda source_sequences, target_sequences: (target_sequences)

    def load_tokenizers(self, identifier):
        raise NotImplementedError

    def load_model_and_optimizer(self, run_dir, checkpoint_model_name='transformer_checkpoint.pth.tar') -> Tuple[nn.Module, torch.optim.Optimizer]:
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError

    def train(self, model_name_prefix=''):
        self.steps = 0
        self.start_epoch = self.args.start_epoch

        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        if hasattr(self.train_loader, 'create_batches'):
            self.epochs = (self.args.n_steps // ((self.train_loader.n_batches * len(self.train_loader.src_file_paths)) // self.args.batches_per_step)) + 1
        else:
            if hasattr(self.args, 'n_epochs') and self.args.n_epochs is not None and int(self.args.n_epochs) > 0:
                self.epochs = int(self.args.n_epochs)
            else:
                raise ValueError("n_epochs must be set to a positive integer if using non SequenceLoader dataloader")
        if hasattr(self.args, 'n_epochs') and self.args.n_epochs is not None and int(self.args.n_epochs) > 0:
            self.epochs = int(self.args.n_epochs)

        print(f"Training for {self.epochs} epochs...")
        for epoch in range(self.start_epoch, self.epochs):
            if hasattr(self.train_loader, 'create_batches'):
                self.train_loader.create_batches()
            self.train_epoch(self.compiled_model, epoch=epoch)

            if self.val_loader is not None:
                if hasattr(self.val_loader, 'create_batches'):
                    self.val_loader.create_batches()
                val_loss_avg = self.validate_epoch(self.model)

            utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/{model_name_prefix}")

            if self.early_stopping is not None:
                if self.early_stopping(val_loss_avg):
                    print("Early stopping")
                    utils.average_checkpoints(self.epochs, self.optimizer, self.run_dir, self.args.early_stop_checkpoint_window, model_name_prefix='step')

                    print(f"Training complete. Evaluating one last time...")
                    if hasattr(self.val_loader, 'create_batches'):
                        self.val_loader.create_batches()
                    self.validate_epoch(self.model)
                    break

        print(f"Training complete. Averaging checkpoints...")
        utils.average_checkpoints(self.epochs, self.optimizer, self.run_dir, model_name_prefix='step')

        print(f"Training complete. Evaluating one last time...")
        if hasattr(self.val_loader, 'create_batches'):
            self.val_loader.create_batches()
        self.validate_epoch(self.model)

    def train_epoch(self, model: megatransformer.MegaTransformer, epoch):
        raise NotImplementedError
    
    def validate_epoch(self, model: megatransformer.MegaTransformer):
        raise NotImplementedError
    
    def evaluate(self, src, tgt):
        raise NotImplementedError
    
    def viz_attn_weight(self, stack_name, layer_num, n_head, activation_weights, attendee_tokens, attending_tokens):
        fig, ax = plt.subplots(figsize=(10, 10))
        s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
        s.set(xlabel="Attending Tokens", ylabel="Attended Tokens", title=f"{stack_name}-Attn Layer {layer_num} Head {n_head} Weights")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf
    
    def viz_attn_weights(self, attn: multihead_attn.MultiHeadAttention, attn_residual, layer_num, src_seq, src_seq_len, key_padding_mask, tgt_seq, tgt_seq_len, src_tokens, tgt_tokens, summary_writer, step):
        decoder_or_encoder = 'decoder' if attn.in_decoder else 'encoder'
        self_or_cross = 'self' if attn.self_attn else 'cross'
        stack_name = f"{decoder_or_encoder}-{self_or_cross}"

        residual, attention_weights = attn(tgt_seq, src_seq, src_seq, key_padding_mask, return_attn_values=True)
        tgt_seq = attn_residual(tgt_seq, residual)

        for a, attention_weight_grid in enumerate(attention_weights):
            attention_weight_grid = attention_weight_grid.cpu().contiguous()
            for head_num in range(attention_weight_grid.size(1)):
                image_data = self.viz_attn_weight(stack_name, layer_num, head_num, attention_weight_grid[:, head_num, :tgt_seq_len, :src_seq_len].transpose(-2, -1).squeeze(0).to(torch.float32).cpu().detach().numpy(), tgt_tokens, src_tokens)
                summary_writer.add_image(f"{decoder_or_encoder}/viz/layer_{layer_num}/segment_{a}/head_{head_num}/{self_or_cross}-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

        return tgt_seq

    def viz_encoder_layer(self, encoder_layer: megatransformer.EncoderLayer, src_seq, src_seq_len, src_key_padding_mask, src_tokens, summary_writer, step, layer_num):
        print(f"Visualizing encoder layer {layer_num}...")
        residual = self.viz_attn_weights(encoder_layer.self_attn, encoder_layer.self_attn_residual, layer_num, src_seq, src_seq_len, src_key_padding_mask, src_seq, src_seq_len, src_tokens, src_tokens, summary_writer, step)
        fcn_out, _ = encoder_layer.ffn(residual)
        return encoder_layer.ffn_residual(residual, fcn_out)

    def viz_encoder_layers(self, encoder_layers: list[megatransformer.EncoderLayer], seq, seq_len, key_padding_mask, tokens, summary_writer, step):
        for e, encoder_layer in enumerate(encoder_layers):
            seq = self.viz_encoder_layer(encoder_layer, seq, seq_len, key_padding_mask, tokens, summary_writer, step, e)
        return seq

    def viz_encoder(self, encoder: megatransformer.Encoder, seq, seq_len, key_padding_mask, tokens, summary_writer, step):
        print("Visualizing encoder...")
        if (hasattr(encoder, 'embed_tokens') and encoder.embed_tokens is not None) or (hasattr(encoder, 'embedding') and encoder.embed_tokens is not None):
            seq = encoder.apply_embedding_transformation(seq)
        seq = encoder.apply_positional_embedding(seq)
        seq = self.viz_encoder_layers(encoder.encoder_layers, seq, seq_len, key_padding_mask, tokens, summary_writer, step)
        return encoder.post_encoder_norm(seq).to(self.args.decoder_device)

    def viz_decoder_layer(self, decoder_layer: megatransformer.DecoderLayer, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step, layer_num):
        print(f"Visualizing decoder layer {layer_num}...")
        tgt_seq = self.viz_attn_weights(decoder_layer.self_attn, decoder_layer.self_attn_residual, layer_num, tgt_seq, tgt_seq_len, tgt_key_padding_mask, tgt_seq, tgt_seq_len, tgt_tokens, tgt_tokens, summary_writer, step)
        if decoder_layer.cross_attn is not None:
            residual = self.viz_attn_weights(decoder_layer.cross_attn, decoder_layer.cross_attn_residual, layer_num, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, src_tokens, tgt_tokens, summary_writer, step)
        else:
            residual = tgt_seq
        fcn_out, _ = decoder_layer.ffn(residual)
        return decoder_layer.ffn_residual(residual, fcn_out)

    def viz_decoder_layers(self, decoder_layers, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step):
        for d, decoder_layer in enumerate(decoder_layers):
            tgt_seq = self.viz_decoder_layer(decoder_layer, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step, d)
        return tgt_seq

    def viz_decoder(self, decoder: megatransformer.Decoder, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step):
        print("Visualizing decoder...")
        if (hasattr(decoder, 'embed_tokens') and decoder.embed_tokens is not None) or (hasattr(decoder, 'embedding') and decoder.embed_tokens is not None):
            tgt_seq = decoder.apply_embedding_transformation(tgt_seq)
        tgt_seq = decoder.apply_positional_embedding(tgt_seq.to(self.args.decoder_device))
        return self.viz_decoder_layers(decoder.decoder_layers, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step)

    def viz_model(self, step, model, src, **kwargs):
        raise NotImplementedError
