from dataloader import SequenceLoader
from criteria.labelsmooth import LabelSmoothedCE
from grokfast import gradfilter_ma, gradfilter_ema
from model_provider import TranslationTransformerModelProvider
from modules.transformer import Transformer
from prettytable import PrettyTable
from tqdm import tqdm
from typing import List, Optional

import avg_meter
import base_trainer
import matplotlib.pyplot as plt
import os
import time
import torch
import utils

class TranslationTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(TranslationTrainer, self).__init__(args)

        self.grads = None

    def load_model_and_optimizer(self):
        print('Initializing model...')

        if os.path.exists(os.path.join(self.run_dir, 'transformer_checkpoint.pth.tar')):
            checkpoint = torch.load(os.path.join(self.run_dir, 'transformer_checkpoint.pth.tar'))
            if hasattr(self.args, 'start_epoch') and self.args.start_epoch == 0:
                self.args.start_epoch = checkpoint['epoch'] + 1
                print('\nLoaded checkpoint from epoch %d.\n' % self.args.start_epoch)

            model = TranslationTransformerModelProvider().provide_transformer(self.args, self.src_bpe_model.vocab_size(), self.tgt_bpe_model.vocab_size(), tie_embeddings=self.tgt_bpe_model==self.src_bpe_model)

            model.load_state_dict(checkpoint['model'].state_dict())

            if 'optimizer' in checkpoint:
                optimizer = checkpoint['optimizer']
            else:
                optimizer = None
        else:
            print("Starting from scratch...")
            model = TranslationTransformerModelProvider().provide_transformer(self.args, self.src_bpe_model.vocab_size(), self.tgt_bpe_model.vocab_size(), tie_embeddings=self.tgt_bpe_model==self.src_bpe_model)

            optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=self.args.lr, betas=[self.args.beta1, self.args.beta2], eps=self.args.epsilon)

        return model, optimizer

    def get_criteria(self):
        return LabelSmoothedCE(args=self.args, eps=self.args.label_smoothing)

    def load_data(self):
        return utils.load_translation_data(self.args.tokens_in_batch, self.bpe_run_dir, self.src_bpe_model, self.tgt_bpe_model, pad_to_length=self.args.maxlen if self.args.use_infinite_attention else None)
    
    def train(self, model_name_prefix=''):
        if self.args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        super().train()

    def train_epoch(self, model: Transformer, epoch):
        # training mode enables dropout
        model.train()

        data_time = avg_meter.AverageMeter()
        step_time = avg_meter.AverageMeter()
        total_losses = avg_meter.AverageMeter()
        translation_losses = avg_meter.AverageMeter()
        encoder_moe_gating_variance_losses = avg_meter.AverageMeter()
        decoder_moe_gating_variance_losses = avg_meter.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths) in enumerate(self.train_loader):
            src_seqs = src_seqs.to(self.encoder_device)
            tgt_seqs = tgt_seqs.to(self.encoder_device)
            src_seq_lengths = src_seq_lengths.to(self.encoder_device)
            tgt_seq_lengths = tgt_seq_lengths.to(self.decoder_device)

            tgt_seqs, tgt_seq_lengths = self.target_sequence_transform(src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths)

            src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
            tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

            data_time.update(time.time() - start_data_time)

            predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            del src_seqs, src_seq_lengths, src_key_padding_mask
            del tgt_key_padding_mask

            if self.args.moe_diversity_loss_coefficient > 0 and epoch >= self.args.moe_diversity_inclusion_epoch:
                encoder_moe_gating_variances = torch.stack(encoder_moe_gating_variances).std(dim=0).mean()
                decoder_moe_gating_variances = torch.stack(decoder_moe_gating_variances).std(dim=0).mean()
                moe_diversity_loss = (encoder_moe_gating_variances + decoder_moe_gating_variances) / 2
                encoder_moe_gating_variance_losses.update(encoder_moe_gating_variances.item(), 1)
                decoder_moe_gating_variance_losses.update(decoder_moe_gating_variances.item(), 1)

                moe_diversity_loss = moe_diversity_loss * self.args.moe_diversity_loss_coefficient
            else:
                moe_diversity_loss = 0

            tgt_seqs = tgt_seqs.to(self.decoder_device)

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            translation_loss: torch.Tensor = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

            translation_losses.update(translation_loss.item(), (tgt_seq_lengths - 1).sum().item())

            loss = translation_loss + moe_diversity_loss

            (loss / self.batches_per_step).backward()

            total_losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            del tgt_seqs, tgt_seq_lengths, predicted_sequences, translation_loss, loss

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.batches_per_step == 0:
                if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
                
                if self.args.use_grokfast == 'ema':
                    self.grads = gradfilter_ema(utils.sanitize_model(model), grads=self.grads, alpha=self.args.grokfast_alpha, lamb=self.args.grokfast_lambda)
                elif self.args.use_grokfast == 'ma':
                    self.grads = gradfilter_ma(utils.sanitize_model(model), grads=self.grads, window_size=self.args.grokfast_window_size, lamb=self.args.grokfast_lambda)

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                utils.change_lr(self.optimizer, new_lr=utils.get_lr(self.steps, self.args.d_model, self.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.steps % self.print_frequency == 0:
                    print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches, self.steps, self.n_steps, step_time=step_time, data_time=data_time, total_losses=total_losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.')

                self.summary_writer.add_scalar('train/translation_loss', translation_losses.avg, self.steps)
                self.summary_writer.add_scalar('train/avg_loss', total_losses.avg, self.steps)
                if moe_diversity_loss > 0:
                    self.summary_writer.add_scalar('Encoder MoE Gating Variances', encoder_moe_gating_variance_losses.avg, self.steps)
                    self.summary_writer.add_scalar('Decoder MoE Gating Variances', decoder_moe_gating_variance_losses.avg, self.steps)

                start_step_time = time.time()

                # 'epoch' is 0-indexed
                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (epoch in [self.epochs - 1, self.epochs - 2] or bool(self.args.early_stop)) and self.steps % 1500 == 0:
                    utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/step{str(self.steps)}_")
            start_data_time = time.time()
    
    def validate_epoch(self, model):
        model.eval()

        with torch.no_grad():
            losses = avg_meter.AverageMeter()
            for (src_seqs, tgt_seqs, _, tgt_seq_lengths) in tqdm(self.val_loader, total=self.val_loader.n_batches):
                src_seqs = src_seqs.to(self.encoder_device)
                tgt_seqs = tgt_seqs.to(self.encoder_device)
                tgt_seq_lengths = tgt_seq_lengths.to(self.decoder_device)

                src_key_padding_mask = src_seqs == 0
                tgt_key_padding_mask = (tgt_seqs == 0).to(self.decoder_device)

                predicted_sequences = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)[0]

                tgt_seqs = tgt_seqs.to(self.decoder_device)

                loss: torch.Tensor = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1)

                losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            self.summary_writer.add_scalar('val/avg_loss', losses.avg, self.steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            self.viz_model(self.steps, model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

            return losses.avg

    def evaluate(self, src, tgt):
        best, _ = utils.beam_search_translate(self.args, src, self.model, self.src_bpe_model, self.tgt_bpe_model, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

    def viz_model(self, step, model: Transformer, src, tgt=None):
        if self.args.use_infinite_attention:
            return # temporary; would like to visualize memory block attention weights in the future

        with torch.no_grad():
            model.eval()

            input_sequence = torch.LongTensor(self.src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(self.encoder_device) # (1, input_sequence_length)
            input_tokens = [self.src_bpe_model.decode([id.item()])[0] for id in input_sequence.squeeze(0)]
            input_sequence_length = input_sequence.size(1)

            # pad input sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                input_sequence = torch.cat([input_sequence, torch.zeros([1, self.args.maxlen - input_sequence.size(1)], dtype=torch.long, device=input_sequence.device)], dim=1)

            target_sequence = torch.LongTensor(self.tgt_bpe_model.encode(tgt, eos=True)).unsqueeze(0).to(self.encoder_device) # (1, target_sequence_length)
            target_tokens = [self.tgt_bpe_model.decode([id.item()])[0] for id in target_sequence.squeeze(0)]
            target_sequence_length = target_sequence.size(1)

            # pad target sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                target_sequence = torch.cat([target_sequence, torch.zeros([1, self.args.maxlen - target_sequence.size(1)], dtype=torch.long, device=target_sequence.device)], dim=1)

            src_key_padding_mask = input_sequence == 0 # (N, pad_length)
            tgt_key_padding_mask = (target_sequence == 0).to(self.decoder_device) # (N, pad_length)

            input_sequence = model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
            input_sequence = model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)

            for e, encoder_layer in enumerate(model.encoder.encoder_layers):
                input_sequence, attention_weights = encoder_layer.self_attn(input_sequence, input_sequence, input_sequence, src_key_padding_mask)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().contiguous()

                    # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Encoder-Self', e, i, attention_weight_grid[:, i, :input_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                        self.summary_writer.add_image(f"encoder/viz/layer_{e}/segment_{a}/head_{i}/self-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

                fcn_output = encoder_layer.fcn(input_sequence) # (N, pad_length, d_model)

                if fcn_output is tuple:
                    fcn_output = fcn_output[0]

            input_sequence = model.encoder.norm(input_sequence)
            input_sequence = input_sequence.to(self.decoder_device)
            src_key_padding_mask = src_key_padding_mask.to(self.decoder_device)
                
            target_sequence = model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
            
            target_sequence = target_sequence.to(self.decoder_device)
            
            target_sequence = model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)

            attention_weights: Optional[List[torch.Tensor]] = None
            for d, decoder_layer in enumerate(model.decoder.decoder_layers):
                target_sequence, attention_weights = decoder_layer.self_attn(target_sequence, target_sequence, target_sequence, tgt_key_padding_mask) # (N, pad_length, d_model)
                
                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weight_grid will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Self', d, i, attention_weight_grid[:, i, :target_sequence_length, :target_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, target_tokens)
                        self.summary_writer.add_image(f"decoder/viz/layer_{d}/segment_{a}/head_{i}/self-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

                target_sequence, attention_weights = decoder_layer.cross_attn(target_sequence, input_sequence, input_sequence, src_key_padding_mask) # (N, pad_length, d_model)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Cross', d, i, attention_weight_grid[:, i, :target_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, input_tokens)
                        self.summary_writer.add_image(f"decoder/viz/layer_{d}/segment_{a}/head_{i}/cross-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

                fcn_output = decoder_layer.fcn(target_sequence) # (N, pad_length, d_model)

                if fcn_output is tuple:
                    fcn_output = fcn_output[0]
