from criteria.labelsmooth import LabelSmoothedCE
from model_provider import TranslationTransformerModelProvider
from modules.grokfast import gradfilter_ma, gradfilter_ema
from modules.transformer import Transformer
from prettytable import PrettyTable
from tqdm import tqdm
from typing import List, Optional
from multigpu_translation_training_wrapper import MultiGPUTranslationWrapper

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

    def load_model_and_optimizer(self, run_dir, checkpoint_model_name='transformer_checkpoint.pth.tar'):
        print('Initializing model...')

        tie_embeddings = self.src_bpe_model == self.tgt_bpe_model
        if hasattr(self.args, 'tie_embeddings'):
            tie_embeddings = bool(self.args.tie_embeddings)

        if os.path.exists(os.path.join(run_dir, checkpoint_model_name)):
            checkpoint = torch.load(os.path.join(run_dir, checkpoint_model_name))
            if hasattr(self.args, 'start_epoch') and self.args.start_epoch == 0:
                self.args.start_epoch = checkpoint['epoch'] + 1
                print('\nLoaded checkpoint from epoch %d.\n' % self.args.start_epoch)

            model = TranslationTransformerModelProvider().provide_transformer(self.args, utils.vocab_size(self.src_bpe_model), utils.vocab_size(self.tgt_bpe_model), tie_embeddings=tie_embeddings)

            model.load_state_dict(checkpoint['model'].state_dict())

            if 'optimizer' in checkpoint:
                optimizer = checkpoint['optimizer']
            else:
                optimizer = None
        else:
            print("Starting from scratch...")
            model = TranslationTransformerModelProvider().provide_transformer(self.args, utils.vocab_size(self.src_bpe_model), utils.vocab_size(self.tgt_bpe_model), tie_embeddings=tie_embeddings)

            optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=self.args.lr, betas=[self.args.beta1, self.args.beta2], eps=self.args.epsilon)

        if bool(self.args.multidevice):
            self.model = MultiGPUTranslationWrapper(
                model=model,
                optimizer=optimizer,
                gpu_ids=list(range(torch.cuda.device_count())),
                sync_steps=self.args.multidevice_sync_steps
            )
            self.model.criterion = self.criterion
            self.model.moe_criterion = self.moe_criterion
        
        return model, optimizer

    def get_criteria(self):
        return LabelSmoothedCE(args=self.args, eps=self.args.label_smoothing)

    def load_data(self):
        return utils.load_translation_data(self.args, int(self.args.tokens_in_batch), self.bpe_run_dir, self.src_bpe_model, self.tgt_bpe_model, pad_to_length=self.args.maxlen if self.args.use_infinite_attention else None)
    
    def train(self, model_name_prefix=''):
        if self.args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, "en__Anyone who retains the ability to recognise beauty will never become old.", "de__Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        super().train()

    def forward_pass(self, model, epoch,
                     src_seqs: torch.Tensor,
                     tgt_seqs: torch.Tensor,
                     tgt_seq_lengths: torch.Tensor,
                     src_key_padding_mask: torch.Tensor,
                     tgt_key_padding_mask: torch.Tensor):
        predicted_sequences, encoder_moe_gating_variances, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)

        # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
        # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
        # Therefore, pads start after (length - 1) positions
        translation_loss: torch.Tensor = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1)

        moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances = self.moe_criterion(epoch, encoder_moe_gating_variances, decoder_moe_gating_variances)

        loss = translation_loss + moe_diversity_loss
        loss = (loss / self.batches_per_step).to(self.args.decoder_device)

        loss.backward()

        return translation_loss, loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances
    
    def train_epoch(self, model: Transformer, epoch):
        # training mode enables dropout
        model.train()

        data_time = avg_meter.AverageMeter()
        step_time = avg_meter.AverageMeter()
        total_losses = avg_meter.AverageMeter()
        translation_losses = avg_meter.AverageMeter()
        moe_losses = avg_meter.AverageMeter()
        encoder_moe_gating_variance_losses = avg_meter.AverageMeter()
        decoder_moe_gating_variance_losses = avg_meter.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths) in enumerate(self.train_loader):
            src_seqs, src_seq_lengths = self.target_sequence_transform(tgt_seqs, tgt_seq_lengths, src_seqs, src_seq_lengths)
            tgt_seqs, tgt_seq_lengths = self.target_sequence_transform(src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths)

            src_key_padding_mask = src_seqs == 0
            tgt_key_padding_mask = tgt_seqs == 0

            data_time.update(time.time() - start_data_time)

            sum_total_lengths = (tgt_seq_lengths - 1).sum().item()

            forward_translation_loss, total_loss, moe_loss, encoder_moe_loss, decoder_moe_loss = self.forward_pass(model, epoch, src_seqs, tgt_seqs, tgt_seq_lengths, src_key_padding_mask, tgt_key_padding_mask)
            translation_losses.update(forward_translation_loss.item(), sum_total_lengths)
            total_losses.update(total_loss.item(), sum_total_lengths)
            moe_losses.update(moe_loss.item(), 1)
            encoder_moe_gating_variance_losses.update(encoder_moe_loss.item(), 1)
            decoder_moe_gating_variance_losses.update(decoder_moe_loss.item(), 1)

            backward_translation_loss, total_loss, moe_loss, encoder_moe_loss, decoder_moe_loss = self.forward_pass(model, epoch, tgt_seqs, src_seqs, src_seq_lengths, tgt_key_padding_mask, src_key_padding_mask)
            translation_losses.update(backward_translation_loss.item(), sum_total_lengths)
            total_losses.update(total_loss.item(), sum_total_lengths)
            moe_losses.update(moe_loss.item(), 1)
            encoder_moe_gating_variance_losses.update(encoder_moe_loss.item(), 1)
            decoder_moe_gating_variance_losses.update(decoder_moe_loss.item(), 1)

            # translation_loss, total_loss, moe_loss, encoder_moe_loss, decoder_moe_loss = self.forward_pass(model, epoch, src_seqs, src_seqs, src_seq_lengths, src_key_padding_mask, src_key_padding_mask)
            # translation_losses.update(translation_loss.item(), sum_total_lengths)
            # total_losses.update(total_loss.item(), sum_total_lengths)
            # moe_losses.update(moe_loss.item(), 1)
            # encoder_moe_gating_variance_losses.update(encoder_moe_loss.item(), 1)
            # decoder_moe_gating_variance_losses.update(decoder_moe_loss.item(), 1)

            # translation_loss, total_loss, moe_loss, encoder_moe_loss, decoder_moe_loss = self.forward_pass(model, epoch, tgt_seqs, tgt_seqs, tgt_seq_lengths, tgt_key_padding_mask, tgt_key_padding_mask)
            # translation_losses.update(translation_loss.item(), sum_total_lengths)
            # total_losses.update(total_loss.item(), sum_total_lengths)
            # moe_losses.update(moe_loss.item(), 1)
            # encoder_moe_gating_variance_losses.update(encoder_moe_loss.item(), 1)
            # decoder_moe_gating_variance_losses.update(decoder_moe_loss.item(), 1)

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
                          'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches * len(self.train_loader.src_file_paths), self.steps, self.n_steps, step_time=step_time, data_time=data_time, total_losses=total_losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    self.evaluate(src='en__Anyone who retains the ability to recognise beauty will never become old.', tgt='de__Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt_lang_code='de')
                    self.evaluate(src='de__Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='en__Anyone who retains the ability to recognise beauty will never become old.', tgt_lang_code='en')
                    self.evaluate(src='de__Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='fr__Ceux qui conservent la capacité de reconnaître la beauté ne vieillissent jamais.', tgt_lang_code='fr')
                    self.evaluate(src='en__Anyone who retains the ability to recognise beauty will never become old.', tgt='lv_Tie, kas saglabā spēju atpazīt skaistumu, nekad nenoveco.', tgt_lang_code='lv')
                    self.evaluate(src='fr__Ceux qui conservent la capacité de reconnaître la beauté ne vieillissent jamais.', tgt='lv_Tie, kas saglabā spēju atpazīt skaistumu, nekad nenoveco.', tgt_lang_code='lv')

                self.summary_writer.add_scalar('train/translation_loss', translation_losses.avg, self.steps)
                self.summary_writer.add_scalar('train/avg_loss', total_losses.avg, self.steps)
                if moe_loss > 0:
                    self.summary_writer.add_scalar('Encoder MoE Gating Variances', encoder_moe_gating_variance_losses.avg, self.steps)
                    self.summary_writer.add_scalar('Decoder MoE Gating Variances', decoder_moe_gating_variance_losses.avg, self.steps)

                start_step_time = time.time()

                # epoch is 0-indexed
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

            self.viz_model(self.steps, model, "en__Anyone who retains the ability to recognise beauty will never become old.", "de__Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

            return losses.avg

    def evaluate(self, src, tgt, tgt_lang_code):
        best, _ = utils.beam_search_translate(self.args, src, utils.lang_code_to_id(tgt_lang_code), self.model, self.src_bpe_model, self.tgt_bpe_model, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

    def viz_model(self, step, model: Transformer, src, tgt=None):
        if self.args.use_infinite_attention:
            return # todo: temporary; would like to visualize memory block attention weights in the future

        with torch.no_grad():
            model.eval()

            input_sequence = torch.LongTensor(utils.encode(self.args, self.src_bpe_model, src, eos=True)).unsqueeze(0).to(self.encoder_device) # (1, input_sequence_length)
            _, input_tokens = utils.decode(self.args, self.src_bpe_model, input_sequence)
            input_tokens = input_tokens[0]
            input_sequence_length = input_sequence.size(1)

            # pad input sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                input_sequence = torch.cat([input_sequence, torch.zeros([1, self.args.maxlen - input_sequence.size(1)], dtype=torch.long, device=input_sequence.device)], dim=1)

            target_sequence = torch.LongTensor(utils.encode(self.args, self.tgt_bpe_model, tgt, eos=True)).unsqueeze(0).to(self.encoder_device) # (1, target_sequence_length)
            _, target_tokens = utils.decode(self.args, self.tgt_bpe_model, target_sequence)
            target_tokens = target_tokens[0]
            target_sequence_length = target_sequence.size(1)

            # pad target sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                target_sequence = torch.cat([target_sequence, torch.zeros([1, self.args.maxlen - target_sequence.size(1)], dtype=torch.long, device=target_sequence.device)], dim=1)

            src_key_padding_mask = input_sequence == 0 # (N, pad_length)
            tgt_key_padding_mask = (target_sequence == 0).to(self.decoder_device) # (N, pad_length)

            input_sequence = model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
            input_sequence = model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)

            for e, encoder_layer in enumerate(model.encoder.encoder_layers):
                encoder_sequences, attention_weights = encoder_layer.self_attn(input_sequence, input_sequence, input_sequence, src_key_padding_mask)
                input_sequence = encoder_layer.self_attn_residual(input_sequence, encoder_sequences)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().contiguous()

                    # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Encoder-Self', e, i, attention_weight_grid[:, i, :input_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                        self.summary_writer.add_image(f"encoder/viz/layer_{e}/segment_{a}/head_{i}/self-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

                encoder_sequences = encoder_layer.fcn(input_sequence) # (N, pad_length, d_model)
                input_sequence = encoder_layer.fcn_residual(input_sequence, encoder_sequences)

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
                decoder_sequences, attention_weights = decoder_layer.self_attn(target_sequence, target_sequence, target_sequence, tgt_key_padding_mask) # (N, pad_length, d_model)
                target_sequence = decoder_layer.self_attn_residual(target_sequence, decoder_sequences)
                
                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weight_grid will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Self', d, i, attention_weight_grid[:, i, :target_sequence_length, :target_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, target_tokens)
                        self.summary_writer.add_image(f"decoder/viz/layer_{d}/segment_{a}/head_{i}/self-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

                decoder_sequences, attention_weights = decoder_layer.cross_attn(target_sequence, input_sequence, input_sequence, src_key_padding_mask) # (N, pad_length, d_model)
                target_sequence = decoder_layer.cross_attn_residual(target_sequence, decoder_sequences)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Cross', d, i, attention_weight_grid[:, i, :target_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, input_tokens)
                        self.summary_writer.add_image(f"decoder/viz/layer_{d}/segment_{a}/head_{i}/cross-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

                decoder_sequences = decoder_layer.fcn(target_sequence) # (N, pad_length, d_model)
                target_sequence = decoder_layer.fcn_residual(target_sequence, decoder_sequences)

                if fcn_output is tuple:
                    fcn_output = fcn_output[0]
