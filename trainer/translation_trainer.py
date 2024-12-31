from criteria.labelsmooth import LabelSmoothedCE
from functools import partial
from model_provider import TranslationTransformerModelProvider
from modules import multihead_attn, transformer
from modules.grokfast import gradfilter_ma, gradfilter_ema
from prettytable import PrettyTable
from tqdm import tqdm
from typing import List, Optional
from multigpu_translation_training_wrapper import MultiGPUTranslationWrapper

import avg_meter
import base_trainer
import debugger
import matplotlib.pyplot as plt
import os
import time
import torch
import utils
import youtokentome as yttm

class TranslationTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(TranslationTrainer, self).__init__(args)

        self.grads = None

    def load_model_and_optimizer(self, run_dir, checkpoint_model_name='transformer_checkpoint.pth.tar'):
        print('Initializing model...')

        tie_embeddings = self.src_tokenizer == self.tgt_tokenizer
        if hasattr(self.args, 'tie_embeddings'):
            tie_embeddings = bool(self.args.tie_embeddings)

        if os.path.exists(os.path.join(run_dir, checkpoint_model_name)):
            checkpoint = torch.load(os.path.join(run_dir, checkpoint_model_name))
            if hasattr(self.args, 'start_epoch') and self.args.start_epoch == 0:
                self.args.start_epoch = checkpoint['epoch'] + 1
                print('\nLoaded checkpoint from epoch %d.\n' % self.args.start_epoch)

            model = TranslationTransformerModelProvider().provide_transformer(self.args, utils.vocab_size(self.args, self.src_tokenizer), utils.vocab_size(self.args, self.tgt_tokenizer), tie_embeddings=tie_embeddings)

            model.load_state_dict(checkpoint['model'].state_dict())

            if 'optimizer' in checkpoint:
                optimizer = checkpoint['optimizer']
            else:
                optimizer = None
        else:
            print("Starting from scratch...")
            model = TranslationTransformerModelProvider().provide_transformer(self.args, utils.vocab_size(self.args, self.src_tokenizer), utils.vocab_size(self.args, self.tgt_tokenizer), tie_embeddings=tie_embeddings)

            optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=self.args.lr, betas=[self.args.beta1, self.args.beta2], eps=self.args.epsilon)

        if hasattr(self.args, 'multidevice') and bool(self.args.multidevice):
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
        return utils.load_translation_data(self.args, int(self.args.tokens_in_batch), self.bpe_run_dir, self.src_tokenizer, self.tgt_tokenizer, pad_to_length=self.args.maxlen if self.args.use_infinite_attention else None)
    
    def train(self, model_name_prefix=''):
        if self.args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", src_lang_code="en", tgt_lang_code="de")

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

        predicted_sequences = predicted_sequences.to(self.args.decoder_device)
        tgt_seqs = tgt_seqs.to(self.args.decoder_device)

        moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances = self.moe_criterion(epoch, encoder_moe_gating_variances, decoder_moe_gating_variances)

        loss = translation_loss + moe_diversity_loss
        loss = (loss / self.batches_per_step).to(self.args.decoder_device)

        loss.backward()

        return translation_loss, loss, moe_diversity_loss, encoder_moe_gating_variances, decoder_moe_gating_variances
    
    def train_epoch(self, model: transformer.Transformer, epoch):
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

            if bool(self.args.multilang):
                backward_translation_loss, total_loss, moe_loss, encoder_moe_loss, decoder_moe_loss = self.forward_pass(model, epoch, tgt_seqs, src_seqs, src_seq_lengths, tgt_key_padding_mask, src_key_padding_mask)
                translation_losses.update(backward_translation_loss.item(), sum_total_lengths)
                total_losses.update(total_loss.item(), sum_total_lengths)
                moe_losses.update(moe_loss.item(), 1)
                encoder_moe_gating_variance_losses.update(encoder_moe_loss.item(), 1)
                decoder_moe_gating_variance_losses.update(decoder_moe_loss.item(), 1)

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
                    if self.args.multilang:
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', src_lang_code='en', tgt_lang_code='de')
                        self.evaluate(src='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='Anyone who retains the ability to recognise beauty will never become old.', src_lang_code='de', tgt_lang_code='en')
                        self.evaluate(src='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.', tgt='Ceux qui conservent la capacité de reconnaître la beauté ne vieillissent jamais.', src_lang_code='de', tgt_lang_code='fr')
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Tie, kas saglabā spēju atpazīt skaistumu, nekad nenoveco.', src_lang_code='en', tgt_lang_code='lv')
                        self.evaluate(src='Ceux qui conservent la capacité de reconnaître la beauté ne vieillissent jamais.', tgt='Tie, kas saglabā spēju atpazīt skaistumu, nekad nenoveco.', src_lang_code='fr', tgt_lang_code='lv')
                    else:
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.')

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

            self.viz_model(self.steps, model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", src_lang_code="en", tgt_lang_code="de")

            return losses.avg

    def evaluate(self, src, tgt, src_lang_code=None, tgt_lang_code=None):
        if src_lang_code is None or tgt_lang_code is None:
            best, _ = utils.beam_search_translate(self.args, src, self.tgt_tokenizer.subword_to_id('<BOS>'), self.model, self.src_tokenizer, self.tgt_tokenizer, beam_size=4, length_norm_coefficient=0.6)
        else:
            src = f"{src_lang_code}__{src}"
            tgt = f"{tgt_lang_code}__{tgt}"
            best, _ = utils.beam_search_translate(self.args, src, utils.lang_code_to_id(tgt_lang_code), self.model, self.src_tokenizer, self.tgt_tokenizer, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(debug_validate_table)

    def viz_attn_weights(self, attn: multihead_attn.MultiHeadAttention, attn_residual, layer_num, src_seq, src_seq_len, key_padding_mask, tgt_seq, tgt_seq_len, src_tokens, tgt_tokens, summary_writer, step):
        decoder_or_encoder = 'decoder' if attn.in_decoder else 'encoder'
        self_or_cross = 'self' if attn.self_attn else 'cross'
        stack_name = f"{decoder_or_encoder}-{self_or_cross}"

        residual, attention_weights = attn(tgt_seq, src_seq, src_seq, key_padding_mask)
        tgt_seq = attn_residual(tgt_seq, residual)

        for a, attention_weight_grid in enumerate(attention_weights):
            attention_weight_grid = attention_weight_grid.cpu().contiguous()
            for head_num in range(attention_weight_grid.size(1)):
                image_data = self.viz_attn_weight(stack_name, layer_num, head_num, attention_weight_grid[:, head_num, :tgt_seq_len, :src_seq_len].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), tgt_tokens, src_tokens)
                summary_writer.add_image(f"{decoder_or_encoder}/viz/layer_{layer_num}/segment_{a}/head_{head_num}/{self_or_cross}-attn", plt.imread(image_data), global_step=step, dataformats='HWC')

        return tgt_seq

    def viz_encoder_layer(self, encoder_layer: transformer.EncoderLayer, src_seq, src_seq_len, src_key_padding_mask, src_tokens, summary_writer, step, layer_num):
        print(f"Visualizing encoder layer {layer_num}...")
        residual = self.viz_attn_weights(encoder_layer.self_attn, encoder_layer.self_attn_residual, layer_num, src_seq, src_seq_len, src_key_padding_mask, src_seq, src_seq_len, src_tokens, src_tokens, summary_writer, step)
        fcn_out, _ = encoder_layer.fcn(residual)
        return encoder_layer.fcn_residual(residual, fcn_out)

    def viz_encoder_layers(self, encoder_layers: list[transformer.EncoderLayer], seq, seq_len, key_padding_mask, tokens, summary_writer, step):
        for e, encoder_layer in enumerate(encoder_layers):
            seq = self.viz_encoder_layer(encoder_layer, seq, seq_len, key_padding_mask, tokens, summary_writer, step, e)
        return seq

    def viz_encoder(self, encoder: transformer.Encoder, seq, seq_len, key_padding_mask, tokens, summary_writer, step):
        print("Visualizing encoder...")
        if (hasattr(encoder, 'embed_tokens') and encoder.embed_tokens is not None) or (hasattr(encoder, 'embedding') and encoder.embed_tokens is not None):
            seq = encoder.apply_embedding_transformation(seq)
        seq = encoder.apply_positional_embedding(seq)
        seq = self.viz_encoder_layers(encoder.encoder_layers, seq, seq_len, key_padding_mask, tokens, summary_writer, step)
        return encoder.post_encoder_norm(seq).to(self.args.decoder_device)

    def viz_decoder_layer(self, decoder_layer: transformer.DecoderLayer, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step, layer_num):
        print(f"Visualizing decoder layer {layer_num}...")
        tgt_seq = self.viz_attn_weights(decoder_layer.self_attn, decoder_layer.self_attn_residual, layer_num, tgt_seq, tgt_seq_len, tgt_key_padding_mask, tgt_seq, tgt_seq_len, tgt_tokens, tgt_tokens, summary_writer, step)
        residual = self.viz_attn_weights(decoder_layer.cross_attn, decoder_layer.cross_attn_residual, layer_num, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, src_tokens, tgt_tokens, summary_writer, step)
        fcn_out, _ = decoder_layer.fcn(residual)
        return decoder_layer.fcn_residual(residual, fcn_out)

    def viz_decoder_layers(self, decoder_layers, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step):
        for d, decoder_layer in enumerate(decoder_layers):
            tgt_seq = self.viz_decoder_layer(decoder_layer, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step, d)
        return tgt_seq

    def viz_decoder(self, decoder: transformer.Decoder, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step):
        print("Visualizing decoder...")
        if (hasattr(decoder, 'embed_tokens') and decoder.embed_tokens is not None) or (hasattr(decoder, 'embedding') and decoder.embed_tokens is not None):
            tgt_seq = decoder.apply_embedding_transformation(tgt_seq)
        tgt_seq = decoder.apply_positional_embedding(tgt_seq.to(self.args.decoder_device))
        return self.viz_decoder_layers(decoder.decoder_layers, src_seq, src_seq_len, src_key_padding_mask, tgt_seq, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, summary_writer, step)

    def viz_model(self, step, model: transformer.Transformer, src, tgt=None, src_lang_code=None, tgt_lang_code=None):
        print("Visualizing model...")
        if self.args.use_infinite_attention:
            return # todo: temporary; would like to visualize memory block attention weights in the future
        
        if src_lang_code is not None and tgt_lang_code is not None and self.args.multilang:
            src = f"{src_lang_code}__{src}"
            tgt = f"{tgt_lang_code}__{tgt}"

        src_encode = partial(utils.encode, bool(self.args.multilang), self.src_tokenizer)
        src_decode = partial(utils.decode, bool(self.args.multilang), self.src_tokenizer)

        tgt_encode = partial(utils.encode, bool(self.args.multilang), self.tgt_tokenizer)
        tgt_decode = partial(utils.decode, bool(self.args.multilang), self.tgt_tokenizer)

        model.eval()
        with torch.no_grad():
            src = torch.LongTensor(src_encode(sentences=src, eos=bool(self.args.multilang)))
            src_tokens = src_decode(ids=src)[-1][0]
            src = src.to(self.encoder_device)

            tgt = torch.LongTensor(tgt_encode(sentences=tgt, bos=True, eos=True))
            tgt_tokens = tgt_decode(ids=tgt)[-1][0]
            tgt = tgt.to(self.decoder_device)

            src_seq_len = src.size(1)
            tgt_seq_len = tgt.size(1)
            
            maxlen = self.args.maxlen

            # pad input sequence to args.maxlen
            src = torch.cat([src, torch.zeros([1, maxlen - src.size(1)], dtype=torch.long, device=src.device)], dim=1)
            tgt = torch.cat([tgt, torch.zeros([1, maxlen - tgt.size(1)], dtype=torch.long, device=tgt.device)], dim=1)

            src_key_padding_mask = src == 0
            tgt_key_padding_mask = tgt.to(self.args.decoder_device) == 0

            if bool(self.args.debug):
                debugger.analyze_encoder_decoder_devices(self.args, model.encoder, model.decoder, src, tgt, src_key_padding_mask, tgt_key_padding_mask, torch.LongTensor([tgt_seq_len]))

            # permanent device assignments
            src = src.to(self.args.encoder_device)
            tgt = tgt.to(self.args.decoder_device)
            src_key_padding_mask = src_key_padding_mask.to(self.args.encoder_device)
            tgt_key_padding_mask = tgt_key_padding_mask.to(self.args.decoder_device)
            model.encoder.embed_tokens = model.encoder.embed_tokens.to(self.args.encoder_device)


            src = self.viz_encoder(model.encoder, src, src_seq_len, src_key_padding_mask, src_tokens, self.summary_writer, step)

            # things that need to move to decoder device
            src = src.to(self.args.decoder_device)
            src_key_padding_mask = src_key_padding_mask.to(self.args.decoder_device)
            model.decoder.embed_tokens = model.decoder.embed_tokens.to(self.args.decoder_device)
            model.decoder.lm_head = model.decoder.lm_head.to(self.args.decoder_device)

            tgt = self.viz_decoder(model.decoder, src, src_seq_len, src_key_padding_mask, tgt, tgt_seq_len, tgt_key_padding_mask, src_tokens, tgt_tokens, self.summary_writer, step)
