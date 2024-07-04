from criteria.labelsmooth import LabelSmoothedCE
from prettytable import PrettyTable
from tqdm import tqdm
from utils import *
from modules import calc_kl_loss

import base_trainer
import matplotlib.pyplot as plt
import os
import time
import utils

class TranslationTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(TranslationTrainer, self).__init__(args, 'translation')

    def load_model_and_optimizer(self):
        return utils.load_translation_checkpoint_or_generate_new(self.args, self.run_dir, self.src_bpe_model.vocab_size(), self.tgt_bpe_model.vocab_size(), tie_embeddings=self.tgt_bpe_model==self.src_bpe_model, vae_model=self.args.train_vae)

    def get_criteria(self):
        return LabelSmoothedCE(args=self.args, eps=self.args.label_smoothing).to(self.device)

    def load_data(self):
        return utils.load_translation_data(self.args.tokens_in_batch, self.bpe_run_dir, self.src_bpe_model, self.tgt_bpe_model, vae_model=self.args.train_vae, pad_to_length=self.args.maxlen if self.args.use_infinite_attention else None)
    
    def train(self, model_name_prefix=''):
        if self.args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                if self.args.train_vae:
                    self.viz_model(0, self.model, "In protest against the planned tax on the rich, the French Football Association is set to actually go through with the first strike since 1972.")
                else:
                    self.viz_model(0, self.model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        super().train()

    def train_epoch(self, model, epoch):
        # training mode enables dropout
        model.train()

        data_time = AverageMeter()
        step_time = AverageMeter()
        total_losses = AverageMeter()
        kl_losses = AverageMeter()
        translation_losses = AverageMeter()
        encoder_moe_gating_variance_losses = AverageMeter()
        decoder_moe_gating_variance_losses = AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths) in enumerate(self.train_loader):
            src_seqs = src_seqs.to(self.device) # (N, max_source_sequence_pad_length_this_batch)
            tgt_seqs = tgt_seqs.to(self.device) # (N, max_target_sequence_pad_length_this_batch)
            src_seq_lengths = src_seq_lengths.to(self.device) # (N)
            tgt_seq_lengths = tgt_seq_lengths.to(self.device) # (N)

            tgt_seqs, tgt_seq_lengths = self.target_sequence_transform(src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths)

            src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
            tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

            data_time.update(time.time() - start_data_time)

            predicted_sequences, e_t_vars, e_q_vars, e_k_vars, e_v_vars, encoder_moe_gating_variances, d_t_vars, d_s_q_vars, d_s_k_vars, d_s_v_vars, d_c_q_vars, d_c_k_vars, d_c_v_vars, decoder_moe_gating_variances = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            if self.args.moe_diversity_loss_coefficient > 0 and epoch >= self.args.moe_diversity_inclusion_epoch:
                encoder_moe_gating_variances = torch.stack(encoder_moe_gating_variances).std(dim=0).mean()
                decoder_moe_gating_variances = torch.stack(decoder_moe_gating_variances).std(dim=0).mean()
                moe_diversity_loss = (encoder_moe_gating_variances + decoder_moe_gating_variances) / 2
                encoder_moe_gating_variance_losses.update(encoder_moe_gating_variances.item(), 1)
                decoder_moe_gating_variance_losses.update(decoder_moe_gating_variances.item(), 1)

                moe_diversity_loss = moe_diversity_loss * self.args.moe_diversity_loss_coefficient
            else:
                moe_diversity_loss = 0

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            translation_loss = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

            translation_losses.update(translation_loss.item(), (tgt_seq_lengths - 1).sum().item())

            kl_loss = torch.FloatTensor([0]).to(self.device)

            if e_t_vars is not None and e_t_vars[0] is not None and e_t_vars[1] is not None:
                kl_loss += calc_kl_loss(e_t_vars[0], e_t_vars[1])
            if e_q_vars is not None and e_q_vars[0] is not None and e_q_vars[1] is not None:
                kl_loss += calc_kl_loss(e_q_vars[0], e_q_vars[1])
            if e_k_vars is not None and e_k_vars[0] is not None and e_k_vars[1] is not None:
                kl_loss += calc_kl_loss(e_k_vars[0], e_k_vars[1])
            if e_v_vars is not None and e_v_vars[0] is not None and e_v_vars[1] is not None:
                kl_loss += calc_kl_loss(e_v_vars[0], e_v_vars[1])

            if d_t_vars is not None and d_t_vars[0] is not None and d_t_vars[1] is not None:
                kl_loss += calc_kl_loss(d_t_vars[0], d_t_vars[1])
            if d_s_q_vars is not None and d_s_q_vars[0] is not None and d_s_q_vars[1] is not None:
                kl_loss += calc_kl_loss(d_s_q_vars[0], d_s_q_vars[1])
            if d_s_k_vars is not None and d_s_k_vars[0] is not None and d_s_k_vars[1] is not None:
                kl_loss += calc_kl_loss(d_s_k_vars[0], d_s_k_vars[1])
            if d_s_v_vars is not None and d_s_v_vars[0] is not None and d_s_v_vars[1] is not None:
                kl_loss += calc_kl_loss(d_s_v_vars[0], d_s_v_vars[1])
            if d_c_q_vars is not None and d_c_q_vars[0] is not None and d_c_q_vars[1] is not None:
                kl_loss += calc_kl_loss(d_c_q_vars[0], d_c_q_vars[1])
            if d_c_k_vars is not None and d_c_k_vars[0] is not None and d_c_k_vars[1] is not None:
                kl_loss += calc_kl_loss(d_c_k_vars[0], d_c_k_vars[1])
            if d_c_v_vars is not None and d_c_v_vars[0] is not None and d_c_v_vars[1] is not None:
                kl_loss += calc_kl_loss(d_c_v_vars[0], d_c_v_vars[1])

            kl_losses.update(kl_loss.item(), (tgt_seq_lengths - 1).sum().item())

            loss = translation_loss + moe_diversity_loss + kl_loss

            (loss / self.batches_per_step).backward()

            total_losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.batches_per_step == 0:
                if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                change_lr(self.optimizer, new_lr=get_lr(self.steps, self.d_model, self.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.steps % self.print_frequency == 0:
                    print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches, self.steps, self.n_steps, step_time=step_time, data_time=data_time, total_losses=total_losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    if self.args.train_vae:
                        self.evaluate(src='In protest against the planned tax on the rich, the French Football Association is set to actually go through with the first strike since 1972.', tgt='In protest against the planned tax on the rich, the French Football Association is set to actually go through with the first strike since 1972.')
                    else:
                        self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.')

                self.summary_writer.add_scalar('Translation Training Loss', translation_losses.avg, self.steps)
                self.summary_writer.add_scalar('Training Loss', total_losses.avg, self.steps)
                if moe_diversity_loss > 0:
                    self.summary_writer.add_scalar('Encoder MoE Gating Variances', encoder_moe_gating_variance_losses.avg, self.steps)
                    self.summary_writer.add_scalar('Decoder MoE Gating Variances', decoder_moe_gating_variance_losses.avg, self.steps)

                if self.args.train_vae:
                    self.summary_writer.add_scalar('KL Loss', kl_losses.avg, self.steps)

                start_step_time = time.time()

                # 'epoch' is 0-indexed
                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (epoch in [self.epochs - 1, self.epochs - 2] or self.args.early_stop) and self.steps % 1500 == 0:
                    save_checkpoint(epoch, self.model, self.optimizer, prefix=f"{self.run_dir}/step{str(self.steps)}_")
            start_data_time = time.time()
    
    def validate_epoch(self, model):
        model.eval()

        with torch.no_grad():
            losses = AverageMeter()
            for (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths) in tqdm(self.val_loader, total=self.val_loader.n_batches):
                src_seqs = src_seqs.to(self.device) # (1, max_source_sequence_pad_length_this_batch)
                tgt_seqs = tgt_seqs.to(self.device) # (1, max_target_sequence_pad_length_this_batch)
                src_seq_lengths = src_seq_lengths.to(self.device) # (1)
                tgt_seq_lengths = tgt_seq_lengths.to(self.device) # (1)

                src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)
                tgt_key_padding_mask = tgt_seqs == 0 # (N, max_target_sequence_pad_length_this_batch)

                predicted_sequences = model(src_seqs, tgt_seqs, src_key_padding_mask, tgt_key_padding_mask)[0] # (N, max_target_sequence_pad_length_this_batch, vocab_size)

                # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
                # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
                # Therefore, pads start after (length - 1) positions
                loss = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

                losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            self.summary_writer.add_scalar('Validation Loss', losses.avg, self.steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            if self.args.train_vae:
                self.viz_model(self.steps, model, "In protest against the planned tax on the rich, the French Football Association is set to actually go through with the first strike since 1972.", "In protest against the planned tax on the rich, the French Football Association is set to actually go through with the first strike since 1972.")
            else:
                self.viz_model(self.steps, model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

            return losses.avg

    def evaluate(self, src, tgt):
        if self.args.train_vae:
            translation = greedy_translate(self.args, src, self.model, self.src_bpe_model, self.tgt_bpe_model, device=self.device)

            debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
            debug_validate_table.add_row([src, translation, tgt])

            console_size = os.get_terminal_size()
            debug_validate_table.max_width = (console_size.columns // 3) - 15
            debug_validate_table.min_width = (console_size.columns // 3) - 15

            # print(f"src: {src} | prediction: {translation} | actual: {tgt}")
            print(debug_validate_table)
        else:
            best, _ = beam_search_translate(self.args, src, self.model, self.src_bpe_model, self.tgt_bpe_model, device=self.device, beam_size=4, length_norm_coefficient=0.6)

            debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
            debug_validate_table.add_row([src, best, tgt])

            console_size = os.get_terminal_size()
            debug_validate_table.max_width = (console_size.columns // 3) - 15
            debug_validate_table.min_width = (console_size.columns // 3) - 15

            # print(f"src: {src} | prediction: {best} | actual: {tgt}")
            print(debug_validate_table)

    def viz_model(self, step, model, src, tgt=None):
        with torch.no_grad():
            model.eval()
            is_vae = False
            if tgt == None:
                is_vae = True
                tgt = src

            input_sequence = torch.LongTensor(self.src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(self.device) # (1, input_sequence_length)
            input_tokens = [self.src_bpe_model.decode([id.item()])[0] for id in input_sequence.squeeze(0)]
            input_sequence_length = input_sequence.size(1)
            input_zeros = torch.zeros([1, self.args.maxlen - input_sequence.size(1)], dtype=torch.long, device=input_sequence.device)
            input_sequence = torch.cat([input_sequence, input_zeros], dim=1)

            # pad input sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                input_sequence = torch.cat([input_sequence, torch.zeros([1, self.args.maxlen - input_sequence.size(1)], dtype=torch.long, device=input_sequence.device)], dim=1)

            target_sequence = torch.LongTensor(self.tgt_bpe_model.encode(tgt, eos=True)).unsqueeze(0).to(self.device) # (1, target_sequence_length)
            target_tokens = [self.tgt_bpe_model.decode([id.item()])[0] for id in target_sequence.squeeze(0)]
            target_sequence_length = target_sequence.size(1)
            target_zeros = torch.zeros([1, self.args.maxlen - target_sequence.size(1)], dtype=torch.long, device=target_sequence.device)
            target_sequence = torch.cat([target_sequence, target_zeros], dim=1)

            # pad target sequence to args.maxlen
            if self.args.use_infinite_attention or True:
                target_sequence = torch.cat([target_sequence, torch.zeros([1, self.args.maxlen - target_sequence.size(1)], dtype=torch.long, device=target_sequence.device)], dim=1)

            src_key_padding_mask = input_sequence == 0 # (N, pad_length)
            tgt_key_padding_mask = target_sequence == 0 # (N, pad_length)

            input_sequence, _, _ = model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
            input_sequence = model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)

            for e, encoder_layer in enumerate(model.encoder.encoder_layers):
                input_sequence, attention_weights, _, _, _, _, _, _ = encoder_layer.self_attn(input_sequence, input_sequence, input_sequence, src_key_padding_mask)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().contiguous()

                    # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Encoder-Self', e, i, attention_weight_grid[:, i, :input_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                        self.summary_writer.add_image(f"Encoder Layer {e} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

                input_sequence, _ = encoder_layer.fcn(sequences=input_sequence) # (N, pad_length, d_model)

            input_sequence = model.encoder.norm(input_sequence)

            if is_vae:
                cls_token = input_sequence[:, 0, :]
                mu = model.mu(cls_token)
                logvar = model.logvar(cls_token)
                z = model.reparameterize(mu, logvar).unsqueeze(1)

                src_key_padding_mask = torch.zeros([1, self.args.latent_seq_len], dtype=torch.bool, device=z.device)

                input_sequence = model.decoder_extrapolator(z).view(z.size(0), -1, self.args.d_model)

                input_tokens = [f"latent_{i}" for i in range(self.args.latent_seq_len)]
                
            target_sequence, _, _ = model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
            target_sequence = model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)

            for d, decoder_layer in enumerate(model.decoder.decoder_layers):
                target_sequence, attention_weights, _, _, _, _, _, _ = decoder_layer.self_attn(target_sequence, target_sequence, target_sequence, tgt_key_padding_mask) # (N, pad_length, d_model)
                
                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weight_grid will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Self', d, i, attention_weight_grid[:, i, :target_sequence_length, :target_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, target_tokens)
                        self.summary_writer.add_image(f"Decoder Layer {d} Head {i} Self-Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

                target_sequence, attention_weights, _, _, _, _, _, _ = decoder_layer.cross_attn(target_sequence, input_sequence, input_sequence, src_key_padding_mask) # (N, pad_length, d_model)

                for a, attention_weight_grid in enumerate(attention_weights):
                    attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                    # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
                    for i in range(attention_weight_grid.size(1)):
                        image_data = self.viz_attn_weights('Decoder-Cross', d, i, attention_weight_grid[:, i, :target_sequence_length, :input_sequence_length].transpose(-2, -1).squeeze(0).numpy(), target_tokens, input_tokens)
                        self.summary_writer.add_image(f"Decoder Layer {d} Head {i} Cross-Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

                target_sequence, _ = decoder_layer.fcn(target_sequence) # (N, pad_length, d_model)
