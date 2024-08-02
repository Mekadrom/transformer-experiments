from criteria.labelsmooth import LabelSmoothedCE

import avg_meter
import base_trainer
import matplotlib.pyplot as plt
import time
import torch
import tqdm
import utils

class LLMTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(LLMTrainer, self).__init__(args, 'llm')
        
        if args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            with torch.no_grad():
                self.model.eval()
                self.viz_model(0, self.model, "Anyone who retains the ability to recognise beauty will never become ", "Anyone who retains the ability to recognise beauty will never become old")

    def load_model_and_optimizer(self):
        return utils.load_llm_checkpoint_or_generate_new(self.args, self.run_dir, self.src_bpe_model.vocab_size())

    def get_criteria(self):
        return LabelSmoothedCE(args=self.args, eps=self.args.label_smoothing)

    def load_data(self):
        return utils.load_llm_data(self.args.train_dataset, self.src_bpe_model, self.args.maxlen, batch_size=self.args.batch_size)

    def train_epoch(self, model, epoch):
        # training mode enables dropout
        model.train()

        data_time = avg_meter.AverageMeter()
        step_time = avg_meter.AverageMeter()
        total_losses = avg_meter.AverageMeter()
        llm_losses = avg_meter.AverageMeter()
        moe_gating_variance_losses = avg_meter.AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, src_seqs in enumerate(self.train_loader):
            src_seqs = src_seqs.to(self.decoder_device) # (N, max_source_sequence_pad_length_this_batch)
            src_seq_lengths = src_seq_lengths.to(self.decoder_device) # (N)

            src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)

            data_time.update(time.time() - start_data_time)

            predicted_sequences, d_t_vars, d_s_q_vars, d_s_k_vars, d_s_v_vars, d_c_q_vars, d_c_k_vars, d_c_v_vars, gating_variances = model(src_seqs, src_seqs, src_seq_lengths.unsqueeze(-1), src_seq_lengths.unsqueeze(-1), src_key_padding_mask, src_key_padding_mask) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            if self.args.moe_diversity_loss_coefficient > 0 and epoch >= self.args.moe_diversity_inclusion_epoch:
                moe_diversity_loss = torch.stack(gating_variances).std(dim=0).mean()
                moe_gating_variance_losses.update(moe_diversity_loss.item(), 1)

                moe_diversity_loss = moe_diversity_loss * self.args.moe_diversity_loss_coefficient
            else:
                moe_diversity_loss = 0

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            llm_loss = self.criterion(inputs=predicted_sequences, targets=src_seqs[:, 1:], lengths=src_seq_lengths - 1) # scalar

            llm_losses.update(llm_loss.item(), (src_seq_lengths - 1).sum().item())

            loss = llm_loss + moe_diversity_loss

            (loss / self.batches_per_step).backward()

            total_losses.update(loss.item(), (src_seq_lengths - 1).sum().item())

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.batches_per_step == 0:
                if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                utils.change_lr(self.optimizer, new_lr=utils.get_lr(self.steps, self.d_model, self.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.steps % self.print_frequency == 0:
                    print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {total_losses.val:.4f} ({total_losses.avg:.4f})-----Early Stopping Counter: {early_stop_counter}/{early_stop_patience}'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches, self.steps, self.n_steps, step_time=step_time, data_time=data_time, total_losses=total_losses, early_stop_counter=self.early_stopping.counter if self.early_stopping is not None else 0, early_stop_patience=self.early_stopping.patience if self.early_stopping is not None else 0))
                    self.evaluate(src='Anyone who retains the ability to recognise beauty will never become ', tgt='Anyone who retains the ability to recognise beauty will never become old')

                self.summary_writer.add_scalar('LLM Training Loss', llm_losses.avg, self.steps)
                self.summary_writer.add_scalar('Training Loss', total_losses.avg, self.steps)
                if moe_diversity_loss > 0:
                    self.summary_writer.add_scalar('LLM MoE Gating Variances', moe_gating_variance_losses.avg, self.steps)

                start_step_time = time.time()

                # 'epoch' is 0-indexed
                # early stopping requires the ability to average the last few checkpoints so just save all of them
                if (epoch in [self.epochs - 1, self.epochs - 2] or bool(self.args.early_stop)) and self.steps % 1500 == 0:
                    utils.save_checkpoint(epoch, self.model, self.optimizer, prefix=f"runs/{self.run_name}/step{str(self.steps)}_")
            start_data_time = time.time()

    def validate_epoch(self, model):
        model.eval()

        losses = avg_meter.AverageMeter()

        with torch.no_grad():
            for src_seqs in tqdm(self.val_loader, total=self.val_loader.n_batches):
                src_seqs = src_seqs.to(self.decoder_device) # (1, max_source_sequence_pad_length_this_batch)
                src_seq_lengths = src_seq_lengths.to(self.decoder_device) # (1)

                src_key_padding_mask = src_seqs == 0 # (N, max_source_sequence_pad_length_this_batch)

                predicted_sequences = model(src_seqs, src_seqs, src_seq_lengths.unsqueeze(-1), src_seq_lengths.unsqueeze(-1), src_key_padding_mask, src_key_padding_mask)[0] # (N, max_target_sequence_pad_length_this_batch, vocab_size)

                # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
                # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
                # Therefore, pads start after (length - 1) positions
                loss = self.criterion(inputs=predicted_sequences, targets=src_seqs[:, 1:], lengths=src_seq_lengths - 1) # scalar

                losses.update(loss.item(), (src_seq_lengths - 1).sum().item())

            self.summary_writer.add_scalar('Validation Loss', losses.avg, self.steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            self.viz_model(self.steps, model, "Anyone who retains the ability to recognise beauty will never become ", "Anyone who retains the ability to recognise beauty will never become old")

            return losses.avg
        
    def evaluate(self, src, tgt):
        src_sequence = torch.LongTensor(self.src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(self.decoder_device)
        for _ in range(self.args.maxlen - src_sequence.size(1)):

            src_key_padding_mask = src_sequence == 0

            predicted_sequences = self.model(src_sequence, src_sequence, src_key_padding_mask, src_key_padding_mask)[0] # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            predicted_token = self.src_bpe_model.decode([torch.argmax(predicted_sequences[0, -1]).item()])[0]

            if predicted_token == self.src_bpe_model.eos_token:
                break

            src += predicted_token

            src_sequence = torch.LongTensor(self.src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(self.decoder_device)

        return src

    def viz_model(self, step, model, src, tgt):
        input_sequence = torch.LongTensor(self.src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(self.decoder_device) # (1, input_sequence_length)
        input_tokens = [self.src_bpe_model.decode([id.item()])[0] for id in input_sequence.squeeze(0)]
        # pad input sequence to args.maxlen
        input_sequence = torch.cat((input_sequence, torch.zeros(1, self.args.maxlen - input_sequence.size(1), dtype=torch.long).to(self.decoder_device)), dim=1) # (1, args.maxlen)

        src_key_padding_mask = input_sequence == 0 # (N, pad_length)

        input_sequence, _, _ = model.apply_embedding_transformation(input_sequence) # (N, pad_length, d_model)
        input_sequence = model.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)

        for d, decoder_layer in enumerate(model.decoder_layers):
            input_sequence, attention_weights, _, _, _, _, _, _ = decoder_layer.self_attn(input_sequence, input_sequence, input_sequence, input_sequence, src_key_padding_mask) # (N, pad_length, d_model)
            
            for a, attention_weight_grid in enumerate(attention_weights):
                attention_weight_grid = attention_weight_grid.cpu().detach().contiguous()

                # shape of attention_weights will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
                for i in range(attention_weight_grid.size(1)):
                    image_data = self.viz_attn_weights('LLM', d, i, attention_weight_grid[:, i, :, :].transpose(-2, -1).squeeze(0).numpy(), input_tokens, input_tokens)
                    self.summary_writer.add_image(f"LLM Layer {d} Head {i} Attn Weights for Segment {a}", plt.imread(image_data), global_step=step, dataformats='HWC')

            input_sequence, _ = decoder_layer.fcn(input_sequence) # (N, pad_length, d_model)
