from criteria.labelsmooth import LabelSmoothedCE
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import *

import io
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time

class ClassicTrainer():
    def __init__(self, args):
        self.args = args

        self.run_name = args.run_name
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.n_steps = args.n_steps
        self.warmup_steps = args.warmup_steps
        self.device = args.device
        self.print_frequency = args.print_frequency
        self.batches_per_step = args.batches_per_step

        self.run_dir = os.path.join('runs', self.run_name)

        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.summary_writer = SummaryWriter(log_dir=self.run_dir)

        self.bpe_run_dir = os.path.join('runs', args.tokenizer_run_name)

        self.src_bpe_model, self.tgt_bpe_model = load_tokenizers(self.bpe_run_dir)

        self.model, self.optimizer, self.positional_encoding = self.load_model_positional_encoding_and_optimizer()
        self.model = self.model.to(args.device)
        self.positional_encoding.to(args.device)

        print_model(self.model)
        print(f"Optimizer: {self.optimizer}")
        print(f"Positional Encoding: {self.positional_encoding.shape if type(self.positional_encoding) == torch.Tensor else self.positional_encoding}")

        if args.save_initial_checkpoint:
            save_checkpoint(-1, self.model, self.optimizer, self.positional_encoding, f"runs/{args.run_name}/")

        if args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            self.visualize_attention_weights(0, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

        self.train_loader, self.val_loader, self.test_loader = load_data(args.tokens_in_batch, self.bpe_run_dir, self.src_bpe_model, self.tgt_bpe_model)

        # todo: make this configurable
        self.criterion = LabelSmoothedCE(args=args, eps=args.label_smoothing).to(self.device)

        self.steps = 0
        self.start_epoch = args.start_epoch
        self.epochs = (args.n_steps // (self.train_loader.n_batches // args.batches_per_step)) + 1

        self.sacrebleu_epochs = []
        self.target_sequence_transform = lambda source_sequences, source_sequence_lengths, target_sequences, target_sequence_lengths: (target_sequences, target_sequence_lengths)

    def load_model_positional_encoding_and_optimizer(self):
        return load_checkpoint_or_generate_new(self.args, self.run_dir, src_bpe_model=self.src_bpe_model, tgt_bpe_model=self.tgt_bpe_model)

    def train(self, model_name_prefix=''):
        print(f"Training for {self.epochs} epochs...")
        for epoch in range(self.start_epoch, self.epochs):
            self.steps = (epoch * self.train_loader.n_batches // self.batches_per_step)

            self.train_loader.create_batches()
            self.train_epoch(epoch=epoch)

            self.val_loader.create_batches()
            self.validate_epoch()

            save_checkpoint(epoch, self.model, self.optimizer, self.positional_encoding, prefix=f"runs/{self.run_name}/{model_name_prefix}")

        average_checkpoints(self.positional_encoding, self.run_name, model_name_prefix=model_name_prefix)

        self.val_loader.create_batches()
        self.validate_epoch()
        sacrebleu_evaluate(self.run_dir, self.src_bpe_model, self.tgt_bpe_model, self.model, device=self.device, sacrebleu_in_python=True)
    
    def train_epoch(self, epoch):
        # training mode enables dropout
        self.model.train()

        data_time = AverageMeter()
        step_time = AverageMeter()
        losses = AverageMeter()

        start_data_time = time.time()
        start_step_time = time.time()

        for i, (src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths) in enumerate(self.train_loader):
            src_seqs = src_seqs.to(self.device) # (N, max_source_sequence_pad_length_this_batch)
            tgt_seqs = tgt_seqs.to(self.device) # (N, max_target_sequence_pad_length_this_batch)
            src_seq_lengths = src_seq_lengths.to(self.device) # (N)
            tgt_seq_lengths = tgt_seq_lengths.to(self.device) # (N)

            tgt_seqs, tgt_seq_lengths = self.target_sequence_transform(src_seqs, src_seq_lengths, tgt_seqs, tgt_seq_lengths)

            data_time.update(time.time() - start_data_time)

            predicted_sequences = self.model(src_seqs, tgt_seqs, src_seq_lengths, tgt_seq_lengths) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            loss = self.criterion(inputs=predicted_sequences, targets=tgt_seqs[:, 1:], lengths=tgt_seq_lengths - 1) # scalar

            (loss / self.batches_per_step).backward()

            losses.update(loss.item(), (tgt_seq_lengths - 1).sum().item())

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % self.batches_per_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.steps += 1

                change_lr(self.optimizer, new_lr=get_lr(self.steps, self.d_model, self.warmup_steps))

                step_time.update(time.time() - start_step_time)

                if self.steps % self.print_frequency == 0:
                    print('Epoch {0}/{1}-----Batch {2}/{3}-----Step {4}/{5}-----Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                          'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(epoch + 1, self.epochs, i + 1,  self.train_loader.n_batches, self.steps, self.n_steps, step_time=step_time, data_time=data_time, losses=losses))
                    self.evaluate(src='Anyone who retains the ability to recognise beauty will never become old.', tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.')

                self.summary_writer.add_scalar('Training Loss', losses.avg, self.steps)

                start_step_time = time.time()

                # 'epoch' is 0-indexed
                if epoch in [self.epochs - 1, self.epochs - 2] and self.steps % 1500 == 0:
                    save_checkpoint(epoch, self.model, self.optimizer, positional_encoding=self.positional_encoding, prefix=f"runs/{self.run_name}/step{str(self.steps)}_")
            start_data_time = time.time()
    
    def validate_epoch(self):
        """
        One epoch's validation.

        :param val_loader: loader for validation data
        :param model: model
        :param criterion: label-smoothed cross-entropy loss
        """
        # eval mode disables dropout
        self.model.eval()

        # prohibit gradient computation explicitly
        with torch.no_grad():
            losses = AverageMeter()
            for (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in tqdm(self.val_loader, total=self.val_loader.n_batches):
                source_sequence = source_sequence.to(self.device) # (1, source_sequence_length)
                target_sequence = target_sequence.to(self.device) # (1, target_sequence_length)
                source_sequence_length = source_sequence_length.to(self.device) # (1)
                target_sequence_length = target_sequence_length.to(self.device) # (1)

                predicted_sequence = self.model(source_sequence, target_sequence, source_sequence_length, target_sequence_length) # (1, target_sequence_length, vocab_size)

                # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
                # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
                # Therefore, pads start after (length - 1) positions
                loss = self.criterion(inputs=predicted_sequence, targets=target_sequence[:, 1:], lengths=target_sequence_length - 1) # scalar

                losses.update(loss.item(), (target_sequence_length - 1).sum().item())

            self.summary_writer.add_scalar('Validation Loss', losses.avg, self.steps)
            print("\nValidation loss: %.3f\n\n" % losses.avg)
            self.visualize_attention_weights(self.steps, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.")

    def evaluate(self, src, tgt):
        best, _ = beam_search_translate(src, self.model, self.src_bpe_model, self.tgt_bpe_model, device=self.device, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(f"src: {src} | prediction: {best} | actual: {tgt}")
        print(debug_validate_table)

    def visualize_attention_weights(self, step, src, tgt):
        input_sequence = torch.LongTensor(self.src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(self.device) # (1, input_sequence_length)
        input_tokens = [self.src_bpe_model.decode([id.item()])[0] for id in input_sequence.squeeze(0)]
        input_sequence_length = torch.LongTensor([input_sequence.size(1)]).unsqueeze(0).to(self.device) # (1)
        target_sequence = torch.LongTensor(self.tgt_bpe_model.encode(tgt, eos=True)).unsqueeze(0).to(self.device) # (1, target_sequence_length)
        target_tokens = [self.tgt_bpe_model.decode([id.item()])[0] for id in target_sequence.squeeze(0)]
        target_sequence_length = torch.LongTensor([target_sequence.size(1)]).unsqueeze(0).to(self.device) # (1)

        input_sequence = self.model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
        input_sequence = self.model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)

        for e, encoder_layer in enumerate(self.model.encoder.encoder_layers):
            self_mha, attention_weights = encoder_layer.mha(input_sequence, input_sequence, input_sequence, input_sequence_length)

            attention_weights = attention_weights.cpu().detach()
            attention_weights = attention_weights.contiguous().view(1, self.n_heads, attention_weights.size(1), attention_weights.size(2))

            # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
            for i in range(attention_weights.size(1)):
                image_data = self.visualize_attention_weights_for_layer('Encoder-Self', e, i, attention_weights[:, i, :, :].squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                self.summary_writer.add_image(f"Encoder Layer {e} Head {i} Self-Attn Weights", plt.imread(image_data), global_step=step, dataformats='HWC')

            if encoder_layer.mca is not None:
                mca_attn = encoder_layer.mca(input_sequence, input_sequence)

                input_sequence = torch.cat([self_mha, mca_attn], dim=-1)

            input_sequence = encoder_layer.fcn(sequences=input_sequence) # (N, pad_length, d_model)

        input_sequence = self.model.encoder.layer_norm(input_sequence)

        target_sequence = self.model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
        target_sequence = self.model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)

        for d, decoder_layer in enumerate(self.model.decoder.decoder_layers):
            self_mha_out, attention_weights = decoder_layer.self_mha(target_sequence, target_sequence, target_sequence, target_sequence_length) # (N, pad_length, d_model)
            
            attention_weights = attention_weights.cpu().detach()
            attention_weights = attention_weights.contiguous().view(1, self.n_heads, attention_weights.size(1), attention_weights.size(2))

            # shape of attention_weights will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
            for i in range(attention_weights.size(1)):
                image_data = self.visualize_attention_weights_for_layer('Decoder-Self', d, i, attention_weights[:, i, :, :].squeeze(0).numpy(), target_tokens, target_tokens)
                self.summary_writer.add_image(f"Decoder Layer {d} Head {i} Self-Attn Weights", plt.imread(image_data), global_step=step, dataformats='HWC')

            if decoder_layer.self_mca is not None:
                mca_attn = decoder_layer.self_mca(target_sequence, target_sequence)

                target_sequence = torch.cat([self_mha_out, mca_attn], dim=-1)

            target_sequence, attention_weights = decoder_layer.cross_mha(target_sequence, input_sequence, input_sequence, input_sequence_length) # (N, pad_length, d_model)

            attention_weights = attention_weights.cpu().detach()
            attention_weights = attention_weights.contiguous().view(1, self.n_heads, attention_weights.size(1), attention_weights.size(2))

            # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
            for i in range(attention_weights.size(1)):
                image_data = self.visualize_attention_weights_for_layer('Decoder-Cross', d, i, attention_weights[:, i, :, :].squeeze(0).numpy(), input_tokens, target_tokens)
                self.summary_writer.add_image(f"Decoder Layer {d} Head {i} Cross-Attn Weights", plt.imread(image_data), global_step=step, dataformats='HWC')

            target_sequence = decoder_layer.fcn(target_sequence) # (N, pad_length, d_model)

    def visualize_attention_weights_for_layer(self, stack_name, layer_num, head_num, activation_weights, attendee_tokens, attending_tokens):
        fig, ax = plt.subplots(figsize=(10, 10))
        s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
        s.set(xlabel="Input Sequence", ylabel="Output Sequence", title=f"{stack_name}-Attn Layer {layer_num} Head {head_num} Weights")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf
