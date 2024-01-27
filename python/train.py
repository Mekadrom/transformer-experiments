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
import torch
import torch.backends.cudnn as cudnn


torch.autograd.set_detect_anomaly(True)

"""
    This entire project is a heavily modified version of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers. Credit to them for the workflow and the implementation of most of the transformer model architecture code in transformer_provider.py.
"""
class Trainer():
    def train(self, args, model_name_prefix=''):
        raise NotImplementedError
    
    def validate_epoch(self, args, step, val_loader, model, src_bpe_model, tgt_bpe_model, criterion, summary_writer):
        """
        One epoch's validation.

        :param val_loader: loader for validation data
        :param model: model
        :param criterion: label-smoothed cross-entropy loss
        """
        model.eval()  # eval mode disables dropout

        # Prohibit gradient computation explicitly
        with torch.no_grad():
            losses = AverageMeter()
            # Batches
            for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(tqdm(val_loader, total=val_loader.n_batches)):
                source_sequence = source_sequence.to(args.device) # (1, source_sequence_length)
                target_sequence = target_sequence.to(args.device) # (1, target_sequence_length)
                source_sequence_length = source_sequence_length.to(args.device) # (1)
                target_sequence_length = target_sequence_length.to(args.device) # (1)

                # Forward prop.
                predicted_sequence = model(source_sequence, target_sequence, source_sequence_length, target_sequence_length) # (1, target_sequence_length, vocab_size)

                # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
                # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
                # Therefore, pads start after (length - 1) positions
                loss = criterion(inputs=predicted_sequence, targets=target_sequence[:, 1:], lengths=target_sequence_length - 1) # scalar

                # Keep track of losses
                losses.update(loss.item(), (target_sequence_length - 1).sum().item())

            # Log to TensorBoard
            summary_writer.add_scalar('Validation Loss', losses.avg, step)
            
            print("\nValidation loss: %.3f\n\n" % losses.avg)

            self.visualize_attention_weights(args, model, src_bpe_model, tgt_bpe_model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", step, summary_writer)

    def evaluate(self, model, device, src_bpe_model, tgt_bpe_model, src, tgt):
        best, _ = beam_search_translate(src, model, src_bpe_model, tgt_bpe_model, device=device, beam_size=4, length_norm_coefficient=0.6)

        debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
        debug_validate_table.add_row([src, best, tgt])

        console_size = os.get_terminal_size()
        debug_validate_table.max_width = (console_size.columns // 3) - 15
        debug_validate_table.min_width = (console_size.columns // 3) - 15

        print(f"src: {src} | prediction: {best} | actual: {tgt}")
        print(debug_validate_table)

    def visualize_attention_weights(self, args, model, src_bpe_model, tgt_bpe_model, src, tgt, step, summary_writer):
        input_sequence = torch.LongTensor(src_bpe_model.encode(src, eos=False)).unsqueeze(0).to(args.device) # (1, input_sequence_length)
        # input_tokens = src_bpe_model.decode(input_sequence[0].tolist())
        input_tokens = [src_bpe_model.decode([id.item()])[0] for id in input_sequence.squeeze(0)]
        input_sequence_length = torch.LongTensor([input_sequence.size(1)]).unsqueeze(0).to(args.device) # (1)
        target_sequence = torch.LongTensor(tgt_bpe_model.encode(tgt, eos=True)).unsqueeze(0).to(args.device) # (1, target_sequence_length)
        # target_tokens = tgt_bpe_model.decode(target_sequence[0].tolist())
        target_tokens = [tgt_bpe_model.decode([id.item()])[0] for id in target_sequence.squeeze(0)]
        target_sequence_length = torch.LongTensor([target_sequence.size(1)]).unsqueeze(0).to(args.device) # (1)

        input_sequence = model.encoder.perform_embedding_transformation(input_sequence) # (N, pad_length, d_model)
        input_sequence = model.encoder.apply_positional_embedding(input_sequence) # (N, pad_length, d_model)
        # input_sequence = self.apply_dropout(input_sequence) # (N, pad_length, d_model) # don't apply dropout for visualization

        for e, encoder_layer in enumerate(model.encoder.encoder_layers):
            input_sequence, attention_weights = encoder_layer[0](query_sequences=input_sequence, key_sequences=input_sequence, value_sequences=input_sequence, key_value_sequence_lengths=input_sequence_length)

            attention_weights = attention_weights.cpu().detach()
            attention_weights = attention_weights.contiguous().view(1, args.n_heads, attention_weights.size(1), attention_weights.size(2))

            # shape of attention_weights will be (1, n_heads, input_sequence_length, input_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
            for i in range(attention_weights.size(1)):
                image_data = self.visualize_attention_weights_for_layer('Encoder-Self', e, i, attention_weights[:, i, :, :].squeeze(0).cpu().detach().numpy(), input_tokens, input_tokens)
                summary_writer.add_image(f"Encoder Layer {e} Head {i} Self-Attn Weights", plt.imread(image_data), global_step=step, dataformats='HWC')

            input_sequence = encoder_layer[1](sequences=input_sequence) # (N, pad_length, d_model)

        input_sequence = model.encoder.layer_norm(input_sequence)

        target_sequence = model.decoder.apply_embedding_transformation(target_sequence) # (N, pad_length, d_model)
        target_sequence = model.decoder.apply_positional_embedding(target_sequence) # (N, pad_length, d_model)
        # target_sequence = self.apply_dropout(target_sequence) # (N, pad_length, d_model) # don't apply dropout for visualization

        for d, decoder_layer in enumerate(model.decoder.decoder_layers):
            target_sequence, attention_weights = decoder_layer[0](query_sequences=target_sequence, key_sequences=target_sequence, value_sequences=target_sequence, key_value_sequence_lengths=target_sequence_length) # (N, pad_length, d_model)
            
            attention_weights = attention_weights.cpu().detach()
            attention_weights = attention_weights.contiguous().view(1, args.n_heads, attention_weights.size(1), attention_weights.size(2))

            # shape of attention_weights will be (1, n_heads, target_sequence_length, target_sequence_length) for self attention (like in encoder layers and beginning of each decoder layer)
            for i in range(attention_weights.size(1)):
                image_data = self.visualize_attention_weights_for_layer('Decoder-Self', d, i, attention_weights[:, i, :, :].squeeze(0).numpy(), target_tokens, target_tokens)
                summary_writer.add_image(f"Decoder Layer {d} Head {i} Self-Attn Weights", plt.imread(image_data), global_step=step, dataformats='HWC')

            target_sequence, attention_weights = decoder_layer[1](query_sequences=target_sequence, key_sequences=input_sequence, value_sequences=input_sequence, key_value_sequence_lengths=input_sequence_length) # (N, pad_length, d_model)

            attention_weights = attention_weights.cpu().detach()
            attention_weights = attention_weights.contiguous().view(1, args.n_heads, attention_weights.size(1), attention_weights.size(2))

            # shape of attention_weights will be (1, n_heads, target_sequence_length, input_sequence_length) for encoder-decoder attention
            for i in range(attention_weights.size(1)):
                image_data = self.visualize_attention_weights_for_layer('Decoder-Cross', d, i, attention_weights[:, i, :, :].squeeze(0).numpy(), input_tokens, target_tokens)
                summary_writer.add_image(f"Decoder Layer {d} Head {i} Cross-Attn Weights", plt.imread(image_data), global_step=step, dataformats='HWC')

            target_sequence = decoder_layer[2](sequences=target_sequence) # (N, pad_length, d_model)

    def visualize_attention_weights_for_layer(self, stack_name, layer_num, head_num, activation_weights, attendee_tokens, attending_tokens):
        # print(f"{stack_name} layer {layer_num} head {head_num} activation_weights.shape: {activation_weights.shape}")
        # activation_weights = activation_weights.transpose(1, 0) # (input_sequence_length, output_sequence_length)
        # print(f"{stack_name} layer {layer_num} head {head_num} activation_weights.shape: {activation_weights.shape}")

        fig, ax = plt.subplots(figsize=(10, 10))
        s = sns.heatmap(activation_weights, square=True, annot=True, annot_kws={"fontsize":6}, fmt=".4f", xticklabels=attendee_tokens, yticklabels=attending_tokens, ax=ax)
        # plt.xlabel("Input Sequence")
        # plt.ylabel("Output Sequence")
        # plt.title(f"{stack_name} Layer {layer_num} Head {head_num}-Attn Weights")
        s.set(xlabel="Input Sequence", ylabel="Output Sequence", title=f"{stack_name}-Attn Layer {layer_num} Head {head_num} Weights")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        return buf

class ClassicTrainer(Trainer):
    def __init__(self):
        super().__init__()

        self.sacrebleu_epochs = []

    def train(self, args, model_name_prefix=''):
        run_dir = os.path.join('runs', args.run_name)

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        summary_writer = SummaryWriter(log_dir=run_dir)

        bpe_run_dir = os.path.join('runs', args.tokenizer_run_name)

        src_bpe_model, tgt_bpe_model = load_tokenizers(bpe_run_dir)

        model, optimizer, positional_encoding = load_checkpoint_or_generate_new(args, run_dir, src_bpe_model=src_bpe_model, tgt_bpe_model=tgt_bpe_model)

        print_model(model)
        print(f"Optimizer: {optimizer}")
        print(f"Positional Encoding: {positional_encoding.shape if type(positional_encoding) == torch.Tensor else positional_encoding}")

        if args.save_initial_checkpoint:
            save_checkpoint(-1, model, optimizer, positional_encoding=positional_encoding, prefix=f"runs/{args.run_name}/{model_name_prefix}")

        model = model.to(args.device)

        if args.start_epoch == 0:
            print("Visualizing attention weights before training...")
            # get attention weight visualization before any updates are made to the model
            self.visualize_attention_weights(args, model, src_bpe_model, tgt_bpe_model, "Anyone who retains the ability to recognise beauty will never become old.", "Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.", 0, summary_writer)

        train_loader, val_loader, test_loader = load_data(args.tokens_in_batch, bpe_run_dir, src_bpe_model, tgt_bpe_model)

        # todo: make this configurable
        criterion = LabelSmoothedCE(args=args, eps=args.label_smoothing).to(args.device)

        epochs = (args.n_steps // (train_loader.n_batches // args.batches_per_step)) + 1

        steps = self.train_n_epochs(args, epochs, model, train_loader, val_loader, test_loader, src_bpe_model, tgt_bpe_model, criterion, optimizer, positional_encoding, summary_writer, model_name_prefix=model_name_prefix)

        average_checkpoints(positional_encoding, args.run_name, model_name_prefix=model_name_prefix)

        val_loader.create_batches()
        self.validate_epoch(
            args=args,
            val_loader=val_loader,
            step=steps,
            model=model,
            src_bpe_model=src_bpe_model,
            tgt_bpe_model=tgt_bpe_model,
            criterion=criterion,
            summary_writer=summary_writer
        )
        sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, device=args.device, sacrebleu_in_python=True)
        return model

    def train_n_epochs(self, args, epochs, model, train_loader, val_loader, test_loader, src_bpe_model, tgt_bpe_model, criterion, optimizer, positional_encoding, summary_writer, model_name_prefix=''):
        step = 1
        print(f"Training for {epochs} epochs...")
        for epoch in range(args.start_epoch, epochs):
            step = (epoch * train_loader.n_batches // args.batches_per_step)

            train_loader.create_batches()
            test_loader.create_batches()
            step = self.train_epoch(
                args=args,
                train_loader=train_loader,
                test_loader=test_loader,
                src_bpe_model=src_bpe_model,
                tgt_bpe_model=tgt_bpe_model,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                epochs=epochs,
                step=step,
                summary_writer=summary_writer
            )

            val_loader.create_batches()
            self.validate_epoch(
                args=args,
                val_loader=val_loader,
                step=step,
                model=model,
                src_bpe_model=src_bpe_model,
                tgt_bpe_model=tgt_bpe_model,
                criterion=criterion,
                summary_writer=summary_writer
            )

            save_checkpoint(epoch, model, optimizer, positional_encoding=positional_encoding, prefix=f"runs/{args.run_name}/{model_name_prefix}")

        return step

    def train_epoch(self, args, train_loader, test_loader, src_bpe_model, tgt_bpe_model, model, criterion, optimizer, epoch, epochs, step, summary_writer):
        """
        One epoch's training.

        :param train_loader: loader for training data
        :param model: model
        :param criterion: label-smoothed cross-entropy loss
        :param optimizer: optimizer
        :param epoch: epoch number
        """
        model.train() # training mode enables dropout

        # Track some metrics
        data_time = AverageMeter() # data loading time
        step_time = AverageMeter() # forward prop. + back prop. time
        losses = AverageMeter() # loss

        # Starting time
        start_data_time = time.time()
        start_step_time = time.time()

        # Batches
        for i, (source_sequences, target_sequences, source_sequence_lengths, target_sequence_lengths) in enumerate(train_loader):
            # Move to default device
            source_sequences = source_sequences.to(args.device) # (N, max_source_sequence_pad_length_this_batch)
            target_sequences = target_sequences.to(args.device) # (N, max_target_sequence_pad_length_this_batch)
            source_sequence_lengths = source_sequence_lengths.to(args.device) # (N)
            target_sequence_lengths = target_sequence_lengths.to(args.device) # (N)

            # Time taken to load data
            data_time.update(time.time() - start_data_time)

            # Forward prop.
            predicted_sequences = model(source_sequences, target_sequences, source_sequence_lengths, target_sequence_lengths) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

            # Note: If the target sequence is "<BOS> w1 w2 ... wN <EOS> <PAD> <PAD> <PAD> <PAD> ..."
            # we should consider only "w1 w2 ... wN <EOS>" as <BOS> is not predicted
            # Therefore, pads start after (length - 1) positions
            loss = criterion(inputs=predicted_sequences, targets=target_sequences[:, 1:], lengths=target_sequence_lengths - 1) # scalar

            # Backward prop.
            (loss / args.batches_per_step).backward()

            # Keep track of losses
            losses.update(loss.item(), (target_sequence_lengths - 1).sum().item())

            # Update model (i.e. perform a training step) only after gradients are accumulated from batches_per_step batches
            if (i + 1) % args.batches_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()

                step += 1

                # Update learning rate after each step
                change_lr(optimizer, new_lr=get_lr(step=step, d_model=args.d_model, warmup_steps=args.warmup_steps))

                # Time taken for this training step
                step_time.update(time.time() - start_step_time)

                # Print status
                if step % args.print_frequency == 0:
                    print('Epoch {0}/{1}-----'
                        'Batch {2}/{3}-----'
                        'Step {4}/{5}-----'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----'
                        'Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                        'Loss {losses.val:.4f} ({losses.avg:.4f})'.format(epoch + 1, 
                                                                            epochs,
                                                                            i + 1, 
                                                                            train_loader.n_batches,
                                                                            step, 
                                                                            args.n_steps,
                                                                            step_time=step_time,
                                                                            data_time=data_time,
                                                                            losses=losses))
                    self.evaluate(
                        model=model,
                        device=args.device,
                        src_bpe_model=src_bpe_model,
                        tgt_bpe_model=tgt_bpe_model,
                        src='Anyone who retains the ability to recognise beauty will never become old.',
                        tgt='Wer die Fähigkeit behält, Schönheit zu erkennen, wird niemals alt.',
                    )

                # Log to TensorBoard
                summary_writer.add_scalar('Training Loss', losses.avg, step)

                # Reset step time
                start_step_time = time.time()

                # If this is the last one or two epochs, save checkpoints at regular intervals for averaging
                if epoch in [epochs - 1, epochs - 2] and step % 1500 == 0:  # 'epoch' is 0-indexed
                    save_checkpoint(epoch, model, optimizer, positional_encoding=model.positional_encoding, prefix=f"runs/{args.run_name}/step{str(step)}_")

            # Reset data time
            start_data_time = time.time()

        return step

if __name__ == '__main__':
    args, unk = get_args()
    cudnn.benchmark = False

    print(f"using learning rate {args.lr}")

    def do_training(trainer: Trainer):
        if args.prune_mode == 'train-prune':
            model = trainer.train(args)
            prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
        elif args.prune_mode == 'train-prune-retrain':
            model = trainer.train(args)

            args.n_steps = args.prune_retrain_n_steps
            args.warmup_steps = args.prune_retrain_warmup_steps

            for i in range(args.n_prune_retrains):
                prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
                trainer.train(args, model_name_prefix=f"pruned_{i}_")
        elif args.prune_mode == 'only-prune':
            src_bpe_model, tgt_bpe_model = load_tokenizers(os.path.join('runs', args.run_name))
            model, _, _ = load_checkpoint_or_generate_new(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, checkpoint_model_name='averaged_transformer_checkpoint.pth.tar')
            prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
            sacrebleu_evaluate(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, model, device=args.device, sacrebleu_in_python=True)
        else:
            trainer.train(args)

    trainer: Trainer | None = None

    if args.distillation_teacher_run_name is not None:
        # trainer = DistillationTrainer()
        pass
    else:
        trainer = ClassicTrainer()

    do_training(trainer)