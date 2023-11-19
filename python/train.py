from dataloader import SequenceLoader
from prettytable import PrettyTable
from rotary_embedding_torch import RotaryEmbedding
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformer_provider import *
from utils import *

import argparse
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import youtokentome

def main(args):
    run_dir = os.path.join('runs', args.run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    src_tokenizer_file = os.path.join(run_dir, 'src_tokenizer.model')
    tgt_tokenizer_file = os.path.join(run_dir, 'tgt_tokenizer.model')

    if not os.path.exists(src_tokenizer_file):
        raise Exception(f"Source tokenizer file {src_tokenizer_file} does not exist")

    print(f"Loading source tokenizer from {src_tokenizer_file}")
    src_bpe_model = youtokentome.BPE(model=src_tokenizer_file)

    if os.path.exists(tgt_tokenizer_file):
        print(f"Loading target tokenizer from {tgt_tokenizer_file}")
        tgt_bpe_model = youtokentome.BPE(model=tgt_tokenizer_file)
    else:
        print(f"Sharing tokenizer between source and target languages")
        tgt_bpe_model = src_bpe_model

    if args.positional_encoding_type == 'sinusoidal' or args.positional_encoding_type == 'buffer':
        positional_encoding = get_buffered_positional_encoding(
            d_model=args.d_model,
            maxlen=args.maxlen,
        )
    elif args.positional_encoding_type == 'rotary':
        positional_encoding = RotaryEmbedding(dim=64)

    print('Initializing model...')

    if os.path.exists(os.path.join(run_dir, 'transformer_checkpoint.pth.tar')):
        checkpoint = torch.load(os.path.join(run_dir, 'transformer_checkpoint.pth.tar'))
        args.start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % args.start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    else:
        print("Starting from scratch...")
        model = NewTransformerModelProvider().provide(args, src_bpe_model.vocab_size(), tgt_bpe_model.vocab_size(), tie_embeddings=tgt_bpe_model==src_bpe_model, positional_encoding=positional_encoding)
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=[args.beta1, args.beta2], eps=args.epsilon)

    print('Loading training data SequenceLoader...')
    # Initialize data-loaders
    train_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="train",
        tokens_in_batch=args.tokens_in_batch
    )

    print('Loading validation data SequenceLoader...')
    val_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="val",
        tokens_in_batch=args.tokens_in_batch
    )

    print('Loading test data SequenceLoader...')
    test_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="test",
        tokens_in_batch=args.tokens_in_batch
    )

    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')

    criterion = LabelSmoothedCE(args=args, eps=args.label_smoothing).to(args.device)

    model = model.to(args.device)
    
    summary_writer = SummaryWriter(log_dir=os.path.join('runs', args.run_name))

    step = 1
    epochs = (args.n_steps // (train_loader.n_batches // args.batches_per_step)) + 1


    print(f"Training for {epochs} epochs...")
    for epoch in range(args.start_epoch, epochs):
        # Step
        step = epoch * train_loader.n_batches // args.batches_per_step

        # One epoch's training
        train_loader.create_batches()
        test_loader.create_batches()
        train(
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

        # One epoch's validation
        val_loader.create_batches()
        validate(
            args=args,
            val_loader=val_loader,
            step=step,
            model=model,
            criterion=criterion,
            summary_writer=summary_writer
        )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, prefix=f"runs/{args.run_name}/")

    sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python=True)

def train(args, train_loader, test_loader, src_bpe_model, tgt_bpe_model, model, criterion, optimizer, epoch, epochs, step, summary_writer):
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

            # This step is now complete
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
                # get one random test example
                # source_sequence, target_sequence, _, _ = random.choice(test_loader.data)
                evaluate(
                    args=args,
                    model=model,
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
                save_checkpoint(epoch, model, optimizer, prefix=f"runs/{args.run_name}/step{str(step)}_")

        # Reset data time
        start_data_time = time.time()

def validate(args, step, val_loader, model, criterion, summary_writer):
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

def evaluate(args, model, src_bpe_model, tgt_bpe_model, src, tgt):
    best, _ = beam_search_translate(args, src, model, src_bpe_model, tgt_bpe_model, beam_size=4, length_norm_coefficient=0.6)

    debug_validate_table = PrettyTable(["Test Source", "Test Prediction", "Test Target"])
    debug_validate_table.add_row([src, best, tgt])

    console_size = os.get_terminal_size()
    debug_validate_table.max_width = (console_size.columns // 3) - 15
    debug_validate_table.min_width = (console_size.columns // 3) - 15

    print(f"src: {src} | prediction: {best} | actual: {tgt}")
    print(debug_validate_table)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--run_name', type=str, required=True)

    argparser.add_argument('--d_model', type=int, default=512)
    argparser.add_argument('--n_heads', type=int, default=8)
    argparser.add_argument('--d_queries', type=int, default=64)
    argparser.add_argument('--d_values', type=int, default=64)
    argparser.add_argument('--d_inner', type=int, default=2048)
    argparser.add_argument('--n_layers', type=int, default=6)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--maxlen', type=int, default=160)
    argparser.add_argument('--positional_encoding_type', type=str, default='sinusoidal')

    argparser.add_argument('--start_epoch', type=int, default=0)
    argparser.add_argument('--tokens_in_batch', type=int, default=2000)
    argparser.add_argument('--target_tokens_per_batch', type=int, default=25000)
    argparser.add_argument('--n_steps', type=int, default=100000)
    argparser.add_argument('--warmup_steps', type=int, default=8000)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.98)
    argparser.add_argument('--epsilon', type=float, default=1e-9)
    argparser.add_argument('--label_smoothing', type=float, default=0.1)

    argparser.add_argument('--print_frequency', type=int, default=20)

    argparser.add_argument('--device', type=str, default='cuda:0')
    cudnn.benchmark = False

    args, unk = argparser.parse_known_args()

    args.__setattr__('batches_per_step', args.target_tokens_per_batch // args.tokens_in_batch)
    args.__setattr__('lr', get_lr(step=1, d_model=args.d_model, warmup_steps=args.warmup_steps))

    print(f"using learning rate {args.lr}")

    main(args)
