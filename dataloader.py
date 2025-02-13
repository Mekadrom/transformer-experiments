from functools import partial
from itertools import groupby
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import Optional

import codecs
import glob
import os
import random
import torch
import youtokentome as yttm
import utils

class SequenceLoader(object):
    """
    An iterator for loading batches of data into the transformer model.

    For training:

        Each batch contains tokens_in_batch target language tokens (approximately),
        target language sequences of the same length to minimize padding and therefore memory usage,
        source language sequences of very similar (if not the same) lengths to minimize padding and therefore memory usage.
        Batches are also shuffled.

    For validation and testing:

        Each batch contains just a single source-target pair, in the same order as in the files from which they were read.
    """
    def __init__(self, args, src_tokenizer: yttm.BPE, tgt_tokenizer: Optional[yttm.BPE], data_folder, source_suffix, target_suffix: Optional[str], split, tokens_in_batch, pad_to_length=None):
        """
        :param data_folder: folder containing the source and target language data files
        :param source_suffix: the filename suffix for the source language files
        :param target_suffix: the filename suffix for the target language files
        :param split: train, or val, or test?
        :param tokens_in_batch: the number of target language tokens in each batch
        """
        self.args = args
        self.src_tokenizer: yttm.BPE = src_tokenizer
        self.tgt_tokenizer: Optional[yttm.BPE] = tgt_tokenizer
        self.data_folder = data_folder
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix

        assert split.lower() in {"train", "val", "test"}, "'split' must be one of 'train', 'val', 'test'! (case-insensitive)"
        self.split = split.lower()

        self.tokens_in_batch = tokens_in_batch
        self.pad_to_length = pad_to_length

        self.for_training = self.split == "train"

        self.src_file_paths = sorted(glob.glob(os.path.join(data_folder, f"{split}_*.{source_suffix}")))
        self.tgt_file_paths = None if target_suffix is None else sorted(glob.glob(os.path.join(data_folder, f"{split}_*.{target_suffix}")))

        if self.tgt_file_paths is not None:
            assert len(self.src_file_paths) == len(self.tgt_file_paths), f"There are a different number of source or target files for split: {split}"

        self.file_idx = 0

        self.src_encode = partial(utils.encode, bool(self.args.multilang), self.src_tokenizer)
        self.tgt_encode = partial(utils.encode, bool(self.args.multilang), self.tgt_tokenizer) if self.tgt_tokenizer is not None else None

        # Create batches
        self.create_batches()

    def create_batches(self):
        # Load data
        with codecs.open(os.path.join(self.data_folder, ".".join([f"{self.split}_{self.file_idx}", self.source_suffix])), "r", encoding="utf-8") as f:
            source_data = f.read().split("\n")[:-1]

        if self.tgt_tokenizer is None:
            with codecs.open(os.path.join(self.data_folder, ".".join([f"{self.split}_{self.file_idx}", self.target_suffix])), "r", encoding="utf-8") as f:
                target_data = f.read().split("\n")[:-1]
            assert len(source_data) == len(target_data), "There are a different number of source or target sequences!"
        else:
            target_data = source_data

        source_lengths = [len(s) for s in tqdm(self.src_encode(source_data, eos=bool(self.args.multilang)), desc='Encoding src sequences')]
        if target_data is not None:
            # target language sequences have <BOS> (language specific) and <EOS> (language agnostic) tokens
            target_lengths = [len(t) for t in tqdm(self.tgt_encode(target_data, bos=True, eos=True), desc='Encoding tgt sequences')]
            self.data = list(zip(source_data, target_data, source_lengths, target_lengths))
        else:
            self.data = list(zip(source_data, source_lengths))

        # If for training, pre-sort by target lengths - required for itertools.groupby() later
        if self.for_training:
            if target_data is not None:
                self.data.sort(key=lambda x: (x[2] + x[3]) // 2)
            else:
                self.data.sort(key=lambda x: x[1])
        
        if self.for_training:
            # Group or chunk based on target sequence lengths
            if target_data is not None:
                chunks = [list(g) for _, g in groupby(self.data, key=lambda x: (x[2] + x[3]) // 2)]
            else:
                chunks = [list(g) for _, g in groupby(self.data, key=lambda x: x[1])]

            # Create batches, each with the same target sequence length
            self.all_batches = list()
            for chunk in chunks:
                # Sort inside chunk by source sequence lengths, so that a batch would also have similar source sequence lengths
                if target_data is not None:
                    chunk.sort(key=lambda x: x[2])
                else:
                    chunk.sort(key=lambda x: x[1])
                # How many sequences in each batch? Divide expected batch size (i.e. tokens) by target sequence length in this chunk
                seqs_per_batch = self.tokens_in_batch // chunk[0][3]
                # Split chunk into batches
                self.all_batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            random.shuffle(self.all_batches)
            self.n_batches = len(self.all_batches)
            self.current_batch = -1
        else:
            # Simply return once pair at a time
            self.all_batches = [[d] for d in self.data]
            self.n_batches = len(self.all_batches)
            self.current_batch = -1

    def __iter__(self):
        return self

    def __next__(self):
        """
        :returns: the next batch, containing:
            source language sequences, a tensor of size (N, encoder_sequence_pad_length)
            target language sequences, a tensor of size (N, decoder_sequence_pad_length)
            true source language lengths, a tensor of size (N)
            true target language lengths, typically the same as decoder_sequence_pad_length as these sequences are bucketed by length, a tensor of size (N)
        """
        self.current_batch += 1
        try:
            try:
                batch = zip(*self.all_batches[self.current_batch])
                if len(batch) == 4:
                    source_data, target_data, source_lengths, target_lengths = batch
                else:
                    source_data, source_lengths = batch
                    target_data = None
                    target_lengths = None
            except:
                try:
                    self.file_idx += 1
                    if self.file_idx >= len(self.src_file_paths):
                        raise StopIteration
                    
                    self.create_batches()
                    self.current_batch += 1

                    batch = zip(*self.all_batches[self.current_batch])
                    if len(batch) == 4:
                        source_data, target_data, source_lengths, target_lengths = batch
                    else:
                        source_data, source_lengths = batch
                        target_data = None
                        target_lengths = None
                except:
                    raise StopIteration
        except:
            self.file_idx = 0
            raise StopIteration
        
        source_data = list(source_data)
        if target_data is not None:
            target_data = list(target_data)

        source_data = self.src_encode(source_data, eos=bool(self.args.multilang))
        source_data = pad_sequence(sequences=[torch.LongTensor(s) for s in source_data], batch_first=True, padding_value=self.src_tokenizer.subword_to_id('<PAD>'))

        if target_data is not None:
            target_data = self.tgt_encode(target_data, bos=True, eos=True)
            target_data = pad_sequence(sequences=[torch.LongTensor(t) for t in target_data], batch_first=True, padding_value=self.tgt_tokenizer.subword_to_id('<PAD>'))

        if self.pad_to_length is not None:
            source_data = torch.cat([source_data, torch.zeros(source_data.size(0), self.pad_to_length - source_data.size(1), dtype=source_data.dtype)], dim=1)

            if target_data is not None:
                target_data = torch.cat([target_data, torch.zeros(target_data.size(0), self.pad_to_length - target_data.size(1), dtype=target_data.dtype)], dim=1)

        source_lengths = torch.LongTensor(source_lengths)

        if target_data is not None:
            target_lengths = torch.LongTensor(target_lengths)

            return source_data, target_data, source_lengths, target_lengths
        else:
            return source_data, source_lengths
