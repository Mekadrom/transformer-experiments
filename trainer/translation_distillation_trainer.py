from .translation_trainer import TranslationTrainer
from utils import *

import os

class DistillationTrainer(TranslationTrainer):
    def __init__(self, args):
        super(DistillationTrainer, self).__init__(args)
        
        self.target_sequence_transform = self._target_sequence_transform_func

    def load_model_and_optimizer(self):
        # load teacher model as well
        self.teacher_model, _ = super(DistillationTrainer, self).load_model_and_optimizer(os.path.join('runs', self.args.distillation_teacher_run_name), checkpoint_model_name='averaged_transformer_checkpoint.pth.tar')
        self.teacher_model.encoder = self.teacher_model.encoder.to(self.args.encoder_device)
        self.teacher_model.decoder = self.teacher_model.decoder.to(self.args.decoder_device)
        self.teacher_model.eval()
        return super().load_model_and_optimizer(self.run_dir)

    def _target_sequence_transform_func(self, source_sequences, source_sequence_lengths, target_sequences, target_sequence_lengths):
        """
        Returns logits from the teacher model instead of just the target sequence IDs.
        """
        with torch.no_grad():
            target_sequences = self.teacher_model(source_sequences, target_sequences, source_sequence_lengths, target_sequence_lengths) # (N, max_target_sequence_pad_length_this_batch, vocab_size)

        # get length of target sequences
        mask = (torch.argmax(target_sequences, dim=-1) == self.tgt_bpe_model.eos_id).cumsum(dim=-1) == 1

        # calculate lengths
        target_sequence_lengths = mask.sum(dim=-1)

        # if EOS is never found, use maximum sequence length for this batch
        target_sequence_lengths[target_sequence_lengths == 0] = target_sequences.size(1)

        return target_sequences, target_sequence_lengths
