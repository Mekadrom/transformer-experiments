import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

argparser = argparse.ArgumentParser()

argparser.add_argument('--model_checkpoint', type=str, default='transformer_checkpoint.pth.tar')

argparser_args, argparser_unk = argparser.parse_known_args()

args, unk = utils.get_args()

setattr(args, "model_checkpoint", argparser_args.model_checkpoint)

run_dir = os.path.join('translation', 'runs', args.run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

src_bpe_model, tgt_bpe_model = utils.load_yttm_tokenizers(os.path.join('runs', args.tokenizer_run_name))

model, optimizer = utils.load_translation_checkpoint_or_generate_new(args, run_dir, utils.vocab_size(args, src_bpe_model), utils.vocab_size(args, tgt_bpe_model), tie_embeddings=src_bpe_model==tgt_bpe_model, checkpoint_model_name=args.model_checkpoint)

utils.average_checkpoints(args.start_step, optimizer, run_dir, args.early_stop_checkpoint_window, model_name_prefix='step')
