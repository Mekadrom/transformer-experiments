import argparse
import os
import utils

argparser = argparse.ArgumentParser()

argparser.add_argument('--model_checkpoint', type=str, default='transformer_checkpoint.pth.tar')

argparser_args, argparser_unk = argparser.parse_known_args()

args, unk = utils.get_args()

args.__setattr__('model_checkpoint', argparser_args.model_checkpoint)

run_dir = os.path.join('runs', args.run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

src_bpe_model, tgt_bpe_model = utils.load_tokenizers(os.path.join('runs', args.tokenizer_run_name))

model, optimizer = utils.load_checkpoint_or_generate_new(args, run_dir, src_bpe_model=src_bpe_model, tgt_bpe_model=tgt_bpe_model, vae_model=False, checkpoint_model_name=args.model_checkpoint)

utils.average_checkpoints(args.start_epoch, optimizer, args.run_name, args.early_stop_num_latest_checkpoints_to_avg, model_name_prefix='step')
