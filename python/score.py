from utils import *

import argparse
import os
import torch.backends.cudnn as cudnn

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--base_run_name', type=str, required=True)
    argparser.add_argument('--sacrebleu_score_model_name', type=str, default=None)

    argparser.add_argument('--device', type=str, default='cuda:0')
    cudnn.benchmark = False

    args, unk = argparser.parse_known_args()

    # args.__setattr__('sacrebleu_interrupted', False)

    if len(unk) > 0:
        print(f"unknown arguments: {unk}")

    run_dir = os.path.join('runs', args.base_run_name)

    if not os.path.exists(run_dir):
        # exit with error
        print(f"run directory {run_dir} does not exist")
        exit(1)

    src_bpe_model, tgt_bpe_model = load_tokenizers(run_dir)

    model, _, _ = load_checkpoint_or_generate_new(args, run_dir, src_bpe_model=src_bpe_model, tgt_bpe_model=tgt_bpe_model, checkpoint_model_name=args.sacrebleu_score_model_name)

    print_model(model)

    model.to(args.device)

    bleu_score= sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python=True)

    print(f"BLEU score: {bleu_score}")
