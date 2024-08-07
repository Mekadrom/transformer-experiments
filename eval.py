from utils import *

import argparse
import os
import torch
import youtokentome

argparser = argparse.ArgumentParser()

argparser.add_argument("--run_name", type=str, required=True)
argparser.add_argument("--model_checkpoint", type=str, default="averaged_transformer_checkpoint.pth.tar")
argparser.add_argument("--sacrebleu_in_python", action="store_true")

argparser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

args, unk = argparser.parse_known_args()

run_dir = f"runs/{args.run_name}"

src_bpe_model = youtokentome.BPE(model=os.path.join(run_dir, 'src_tokenizer.model'))

if os.path.exists(os.path.join(run_dir, 'tgt_tokenizer.model')):
    tgt_bpe_model = youtokentome.BPE(model=os.path.join(run_dir, 'tgt_tokenizer.model'))
else:
    tgt_bpe_model = src_bpe_model

checkpoint = torch.load(os.path.join(run_dir, args.model_checkpoint))
model = checkpoint['model']
model.encoder = model.encoder.to(args.encoder_device)
model.decoder = model.decoder.to(args.decoder_device)
model.eval()

# Use sacreBLEU in Python or in the command-line?
# Using in Python will use the test data downloaded in prepare_data.py
# Using in the command-line will use test data automatically downloaded by sacreBLEU...
# ...and will print a standard signature which represents the exact BLEU method used! (Important for others to be able to reproduce or compare!)
sacrebleu_in_python = args.sacrebleu_in_python

if __name__ == '__main__':
    sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python=sacrebleu_in_python)
