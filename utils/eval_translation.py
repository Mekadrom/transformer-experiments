import argparse
import os
import torch
import utils
import yaml_dict
import youtokentome

argparser = argparse.ArgumentParser()

argparser.add_argument("--run_name", type=str, required=True)
argparser.add_argument("--config", type=str, required=True)
argparser.add_argument("--model_checkpoint", type=str, default="averaged_transformer_checkpoint.pth.tar")
argparser.add_argument("--sacrebleu_in_python", action="store_true")

argparser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

argsparser_args, unk = argparser.parse_known_args()

# convert unk list to dict
unk = {unk[i][2:]: unk[i + 1] for i in range(0, len(unk), 2)}

if len(unk) > 0:
    print(f"unknown arguments: {unk}")

args = yaml_dict.load_yaml(argsparser_args.config, unk)
setattr(args, 'run_name', argsparser_args.run_name)
setattr(args, 'model_checkpoint', argsparser_args.model_checkpoint)
setattr(args, 'sacrebleu_in_python', argsparser_args.sacrebleu_in_python)
setattr(args, 'device', argsparser_args.device)

run_dir = f"runs/{args.run_name}"
tokenizer_run_dir = f"runs/{args.tokenizer_run_name}"

src_bpe_model = youtokentome.BPE(model=os.path.join(tokenizer_run_dir, 'src_tokenizer.model'))

if os.path.exists(os.path.join(tokenizer_run_dir, 'tgt_tokenizer.model')):
    tgt_bpe_model = youtokentome.BPE(model=os.path.join(tokenizer_run_dir, 'tgt_tokenizer.model'))
else:
    tgt_bpe_model = src_bpe_model

checkpoint = torch.load(os.path.join(run_dir, args.model_checkpoint))
model = checkpoint['model']
model.eval()

# Use sacreBLEU in Python or in the command-line?
# Using in Python will use the test data downloaded in prepare_data.py
# Using in the command-line will use test data automatically downloaded by sacreBLEU...
# ...and will print a standard signature which represents the exact BLEU method used! (Important for others to be able to reproduce or compare!)
sacrebleu_in_python = args.sacrebleu_in_python

if __name__ == '__main__':
    utils.sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python=sacrebleu_in_python)
