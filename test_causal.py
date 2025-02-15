from model_provider import CausalTransformerModelProvider
from transformers import AutoTokenizer, TextStreamer

import argparse
import os
import torch
import yaml_dict

argparser = argparse.ArgumentParser()
argparser.add_argument('--run_name', type=str, required=True)
argparser.add_argument('--config', type=str, required=True)
argparser.add_argument('--model_checkpoint', type=str, default='averaged_transformer_checkpoint.pth.tar')
argparser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
argparser.add_argument('--compile_model', type=bool, default=False)
argparser.add_argument('--dynamo_cache_size_limit', type=int, default=16)
argparser.add_argument('--cudnn_benchmark', type=bool, default=False)

argparser.add_argument('--top_p', type=float, default=None)
argparser.add_argument('--top_k', type=int, default=None)
argparser.add_argument('--temperature', type=float, default=1.0)

argsparser_args, unk = argparser.parse_known_args()

# convert unk list to dict
unk = {unk[i][2:]: unk[i + 1] for i in range(0, len(unk), 2)}

if len(unk) > 0:
    print(f'unknown arguments: {unk}')

args = yaml_dict.load_yaml(argsparser_args.config, unk)
setattr(args, 'run_name', argsparser_args.run_name)
setattr(args, 'model_checkpoint', argsparser_args.model_checkpoint)
setattr(args, 'device', argsparser_args.device)
setattr(args, 'top_p', argsparser_args.top_p)
setattr(args, 'top_k', argsparser_args.top_k)
setattr(args, 'temperature', argsparser_args.temperature)
setattr(args, 'compile_model', argsparser_args.compile_model)
setattr(args, 'dynamo_cache_size_limit', argsparser_args.dynamo_cache_size_limit)
setattr(args, 'cudnn_benchmark', argsparser_args.cudnn_benchmark)

if args.cudnn_benchmark:
    torch.backends.cudnn.benchmark = True

if args.dynamo_cache_size_limit > 0:
    torch._dynamo.config.cache_size_limit = args.dynamo_cache_size_limit

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_run_name)

run_dir = os.path.join('runs', args.run_name)

if os.path.exists(os.path.join(run_dir, args.model_checkpoint)):
    checkpoint = torch.load(os.path.join(run_dir, args.model_checkpoint))
    tie_embeddings = args.tie_embeddings if hasattr(args, 'tie_embeddings') else False
    model = CausalTransformerModelProvider().provide_transformer(args, run_dir, tokenizer, tie_embeddings=tie_embeddings)
    model.load_state_dict(checkpoint['model'].state_dict())

    if args.compile_model:
        model = torch.compile(model)
else:
    raise FileNotFoundError(f"Checkpoint {args.model_checkpoint} not found in {run_dir}")

model.eval()

with torch.no_grad():
    streamer = TextStreamer(tokenizer, decoder_kwargs={'clean_up_tokenization_spaces': True})
    while True:
        input_text = input("\nEnter text: ")
        if input_text == 'exit':
            break
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(args.device)
        output_ids = model.generate(
            input_ids,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.top_p is not None or args.top_k is not None,
            temperature=args.temperature,
            max_length=args.maxlen,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id
        )
