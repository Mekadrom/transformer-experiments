from model_provider import TranslationTransformerModelProvider

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

if __name__ == '__main__':
    args, unk = utils.get_args()

    run_dir = os.path.join('runs', args.run_name)
    if not os.path.exists(run_dir):
        # exit with error
        print(f"run directory {run_dir} does not exist")
        exit(1)

    src_bpe_model, tgt_bpe_model = utils.load_tokenizers(os.path.join('runs', args.tokenizer_run_name))

    print('Initializing model...')

    if os.path.exists(os.path.join(run_dir, 'transformer_checkpoint.pth.tar')):
        checkpoint = torch.load(os.path.join(run_dir, 'transformer_checkpoint.pth.tar'))

        model = TranslationTransformerModelProvider().provide_transformer(args, utils.vocab_size(args, src_bpe_model), utils.vocab_size(args, tgt_bpe_model), tie_embeddings=tgt_bpe_model==src_bpe_model)

        model.load_state_dict(checkpoint['model'].state_dict())

        if 'optimizer' in checkpoint:
            optimizer = checkpoint['optimizer']
        else:
            optimizer = None

    utils.print_model(model)

    model.encoder = model.encoder.to(args.encoder_device)
    model.decoder = model.decoder.to(args.decoder_device)

    bleu_score = utils.sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python=True)

    print(f"BLEU 13a tokenization, cased score: {bleu_score.score}")
