import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

if __name__ == '__main__':
    args, unk = utils.get_args()

    run_dir = os.path.join('translation', 'runs', args.run_name)
    if not os.path.exists(run_dir):
        # exit with error
        print(f"run directory {run_dir} does not exist")
        exit(1)

    src_bpe_model, tgt_bpe_model = utils.load_tokenizers(os.path.join('translation', 'runs', args.tokenizer_run_name))

    model, _ = utils.load_translation_checkpoint_or_generate_new(args, run_dir, src_bpe_model.vocab_size(), tgt_bpe_model.vocab_size(), tie_embeddings=src_bpe_model==tgt_bpe_model, checkpoint_model_name=args.sacrebleu_score_model_name)

    utils.print_model(model)

    model.to(args.device)

    bleu_score = utils.sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, device=args.device, sacrebleu_in_python=True)

    print(f"BLEU 13a tokenization, cased score: {bleu_score.score}")
