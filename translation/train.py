from translation.trainer.bayesian_translation_trainer import BayesianIter
from translation.trainer.translation_trainer import TranslationTrainer
from translation.trainer.translation_distillation_trainer import DistillationTrainer

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

def train():
    args, unk = utils.get_args()

    print(f"using learning rate {args.lr}")

    trainer: TranslationTrainer | None = None

    if args.bayesian_iter == True:
        print("Performing Bayesian optimization.")
        bayesian_iter = BayesianIter(args)

        bayesian_iter.train()
    else:
        if args.distillation_teacher_run_name is not None:
            trainer = DistillationTrainer(args)
        else:
            trainer = TranslationTrainer(args)

        if args.prune_mode == 'train-prune':
            trainer.train()
            utils.prune_model(trainer.model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
        elif args.prune_mode == 'train-prune-retrain':
            trainer.train()

            model = trainer.model

            args.n_steps = args.prune_retrain_n_steps
            args.warmup_steps = args.prune_retrain_warmup_steps

            for i in range(args.n_prune_retrains):
                utils.prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
                trainer.train(model_name_prefix=f"pruned_{i}_")
                model = trainer.model
        elif args.prune_mode == 'only-prune':
            src_bpe_model, tgt_bpe_model = utils.load_tokenizers(os.path.join('runs', args.run_name))
            model, _, _ = utils.load_translation_checkpoint_or_generate_new(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, checkpoint_model_name='averaged_transformer_checkpoint.pth.tar')
            utils.prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
            utils.sacrebleu_evaluate(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python=True)
        else:
            trainer.train()

"""
Credit to them for the workflow and the implementation of most of the transformer model architecture code in transformer_provider.py.
This entire project is a heavily modified version of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers. 
Most of the changes I've made have been to enable a highly configurable set of hyperparameters and additional architecture options.
"""
if __name__ == '__main__':
    train()
