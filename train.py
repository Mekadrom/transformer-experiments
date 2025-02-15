from trainer.causal_trainer import CausalTrainer
from trainer.bayesian_translation_trainer import BayesianIter
from trainer.translation_trainer import TranslationTrainer
from trainer.translation_distillation_trainer import DistillationTrainer

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils

def train():

    args, unk = utils.get_args()
    print(f"using learning rate {args.lr}")

    trainer: TranslationTrainer | None = None

    if args.bayesian_iter == True:
        trainer = BayesianIter(args)
    elif args.distillation_teacher_run_name is not None:
        trainer = DistillationTrainer(args)
    elif (hasattr(args, 'causal_lm') and bool(args.causal_lm)) or (hasattr(unk, 'causal_lm') and bool(unk.causal_lm)):
        trainer = CausalTrainer(args)
    else:
        trainer = TranslationTrainer(args)

    trainer.train()

"""
This entire project is a heavily modified version of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers. 
Credit to them for the workflow and the implementation of most of the transformer model architecture code in transformer_provider.py.
Most of the changes I've made have been to enable a highly configurable set of hyperparameters and additional architecture options.
"""
if __name__ == '__main__':
    train()
