from lm_causal.causal_trainer import CausalTrainer
from utils import utils

def train():

    args, unk = utils.get_args()
    print(f"using learning rate {args.lr}")

    trainer = CausalTrainer(args)

    trainer.train()

"""
This entire project is a heavily modified version of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers. 
Credit to them for the workflow and the implementation of most of the transformer model architecture code in transformer_provider.py.
Most of the changes I've made have been to enable a highly configurable set of hyperparameters and additional architecture options.
"""
if __name__ == '__main__':
    train()
