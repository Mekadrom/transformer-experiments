from utils import *
from llm.trainer.llm_trainer import LLMTrainer

def train():
    args, unk = get_args()

    print(f"using learning rate {args.lr}")

    trainer = LLMTrainer(args)

    trainer.train()

if __name__ == '__main__':
    train()
