import base_trainer

class LLMTrainer(base_trainer.BaseTrainer):
    def __init__(self, args):
        super(LLMTrainer, self).__init__(args, 'llm')
