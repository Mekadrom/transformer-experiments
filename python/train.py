from utils import *
from trainers.classic_trainer import ClassicTrainer
from trainers.distillation_trainer import DistillationTrainer

import os
import torch.backends.cudnn as cudnn

"""
This entire project is a heavily modified version of https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers. 
Credit to them for the workflow and the implementation of most of the transformer model architecture code in transformer_provider.py.
Most of the changes I've made have been to enable a highly configurable set of hyperparameters and additional architecture options.
"""
if __name__ == '__main__':
    args, unk = get_args()
    cudnn.benchmark = args.cudnn_benchmark

    print(f"using learning rate {args.lr}")

    trainer: ClassicTrainer | None = None

    if args.distillation_teacher_run_name is not None:
        trainer = DistillationTrainer(args)
    else:
        trainer = ClassicTrainer(args)

    if args.prune_mode == 'train-prune':
        trainer.train()
        prune_model(trainer.model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
    elif args.prune_mode == 'train-prune-retrain':
        trainer.train()

        model = trainer.model

        args.n_steps = args.prune_retrain_n_steps
        args.warmup_steps = args.prune_retrain_warmup_steps

        for i in range(args.n_prune_retrains):
            prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
            trainer.train(model_name_prefix=f"pruned_{i}_")
            model = trainer.model
    elif args.prune_mode == 'only-prune':
        src_bpe_model, tgt_bpe_model = load_tokenizers(os.path.join('runs', args.run_name))
        model, _, _ = load_checkpoint_or_generate_new(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, checkpoint_model_name='averaged_transformer_checkpoint.pth.tar')
        prune_model(model, args.prune_heads_amount, args.prune_heads_norm, args.prune_ffn_amount, args.prune_ffn_norm, args.prune_structured, args.prune_type)
        sacrebleu_evaluate(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, model, device=args.device, sacrebleu_in_python=True)
    else:
        trainer.train()
