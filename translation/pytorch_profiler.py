if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity

    import argparse
    import os
    import torch
    import translation.trainer.translation_trainer
    import utils
    
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--run_name", type=str, required=True)
    argparser.add_argument("--config_file_path", type=str, required=True)
    argparser.add_argument("--model_checkpoint", type=str, default="transformer_checkpoint.pth.tar")
    argparser.add_argument("--profile_training", action="store_true")

    argparser_args, argparser_unk = argparser.parse_known_args()

    args, unk = utils.get_args()

    args.run_name = argparser_args.run_name

    args.__setattr__('model_checkpoint', argparser_args.model_checkpoint)
    args.__setattr__('profile_training', argparser_args.profile_training)

    src_bpe_model, tgt_bpe_model = utils.load_tokenizers(os.path.join('runs', args.tokenizer_run_name))
    model, _ = utils.load_translation_checkpoint_or_generate_new(args, os.path.join('runs', args.run_name), src_bpe_model, tgt_bpe_model, checkpoint_model_name=args.model_checkpoint)

    model.encoder = model.encoder.to(args.encoder_device)
    model.decoder = model.decoder.to(args.decoder_device)

    TRANSLATE_STRINGS = [
        "The quick brown fox jumps over the lazy dog.",
        "I am the walrus.",
        "I am the eggman.",
        "I am the walrus, goo goo g'joob.",
        "I am he as you are he as you are me and we are all together.",
    ]

    # Use torch.profiler to profile the execution

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Track both CPU and CUDA (GPU) activities
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),  # Define profiler schedule
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join('runs', args.run_name)),  # Export results to TensorBoard
            record_shapes=True,  # Record input shapes
            profile_memory=True,  # Profile memory usage
            with_stack=True  # Record stack info
        ) as prof:
        if args.profile_training:
            pass
        else:
            for s in TRANSLATE_STRINGS:
                with record_function("model_inference"):
                    best, all = utils.beam_search_translate(args, s, model, src_bpe_model, tgt_bpe_model)
                    print(f'"{s}" -> "{best}"')
                prof.step()  # Next step in profiling

    # Optionally print the results to the console
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
