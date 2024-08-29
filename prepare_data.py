from tqdm import tqdm

import argparse
import codecs
import os
import youtokentome

def main(args):
    run_dir = os.path.join('runs', args.run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    src_tokenizer_file = os.path.join(run_dir, 'src_tokenizer.model')
    tgt_tokenizer_file = os.path.join(run_dir, 'tgt_tokenizer.model')

    if args.shared_vocab:
        tgt_tokenizer_file = src_tokenizer_file

    if not os.path.exists(src_tokenizer_file):
        print(f"Training source tokenizer and saving to {src_tokenizer_file}")
        data = os.path.join('..', 'data', 'actual', 'train.src')
        if args.shared_vocab:
            data = os.path.join(run_dir, 'temp.srctgt')
            with open(data, 'w', encoding='utf-8') as f:
                with open(os.path.join('..', 'data', 'actual', 'train.src'), 'r', encoding='utf-8') as source:
                    with open(os.path.join('..', 'data', 'actual', 'temp.tgt'), 'r', encoding='utf-8') as target:
                        for line in source:
                            if args.prepend_cls_token:
                                f.write('<CLS> ' + line.strip() + '\n')
                            else:
                                f.write(line.strip() + '\n')
                        f.write('\n')
                        for line in target:
                            if args.prepend_cls_token:
                                f.write('<CLS> ' + line.strip() + '\n')
                            else:
                                f.write(line.strip() + '\n')
        else:
            data = os.path.join('..', 'data', 'actual', 'temp.src')
            tgt_data = os.path.join('..', 'data', 'actual', 'temp.tgt')
            # used to append <CLS> before every line
            with open(data, 'w', encoding='utf-8') as f:
                with open(os.path.join('..', 'data', 'actual', 'train.src'), 'r', encoding='utf-8') as source:
                    for line in source:
                        if args.prepend_cls_token:
                            f.write('<CLS> ' + line.strip() + '\n')
                        else:
                            f.write(line.strip() + '\n')
            with open(tgt_data, 'w', encoding='utf-8') as f:
                with open(os.path.join('..', 'data', 'actual', 'train.tgt'), 'r', encoding='utf-8') as target:
                    for line in target:
                        if args.prepend_cls_token:
                            f.write('<CLS> ' + line.strip() + '\n')
                        else:
                            f.write(line.strip() + '\n')
            
        youtokentome.BPE.train(data=data, vocab_size=args.src_vocab_size, model=src_tokenizer_file)

        if args.shared_vocab and os.path.exists(os.path.join(run_dir, 'temp.srctgt')):
            os.remove(os.path.join(run_dir, 'temp.srctgt'))

    if not os.path.exists(tgt_tokenizer_file) and not args.shared_vocab:
        print(f"Training target tokenizer and saving to {tgt_tokenizer_file}")
        youtokentome.BPE.train(data=os.path.join('..', 'data', 'actual', 'temp.tgt'), vocab_size=args.tgt_vocab_size, model=tgt_tokenizer_file)

    src_bpe_model = youtokentome.BPE(model=src_tokenizer_file)
    tgt_bpe_model = youtokentome.BPE(model=tgt_tokenizer_file) if not args.shared_vocab else src_bpe_model

    def fix_set(file_name):
        print('\nRe-reading single files...')
        with codecs.open(os.path.join('..', 'data', 'actual', f"{file_name}.src"), 'r', encoding='utf-8') as f:
            source = f.read().split('\n')
        with codecs.open(os.path.join('..', 'data', 'actual', f"{file_name}.tgt"), 'r', encoding='utf-8') as f:
            target = f.read().split('\n')

        # Filter
        print('\nFiltering...')
        pairs = list()
        for src, tgt in tqdm(zip(source, target), total=len(source)):
            src_tok = src_bpe_model.encode(src, output_type=youtokentome.OutputType.ID)
            tgt_tok = tgt_bpe_model.encode(tgt, output_type=youtokentome.OutputType.ID)
            len_src_tok = len(src_tok)
            len_tgt_tok = len(tgt_tok)
            if args.min_length < len_src_tok < args.max_length and args.min_length < len_tgt_tok < args.max_length and 1. / args.max_length_ratio <= len_tgt_tok / len_src_tok <= args.max_length_ratio:
                pairs.append((src, tgt))
            else:
                continue
        print('\nNote: %.2f per cent of en-de pairs were filtered out based on sub-word sequence length limits.' % (100. * (len(source) - len(pairs)) / len(source)))

        # Rewrite files
        source, target = zip(*pairs)
        print('\nRe-writing filtered sentences to single files...')
        with codecs.open(os.path.join(run_dir, f"{file_name}.src"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(source))
        with codecs.open(os.path.join(run_dir, f"{file_name}.tgt"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(target))

    fix_set('train')
    fix_set('val')
    fix_set('test')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--run_name', type=str, required=True)

    argparser.add_argument('--src_vocab_size', type=int, default=32000)
    argparser.add_argument('--tgt_vocab_size', type=int, default=32000)
    argparser.add_argument('--shared_vocab', action='store_true')
    argparser.add_argument('--max_length', type=int, default=150)
    argparser.add_argument('--min_length', type=int, default=3)
    argparser.add_argument('--max_length_ratio', type=float, default=1.5)

    args, unk = argparser.parse_known_args()

    main(args)
