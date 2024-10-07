from tqdm import tqdm

import argparse
import os
import random
import shutil

def split(data_path, split, n_files):
    if n_files <= 1:
        shutil.move(os.path.join(data_path, f"{split}.src"), os.path.join(data_path, f"{split}_0.src"))
        shutil.move(os.path.join(data_path, f"{split}.tgt"), os.path.join(data_path, f"{split}_0.tgt"))
        return

    print(f"Splitting {split}...")

    src_datafile = os.path.join(data_path, f"{split}.src")
    tgt_datafile = os.path.join(data_path, f"{split}.tgt")

    with open(src_datafile, 'r') as src_file, open(tgt_datafile, 'r') as tgt_file:
        src_data = src_file.readlines()
        tgt_data = tgt_file.readlines()

    shutil.move(src_datafile, src_datafile + '.split.bak')
    shutil.move(tgt_datafile, tgt_datafile + '.split.bak')

    total_data_len = len(src_data)

    split_data_len = total_data_len // n_files

    for src_line, tgt_line in tqdm(zip(src_data, tgt_data), total=split_data_len, desc=f"Splitting {split}..."):
        i = random.randint(0, n_files - 1)
        with open(os.path.join(data_path, f"{split}_{i}.src"), 'a') as src_file, open(os.path.join(data_path, f"{split}_{i}.tgt"), 'a') as tgt_file:
            src_file.write(f"{src_line}")
            tgt_file.write(f"{tgt_line}")

    print(f"Split {split} into {n_files} files.")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--data_path', type=str, default='data')
    argparser.add_argument('--split', type=str, default='train')
    argparser.add_argument('--n_files', type=int, default=1)

    args = argparser.parse_args()

    split(args.data_path, args.split, args.n_files)