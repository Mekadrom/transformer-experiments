from datasets import load_dataset
from tqdm import tqdm

import os
import re
import regex

dataset = load_dataset('wmt14', 'de-en', cache_dir='data/actual')

allowed_chars = set(' Ω0123456789aàáăǎâäåãāąǻæbcćçčĉdďđeèéěêėëēęfgğģhiıìíǐîi̇ïījkķlļľĺłmnńňñņoòóŏôöőõōøœpqrŕřsśšŝşștťţðuùúűǔûůūüųµvwŵxyýzżźžAÀÁĂǍÂÄÅÃĀĄǺÆBCĆÇČĈDĎĐEÈÉĚÊĖËĒĘFGĞĢHIIÌÍǏÎİÏĪJKĶLĻĽĹŁMNŃŇÑŅOÒÓŎÔÖŐÕŌØŒPQRŔŘSŚŠŜŞȘTŤŢÐUÙÚŰǓÛŮŪÜŲVWŴXYÝZŻŹŽß.,!?»«"\';:()[]{}<>+±≤-*°÷\\/=@#¢$¥€§£%&|~`^_\n')
not_allowed_chars = set('̈†›▸→♪√舣�йĕΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩωþ¹₂²³абвгдэеєжꙃꙁиіклмнопрстꙋоуфхѡцчшщъъіьѣꙗѥюѫѭѧѩѯѱѳѵҁАБВГДЭЕЄЖꙂꙀИІКЛМНОПРСТꙊОУФХѠЦЧШЩЪЪІЬѢꙖѤЮѪѬѦѨѮѰѲѴҀ©®™№¼½¾¿¤¶¡¸·´¨\r')

replacements: dict = {
    '-': ['­', '–', '—', '―', '−', '‑', '¬'],
    '...': ['…'],
    '': ['· ', '• ', '● ', '·', '•', '●', '​', '＊'],
    ' ': ['\t'],
    "'": ["’", "‘", '‚', '‛', '´', 'ʻ', 'ª', '′'],
    '"': ['“', '”', '„', '‟', '˝', '″'],
    '°': ['˚', 'º'],
    '(': ['（'],
    ')': ['）'],
    # 'o': ['ο'],
    # 'A': ['Α'],
    # 'B': ['Β'],
    # 'E': ['Ε'],
    # 'I': ['Ι'],
    # 'K': ['Κ'],
    # 'M': ['Μ'],
    # 'N': ['Ν'],
    # 'O': ['Ο'],
    # 'P': ['Ρ'],
    # 'T': ['Τ'],
    # 'ß': ['β'],
}

def is_valid(s):
    if len(s) == 0:
        return False
    
    # not valid if all whitespace or all punctuation
    if re.match(r'^[\s.,!?»«"\';:()[]{}<>+-*°º÷\\/=@#$§£%&|~`^_]+$', s):
        return False

    valid = True
    for c in s:
        if re.match(r'\s', c):
            continue
        if c not in allowed_chars:
            valid = False
            if c not in not_allowed_chars:
                replacement = "\033[4m" + c + "\033[0m"
                print(f"Invalid character: \"{c}\" in \"{s.replace(c, replacement)}\"")
            break

    # not valid if sentence contains any chinese characters
    if regex.search(r'[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]', s):
        valid = False

    # not valid if sentence contains any thai characters
    if re.search(r'[\u0e00-\u0e7f]', s):
        valid = False

    # not valid if sentence contains any arabic characters
    if re.search(r'[\u0600-\u06ff]', s):
        valid = False

    # not valid if sentence contains any hindi characters
    if re.search(r'[\u0900-\u097f]', s):
        valid = False

    return valid

def save_to_file(data, src_filename, tgt_filename):
    with open(os.path.join('data', 'actual', src_filename), 'w', encoding='utf-8') as src_file, open(os.path.join('data', 'actual', tgt_filename), 'w', encoding='utf-8') as tgt_file:
        for example in tqdm(data):
            en = example['translation']['en'].replace("' s ", "'s ").replace(' , ', ', ')
            de = example['translation']['de']

            for k, v in replacements.items():
                for r in v:
                    en = en.replace(r, k)
                    de = de.replace(r, k)

            # remove leading whitespace
            en = en.strip()
            de = de.strip()

            # remove leading punctuation (.!?) if it exists
            if en[0] in '.!?-':
                en = en[1:].strip()

            if de[0] in '.!?-':
                de = de[1:].strip()

            # regex replacements
            en = re.sub(r'\s*…', '...', en)
            de = re.sub(r'\s*…', '...', de)

            if is_valid(en) and is_valid(de):
                src_file.write(en + '\n')
                tgt_file.write(de + '\n')
            # else:
            #     print(f"Invalid example: {en} ||| {de}")

# Save train, validation, and test sets
save_to_file(dataset['train'], 'train.src', 'train.tgt')
save_to_file(dataset['validation'], 'valid.src', 'valid.tgt')
save_to_file(dataset['test'], 'test.src', 'test.tgt')
