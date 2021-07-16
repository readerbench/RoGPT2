import argparse
from pathlib import Path

from transformers import GPT2Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.normalizers import NFKC

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--path_corpus', type=str, required=True)
    parser.add_argument('--path_save_tokenizer', type=str, required=True)
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.normalizer = NFKC()

    list_dirs_train = [str(x) for x in Path(args.path_corpus).glob("**/*/*") if x.is_dir()]
    files = []
    for director in list_dirs_train:
        files += [str(x) for x in Path(director).glob('**/*.txt') if x.is_file()]

    tokenizer.train(files, vocab_size=args.vocab_size, special_tokens=[
        '<|endoftext|>'
    ])
    Path(args.path_save_tokenizer).mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(args.path_save_tokenizer)
    # convert to GPT2 tokenizer
    GPT2Tokenizer.from_pretrained(args.path_save_tokenizer).save_pretrained(args.path_save_tokenizer)
