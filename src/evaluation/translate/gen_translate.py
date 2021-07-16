import json
from pathlib import Path

from tqdm import tqdm
import datasets
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

from typing import List


def eval_translate(path_models: List[str], mode: str, path_tokenizer: str, path_log: str, configs_generation,
                   strategy: str):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)

    dataset_test = datasets.load_dataset('wmt16', 'ro-en', split='test')

    for path_model in path_models:
        name_model = path_model.split('/')[-1]

        model = TFGPT2LMHeadModel.from_pretrained(path_model)

        for name_config, config in configs_generation.items():
            results = []

            for example in tqdm(dataset_test["translation"]):
                text = f'Romanian: {example["ro"]} English:' \
                    if mode == 'ro-en' else f'English: {example["en"]} Romanian:'
                text = text + ' ' if strategy == 'v2' else text
                tokens_text = tokenizer.encode(text, return_tensors='tf')
                len_tokens_inp = len(tokens_text[0])
                predict_tokens = model.generate(tokens_text, max_length=1024, pad_token_id=tokenizer.eos_token_id,
                                                **config)[0][len_tokens_inp:]
                predict_text = tokenizer.decode(predict_tokens).replace('<|endoftext|>', '')

                results.append({
                    'input': example['ro'] if mode == 'ro-en' else example['en'],
                    'original': example['en'] if mode == 'ro-en' else example['ro'],
                    'predict': predict_text
                })

            Path(f'{path_log}/{mode}/{name_model}').mkdir(parents=True, exist_ok=True)

            with open(f'{path_log}/{mode}/{name_model}/{name_model}-{name_config}-{mode}.json', 'w+') as output_file:
                json.dump(results, output_file, ensure_ascii=False, indent=4)

        del model


if __name__ == '__main__':
    strategy = 'v1'
    configs_gen = {
        'greedy': {},
        'beam-search-4': {'num_beams': 4, 'early_stopping': True},
        'beam-search-8': {'num_beams': 8, 'early_stopping': True},
    }
    path_tokenizer = '../../../model/tokenizer'
    path_log_generate = f'../../../generate/translate/{strategy}'

    model_ro_en = [
        f'../../../model/evaluation/translate/{strategy}/ro-en/large',
        f'../../../model/evaluation/translate/{strategy}/ro-en/medium',
        f'../../../model/evaluation/translate/{strategy}/ro-en/base'
    ]

    model_en_ro = [
        f'../../../model/evaluation/translate/{strategy}/en-ro/large',
        f'../../../model/evaluation/translate/{strategy}/en-ro/medium',
        f'../../../model/evaluation/translate/{strategy}/en-ro/base'
    ]

    eval_translate(model_en_ro, 'en-ro', path_tokenizer, path_log_generate, configs_gen, strategy)
    eval_translate(model_ro_en, 'ro-en', path_tokenizer, path_log_generate, configs_gen, strategy)
