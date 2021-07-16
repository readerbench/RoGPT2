import json
from pathlib import Path

from tqdm import tqdm
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from typing import List


def gen_xquad(path_models: List[str], path_file_test: str, path_tokenizer: str, path_log: str, configs_generation):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)

    with open(path_file_test, 'r') as input_file:
        data = json.load(input_file)

    for path_model in path_models:
        name_model = path_model.split('/')[-1]
        model = TFGPT2LMHeadModel.from_pretrained(path_model)

        for name_config, config in configs_generation.items():
            results = []

            for example in tqdm(data):
                text = f'C: {example["context"]} Q: {example["question"]} A: '
                input_tokens = tokenizer.encode(text, return_tensors='tf')
                len_tokens_inp = len(input_tokens[0])

                predict_tokens = model.generate(input_tokens, max_length=1024, pad_token_id=tokenizer.eos_token_id,
                                                **config)[0][len_tokens_inp:]
                predict_text = tokenizer.decode(predict_tokens).replace('<|endoftext|>', '')

                results.append({
                    'context': example["context"],
                    'question': example["question"],
                    'original': example['answer'],
                    'predict': predict_text
                })

            Path(f'{path_log}/{name_model}').mkdir(parents=True, exist_ok=True)
            with open(f'{path_log}/{name_model}/{name_model}-{name_config}.json', 'w+') as output_file:
                json.dump(results, output_file, ensure_ascii=False, indent=4)

        del model


if __name__ == '__main__':
    configs_gen = {
        'greedy': {},
        'beam-search-4': {'num_beams': 4, 'early_stopping': True},
        'beam-search-8': {'num_beams': 4, 'early_stopping': True},
    }

    path_tokenizer = '../../../tokenizer'
    model = [
        '../../../model/evaluation/xquad/normal/base',
        '../../../model/evaluation/xquad/normal/medium',
        '../../../model/evaluation/xquad/normal/large',
    ]
    model_translate = [
        '../../../model/evaluation/xquad/translate/base-v1',
        '../../../model/evaluation/xquad/translate/medium-v1',
        '../../../model/evaluation/xquad/translate/large-v1',
        '../../../model/evaluation/xquad/translate/large-v2'
    ]

    gen_xquad(model, '../../../dataset/xquad/test.json', path_tokenizer, '../../../generate/xquad/normal', configs_gen)
    gen_xquad(model, '../../../dataset/xquad/test.json', path_tokenizer, '../../../generate/xquad/translate', configs_gen)
