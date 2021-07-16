import json
from pathlib import Path

from tqdm import tqdm
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from typing import List


def gen_rogec(path_models: List[str], path_file_test: str, path_tokenizer: str, path_log: str, configs_generation):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    with open(path_file_test, 'r') as input_file:
        data = json.load(input_file)

    for path_model in path_models:
        name_model = path_model.split('/')[-1]
        model = TFGPT2LMHeadModel.from_pretrained(path_model)

        for name_config, config in configs_generation.items():
            results = []

            for example in tqdm(data):
                text = f'incorrect: {example["wrong"]} correct: '
                input_tokens = tokenizer.encode(text, return_tensors='tf')
                len_tokens_inp = len(input_tokens[0])

                predict_tokens = model.generate(input_tokens, max_length=1024, pad_token_id=tokenizer.eos_token_id,
                                                **config)[0][len_tokens_inp:]
                predict_text = tokenizer.decode(predict_tokens).replace('<|endoftext|>', '')

                results.append({
                    'input': example['wrong'],
                    'original': example['correct'],
                    'predict': predict_text
                })

            Path(f'{path_log}/{name_model}').mkdir(parents=True, exist_ok=True)
            with open(f'{path_log}/{name_model}/{name_model}-{name_config}.json', 'w+') as output_file:
                json.dump(results, output_file, ensure_ascii=False, indent=4)

        del model


if __name__ == '__main__':
    configs_gen = {
        'greedy': {},
        # 'beam-search-4': {'num_beams': 4, 'early_stopping': True},
        # 'beam-search-8': {'num_beams': 8, 'early_stopping': True}
    }
    path = 'tokenizer'
    model = ['model/large']  # 'model/large', 'model/medium',

    gen_rogec(model, 'dataset/test.json', path, 'generate', configs_gen)
