import json
import sys
import datasets
from transformers import GPT2Tokenizer


def creat_dataset_partition(path_raw: str, path_save: str):
    moldavian = {}
    romanian = {}
    all_samples = {}

    with open(f'{path_raw}samples.txt', 'r') as input_file:
        samples = {}

        for line in input_file.readlines():
            line = line.strip()
            id, sample = line.split('\t')
            samples[int(id)] = sample

    with open(f'{path_raw}category_labels.txt', 'r') as input_file:
        categories = {}

        for line in input_file.readlines():
            line = line.strip()
            id, category = line.split('\t')
            categories[int(id)] = category

    with open(f'{path_raw}dialect_labels.txt', 'r') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            id, label = line.split('\t')
            id, label = int(id), int(label)
            all_samples[id] = {'text': samples[id], 'category': categories[id], 'dialect': label}

            if label == 1:
                moldavian[id] = {'text': samples[id], 'category': categories[id]}
            elif label == 2:
                romanian[id] = {'text': samples[id], 'category': categories[id]}
            else:
                sys.exit(-1)

    with open(f'{path_save}romanian.json', 'w+') as output_file:
        json.dump(romanian, output_file, ensure_ascii=False, indent=4)

    with open(f'{path_save}moldavian.json', 'w+') as output_file:
        json.dump(moldavian, output_file, ensure_ascii=False, indent=4)

    with open(f'{path_save}merge_all.json', 'w+') as output_file:
        json.dump(all_samples, output_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    len_tokens = {}
    max_len = 0
    min_len = 10000
    sum_all = 0
    count_all = 0
    tokenizer = GPT2Tokenizer.from_pretrained('../../../model/tokenizer')
    dataset = datasets.load_dataset('moroco')
    print(dataset['train'][0]['sample'])

    for partition in ['train', 'test', 'validation']:
        for text in dataset[partition]['sample']:
            tokens = tokenizer.encode(text)
            len_tokens[len(tokens)] = len_tokens.get(len(tokens), 0) + 1
            max_len = max(max_len, len(tokens))
            min_len = min(min_len, len(tokens))
            sum_all += len(tokens)
            count_all += 1

    print('Max: ', max_len)
    print('Min: ', min_len)
    print('Average:', sum_all / count_all)

    # for 80% of items
    count_max = int(0.8 * count_all)
    count = 0
    len_total = 0
    for k, v in dict(sorted(len_tokens.items(), key=lambda x: x[1], reverse=True)).items():
        if count_max < count + v:
            break

        count += v
        len_total += k * v

    print('Average for 80%: ', len_total / count)
