import json
from transformers import GPT2Tokenizer

if __name__ == '__main__':
    path_dataset = '../../../dataset/xquad/test.json'

    len_tokens = {}
    max_len = 0
    min_len = 10000
    sum_all = 0
    count_all = 0
    count_remove = 0
    tokenizer = GPT2Tokenizer.from_pretrained('../../../model/tokenizer')
    dataset = json.load(open(path_dataset, 'r'))


    for example in dataset:
        text = f'C: {example["context"]} Q: {example["question"]} A: {example["answer"]} <|endoftext|>'
        tokens = tokenizer.encode(text)
        if len(tokens) > 512:
            count_remove += 1
        len_tokens[len(tokens)] = len_tokens.get(len(tokens), 0) + 1
        max_len = max(max_len, len(tokens))
        min_len = min(min_len, len(tokens))
        sum_all += len(tokens)
        count_all += 1

    print('All: ', count_all)
    print('Max: ', max_len)
    print('Min: ', min_len)
    print('Average:', sum_all / count_all)
    print('Remove: ', count_remove)
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