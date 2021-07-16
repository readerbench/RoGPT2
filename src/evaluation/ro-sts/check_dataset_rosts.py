from transformers import GPT2Tokenizer

if __name__ == '__main__':
    len_tokens = {}
    count_all = 0
    limit = 40
    under = 0
    over = 0
    tokenizer = GPT2Tokenizer.from_pretrained('../../../model/tokenizer')
    path_dataset = '../../../dataset/ro-sts/raw'

    for file in [str(x) for x in Path(path_dataset).glob("**/*.tsv")]:

        with open(file, 'r') as input_file:

            for line in input_file.readlines():
                if line.strip() == '':
                    continue

                similarity, sentence1, sentence2 = line.strip().split('\t')

                tokens_sen1 = tokenizer.encode(sentence1)
                tokens_sen2 = tokenizer.encode(sentence2)

                if len(tokens_sen1) <= limit:
                    under += 1
                else:
                    over += 1

                if len(tokens_sen2) <= limit:
                    under += 1
                else:
                    over += 1

                len_tokens[len(tokens_sen1)] = len_tokens.get(len(tokens_sen1), 0) + 1
                len_tokens[len(tokens_sen2)] = len_tokens.get(len(tokens_sen2), 0) + 1

                count_all += 2

    print(under, over)
    count_max = int(0.95 * count_all)
    count = 0
    len_total = 0
    for k, v in dict(sorted(len_tokens.items(), key=lambda x: x[1], reverse=True)).items():
        if count_max < count + v:
            break

        count += v
        len_total += k * v

    print('Average for 80%: ', len_total / count)
    print(dict(sorted(len_tokens.items(), key=lambda x: x[1], reverse=True)))
