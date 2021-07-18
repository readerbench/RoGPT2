import json

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


def generate_paragraph():
    path_dataset_test = '/home/mihai/Documents/EvalGPT2/dataset/digi24/split/test.json'
    tokenizer = GPT2Tokenizer.from_pretrained('/home/mihai/Documents/GPT2Model/tokenizer')
    model = TFGPT2LMHeadModel.from_pretrained('/home/mihai/Documents/EvalGPT2/model/digi24/model/paragraph/large')

    with open(path_dataset_test, 'r') as input_file:
        data = json.load(input_file)

    example = data[444]  # 231
    title = example['title']
    text = ' '.join(example['text'][:-1])
    true_text = example['text'][-1]
    inputs_text = f'Text: {title} {text} Continuation:'

    inputs_token = tokenizer.encode(inputs_text, return_tensors='tf')

    text_predict = model.generate(inputs_token, max_length=1024, no_repeat_ngram_size=2)[0][len(inputs_token[0]):]

    output_file = open('out_text_new_7.txt', 'w+')
    print(inputs_text, file=output_file)
    print(tokenizer.decode(text_predict), file=output_file)
    print(true_text, file=output_file)


def generate_title():
    path_dataset_test = '/home/mihai/Documents/EvalGPT2/dataset/digi24/test.json'
    tokenizer = GPT2Tokenizer.from_pretrained('/home/mihai/Documents/GPT2Model/tokenizer')
    model = TFGPT2LMHeadModel.from_pretrained('/home/mihai/Documents/EvalGPT2/model/digi24/model/title/')

    news_token = '<|news|>'
    title_token = '<|title|>'
    tokenizer.add_tokens([news_token, title_token], special_tokens=True)

    with open(path_dataset_test, 'r') as input_file:
        data = json.load(input_file)

    example = data[4243]
    title = example['title']
    text = ' '.join(example['text'])
    print(title)

    # inputs_text = f'Text: {text} Title:'
    inputs_text = f'Text: {text} Title: '
    inputs_token = tokenizer.encode(inputs_text, return_tensors='tf')
    text_predict = model.generate(inputs_token, max_length=1024, no_repeat_ngram_size=2)[0]
    output_file = open('out_text_title_news.txt', 'w+')

    print(tokenizer.decode(text_predict), file=output_file)
    print(tokenizer.decode(inputs_token[0]), file=output_file)
    print(title, file=output_file)


if __name__ == '__main__':
    path_dataset = '../../dataset/digi24/split/test.json'
    with open(path_dataset, 'r') as input_file:
        data = json.load(input_file)

    search = [
        "Poliția Română: 12 tone de articole",
        "Adrian Năstase ia în calcul",
        "Victor Ponta: Traian Băsescu se joacă cu"
    ]

    for i, example in enumerate(data):
        for s in search:
            if s in example['title'] or s in ' '.join(example['text']):
                print(i, s)

    # generate_paragraph()
