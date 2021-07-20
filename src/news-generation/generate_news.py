import json

import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

from sacrebleu import sentence_bleu


def generate_paragraph(index: int):
    path_dataset_test = '../../dataset/digi24/split/test.json'
    tokenizer = GPT2Tokenizer.from_pretrained('../../model/tokenizer')
    model = TFGPT2LMHeadModel.from_pretrained('../../model/news/paragraph/large')

    with open(path_dataset_test, 'r') as input_file:
        data = json.load(input_file)

    example = data[index]
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


def generate_title(index: int):
    path_dataset_test = '../../dataset/digi24/split/test.json'
    tokenizer = GPT2Tokenizer.from_pretrained('../../model/tokenizer')
    model = TFGPT2LMHeadModel.from_pretrained('../../model/news/title/medium')

    with open(path_dataset_test, 'r') as input_file:
        data = json.load(input_file)

    example = data[index]
    title = example['title']
    text = ' '.join(example['text'])

    inputs_text = f'Text: {text} Title: '
    inputs_token = tokenizer.encode(inputs_text, return_tensors='tf')
    text_predict = model.generate(inputs_token, max_length=1024, no_repeat_ngram_size=2)[0][len(inputs_token[0]):]
    text_predict = tokenizer.decode(text_predict)
    text_predict = text_predict.replace('<|endoftext|>', '')

    bleu = sentence_bleu(text_predict, [title])
    print(bleu)

    print(inputs_text)
    print(text_predict)
    print(title)


if __name__ == '__main__':
    generate_title(6788)
    generate_paragraph(6788)
