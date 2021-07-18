import itertools
import json
import re

import tensorflow as tf
from transformers import GPT2Tokenizer

from eval_utils.utils import Dataset_Info, write_tf_record_wrapper, int64_feature_list, create_dataset_record


def select_train():
    def _clean_line(line: str) -> str:
        line = line.strip()
        line = re.sub('\t', '', line)
        line = re.sub('\n', '', line)
        line = re.sub('==*.', '', line)
        line = re.sub(' +', ' ', line)
        line = line[1:] if line[0] == ' ' else line

        return line

    path_raw = '../../../dataset/rogec/raw/train.txt'
    number_select = 3_000_000
    examples = []
    tokenizer = GPT2Tokenizer.from_pretrained('../../../model/tokenizer')
    with open(path_raw, 'r') as input_file:
        count = 0

        for line1, line2 in itertools.zip_longest(*[input_file] * 2):
            line1 = _clean_line(line1.strip())
            line2 = _clean_line(line2.strip())

            tokens_line1 = tokenizer.encode(line1)
            tokens_line2 = tokenizer.encode(line2)

            if 4 <= len(tokens_line1) < 64 and 4 <= len(tokens_line2) < 64:
                examples.append({'correct': line1, 'wrong': line2})
                count += 1

            if count == number_select:
                break

    with open('../../../dataset/rogec/selectect/train_original.json', 'w+') as output_file:
        json.dump(examples, output_file, ensure_ascii=False, indent=4)


def create_dataset(path_to_files: str, path_tokenizer: str, block_size: int) -> tf.data.Dataset:
    inputs = []
    labels = []
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)

    with open(path_to_files, 'r') as input_file:
        data = json.load(input_file)

    for example in data:

        text_inp = f'incorrect: {example["wrong"]} correct: '
        text_lab = f'{example["correct"]}<|endoftext|>'

        text_inp_tokens = tokenizer.encode(text_inp)
        text_lab_tokens = tokenizer.encode(text_lab)
        text_tokens = text_inp_tokens + text_lab_tokens

        if len(text_tokens) > block_size + 1:
            continue

        inputs.append(text_tokens[:-1])
        labels.append(text_tokens[1:])

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=block_size, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=block_size, padding='post')

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def clean_dataset():
    def _clean_line(line: str) -> str:
        line = line.strip()
        line = re.sub('\t', '', line)
        line = re.sub('\n', '', line)
        line = re.sub('==*.', '', line)
        line = re.sub(' +', ' ', line)
        line = line[1:] if line[0] == ' ' else line

        return line

    path_base = '../../../dataset/rogec/selectect/'

    for partition in ['train', 'dev', 'test']:
        path_file = f'{path_base}{partition}.json'
        clean_data = []

        with open(path_file, 'r') as input_file:
            data = json.load(input_file)

        for example in data:
            clean_data.append({'correct': _clean_line(example['correct']), 'wrong': _clean_line(example['wrong'])})

        with open(path_file, 'w+') as output_file:
            json.dump(clean_data, output_file, ensure_ascii=False, indent=4)


def map_features_rogec(features, outputs):
    return {
        'inputs': int64_feature_list(features),
        'labels': int64_feature_list(outputs),
    }



if __name__ == '__main__':
    select_train()
    data_info = [
        Dataset_Info(
            'dataset/train.json',
            '../../../model/tokenizer',
            '../../../tf-record/rogec/train_smaller.tfrecord'
            , 140,
            'train'
        ),

        Dataset_Info(
            '../../../dataset/train_v2.json',
            '../../../model/tokenizer',
            '../../../tf-record/rogec/train_1gb/train_1gb.tfrecord',
            140,
            'train-v2'
        ),
        Dataset_Info(
            'dataset/dev.json',
            '../../../model/tokenizer',
            '../../../tf-record/rogec/dev.tfrecord',
            140,
            'dev'
        )
    ]

    create_dataset_record(
        create_dataset, write_tf_record_wrapper, data_info, map_features_rogec, '../../../tf-record/rogec/info.json'
    )
