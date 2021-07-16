from pathlib import Path
import json

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from transformers import TFGPT2Model, GPT2Tokenizer

from eval_utils.utils import BinaryF1Score


def get_dataset(path_dataset: str, path_tokenizer: str, block_size: int, name_label: str):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    inputs = []
    labels = []

    with open(path_dataset, 'r') as input_files:
        data = json.load(input_files)

    for k, v in data.items():
        tokens = tokenizer.encode(v['text'])
        label = int(v[name_label]) - 1
        label = tf.keras.utils.to_categorical(label, num_classes=6) if name_label == 'category' else label

        inputs.append(tokens)
        labels.append(label)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post', truncating='post', maxlen=block_size)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def eval_dialect(path_model: str, path_log: str, block_size: int):
    batch_size = 31
    model = tf.keras.models.load_model(path_model)
    model.compile(
        loss=BinaryCrossentropy(),
        metrics=[BinaryF1Score()]
    )
    path_dataset = '../../../dataset/moroco/json/'
    path_tokenizer = '../../../model/tokenizer'

    ds_dev = get_dataset(f'{path_dataset}/dev/merge_all.json', path_tokenizer, block_size, 'dialect').shuffle(10000) \
        .batch(batch_size, drop_remainder=True)
    ds_test = get_dataset(f'{path_dataset}/test/merge_all.json', path_tokenizer, block_size, 'dialect').shuffle(10000) \
        .batch(batch_size, drop_remainder=True)

    Path(path_log).mkdir(parents=True, exist_ok=True)

    _, f1_score_dev = model.evaluate(ds_dev)
    _, f1_score_test = model.evaluate(ds_test)

    with open(f'{path_log}{path_model.split("/")[-1].replace(".h5", "")}.txt', 'w+') as output_file:
        output_file.write(f'F1 Score for Dev: {f1_score_dev}\n')
        output_file.write(f'F1 Score for Test: {f1_score_test}\n')

    del model
    del ds_dev, ds_test


def eval_cross_language(path_model: str, path_log: str, block_size: int):
    model = tf.keras.models.load_model(path_model)
    model.compile(
        loss=CategoricalCrossentropy(),
        metrics=[tfa.metrics.F1Score(num_classes=6, average='macro')]
    )

    path_dataset = '../../../dataset/moroco/json/'
    path_tokenizer = '../../../model/tokenizer'

    ds_dev_ro = get_dataset(f'{path_dataset}/dev/romanian.json', path_tokenizer, block_size, 'category').shuffle(10000) \
        .batch(1, drop_remainder=True)
    ds_test_ro = get_dataset(f'{path_dataset}/test/romanian.json', path_tokenizer, block_size, 'category'). \
        shuffle(10000).batch(1, drop_remainder=True)
    ds_dev_md = get_dataset(f'{path_dataset}/dev/moldavian.json', path_tokenizer, block_size, 'category').shuffle(10000) \
        .batch(1, drop_remainder=True)
    ds_test_md = get_dataset(f'{path_dataset}/test/moldavian.json', path_tokenizer, block_size, 'category').\
        shuffle(10000).batch(1, drop_remainder=True)

    _, f1_score_dev_ro = model.evaluate(ds_dev_ro)
    _, f1_score_test_ro = model.evaluate(ds_test_ro)
    _, f1_score_dev_md = model.evaluate(ds_dev_md)
    _, f1_score_test_md = model.evaluate(ds_test_md)

    Path(path_log).mkdir(parents=True, exist_ok=True)
    with open(f'{path_log}{path_model.split("/")[-1].replace(".h5", "")}.txt', 'w+') as output_file:
        output_file.write(f'F1 Score for Dev Ro: {f1_score_dev_ro}\n')
        output_file.write(f'F1 Score for Test Ro: {f1_score_test_ro}\n')
        output_file.write(f'F1 Score for Dev Md: {f1_score_dev_md}\n')
        output_file.write(f'F1 Score for Test Md: {f1_score_test_md}\n')

    del model
    del ds_dev_ro, ds_test_ro
    del ds_dev_md, ds_test_md


if __name__ == '__main__':
    block_size = 275
    version = 'large'
    path_model = '../../../model/evaluation/moroco'
    path_log = '../../../log/moroco'

    #eval_dialect(f'{path_model}/{version}/{version}-dialect.h5', f'{path_log}/{version}/', block_size)
    #eval_cross_language(f'{path_model}/{version}/{version}-ro.h5', f'{path_log}/{version}/', block_size)
    eval_cross_language(f'{path_model}/{version}/{version}-md.h5', f'{path_log}/{version}/', block_size)
