from pathlib import Path
import json

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from transformers import TFGPT2Model, GPT2Tokenizer
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

from eval_utils.utils import BinaryF1Score


def get_dataset(path_dataset: str, path_tokenizer: str, block_size: int, task: str):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    inputs = []
    labels = []

    with open(path_dataset, 'r') as input_files:
        data = json.load(input_files)

    for example in data:
        text = f'{example["title"]} {example["content"]}'
        tokens = tokenizer.encode(text)
        rating = int(example['starRating'])

        label = rating - 1
        # hardcode, but it useful
        label = tf.keras.utils.to_categorical(label, num_classes=5, dtype='int32') if task == 'multi' else int(
            label > 2)

        inputs.append(tokens)
        labels.append(label)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post', truncating='post', maxlen=block_size)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def eval_laroseda(path_model: str, path_log: str, block_size: int, task: str):
    batch_size = 1
    path_dataset = '../../../dataset/laroseda/split'
    path_tokenizer = '../../../model/tokenizer'
    model = tf.keras.models.load_model(path_model)

    model.compile(
        loss=BinaryCrossentropy() if task == 'binary' else CategoricalCrossentropy(),
        metrics=[BinaryF1Score() if task == 'binary' else F1Score(num_classes=5, average='macro'), 'accuracy']
    )

    ds_dev = get_dataset(f'{path_dataset}/dev.json', path_tokenizer, block_size, task).shuffle(10000) \
        .batch(batch_size, drop_remainder=True)
    ds_test = get_dataset(f'{path_dataset}/test.json', path_tokenizer, block_size, task).shuffle(10000) \
        .batch(batch_size, drop_remainder=True)

    Path(path_log).mkdir(parents=True, exist_ok=True)

    _, f1_score_dev, accuracy_dev = model.evaluate(ds_dev)
    _, f1_score_test, accuracy_test = model.evaluate(ds_test)

    with open(f'{path_log}{path_model.split("/")[-1].replace(".h5", "")}.txt', 'w+') as output_file:
        output_file.write(f'F1 Score for Dev: {f1_score_dev}\n')
        output_file.write(f'Accuracy for Dev: {accuracy_dev}\n')
        output_file.write(f'F1 Score for Test: {f1_score_test}\n')
        output_file.write(f'Accuracy for Test: {accuracy_test}\n')

    del model
    del ds_dev, ds_test


if __name__ == '__main__':
    version = 'large'
    block_size = 128
    path_log = '../../../log/laroseda'
    path_model = '../../../model/evaluation/laroseda'

    # eval_laroseda(f'{path_model}/{version}/{version}-binary.h5', f'{path_log}/{version}/', block_size, 'binary')
    eval_laroseda(f'{path_model}/{version}/{version}-multi-class.h5', f'{path_log}/{version}/', block_size, 'multi')
