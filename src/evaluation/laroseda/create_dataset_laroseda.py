import json

import tensorflow as tf
from transformers import GPT2Tokenizer

from eval_utils import map_features_binary, map_features_multi_class, Dataset_Info, write_tf_record_wrapper, \
    create_dataset_record


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
        label = tf.keras.utils.to_categorical(label, num_classes=5, dtype='int32') if task == 'multi' else int(
            label > 2)

        inputs.append(tokens)
        labels.append(label)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post', truncating='post', maxlen=block_size)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


if __name__ == '__main__':
    path_dataset = '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/tf-record/'
    block_size = 128

    data_info_binary = [
        Dataset_Info(
            '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/split/train.json',
            '/home/mihai/Documents/GPT2Model/tokenizer',
            '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/binary/train.tfrecord',
            block_size,
            'train'
        ),

        Dataset_Info(
            '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/split/dev.json',
            '/home/mihai/Documents/GPT2Model/tokenizer',
            '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/binary/dev.tfrecord',
            block_size,
            'dev'
        ),

        Dataset_Info(
            '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/split/test.json',
            '/home/mihai/Documents/GPT2Model/tokenizer',
            '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/binary/test.tfrecord',
            block_size,
            'test'
        )
    ]

    data_info_multi = [
        Dataset_Info(
            '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/split/train.json',
            '/home/mihai/Documents/GPT2Model/tokenizer',
            '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/multi-class/train.tfrecord',
            block_size,
            'train'
        ),

        Dataset_Info(
            '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/split/dev.json',
            '/home/mihai/Documents/GPT2Model/tokenizer',
            '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/multi-class/dev.tfrecord',
            block_size,
            'dev'
        ),

        Dataset_Info(
            '/home/mihai/Documents/EvalGPT2/dataset/LaRoSeDa/split/test.json',
            '/home/mihai/Documents/GPT2Model/tokenizer',
            '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/multi-class/test.tfrecord',
            block_size,
            'test'
        )
    ]

    create_dataset_record(
        lambda x, y, z: create_dataset_laroseda(x, y, z, task='binary'), write_tf_record_wrapper, data_info_binary,
        map_features_binary, '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/binary/info.json'
    )

    """
    create_dataset_record(
            lambda x, y, z: create_dataset_laroseda(x, y, z, task='multi'), write_tf_record_wrapper, data_info_multi,
            map_features_multi_class, '/home/mihai/Documents/EvalGPT2/tf-record/LaRoSeDa/multi-class/info.json'
    )
    """
