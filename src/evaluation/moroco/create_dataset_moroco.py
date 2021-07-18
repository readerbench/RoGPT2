import json

import tensorflow as tf
from transformers import GPT2Tokenizer

from eval_utils import map_features_binary, map_features_multi_class, Dataset_Info, write_tf_record_wrapper, \
    create_dataset_record


def create_dataset_moroco(path_dataset: str, path_tokenizer: str, block_size: int, name_label: str):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    inputs = []
    labels = []

    with open(path_dataset, 'r') as input_files:
        data = json.load(input_files)

    for k, v in data.items():
        tokens = tokenizer.encode(v['text'])
        label = int(v[name_label]) - 1
        # hardcode, but it useful
        label = tf.keras.utils.to_categorical(label, num_classes=6,
                                              dtype='int32') if name_label == 'category' else label

        inputs.append(tokens)
        labels.append(label)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post', truncating='post', maxlen=block_size)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


if __name__ == '__main__':
    block_size = 275

    data_info_all = [
        Dataset_Info(
            '../../../dataset/moroco/json/train/merge_all.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/dialect/train.tfrecord',
            block_size,
            'train'),

        Dataset_Info(
            '../../../dataset/moroco/json/dev/merge_all.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/dialect/dev.tfrecord',
            block_size,
            'dev'),

        Dataset_Info(
            '../../../dataset/moroco/json/test/merge_all.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/dialect/test.tfrecord',
            block_size,
            'test')
    ]

    data_info_ro = [
        Dataset_Info(
            '../../../dataset/moroco/json/train/romanian.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/ro/train.tfrecord',
            block_size,
            'train'),

        Dataset_Info(
            '../../../dataset/moroco/json/dev/romanian.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/ro/dev.tfrecord',
            block_size,
            'dev'),

        Dataset_Info(
            '../../../dataset/moroco/json/test/romanian.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/ro/test.tfrecord',
            block_size,
            'test')
    ]

    data_info_md = [
        Dataset_Info(
            '../../../dataset/moroco/json/train/moldavian.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/md/train.tfrecord',
            block_size,
            'train'),

        Dataset_Info(
            '../../../dataset/moroco/json/dev/moldavian.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/md/dev.tfrecord',
            block_size,
            'dev'),

        Dataset_Info(
            '../../../dataset/moroco/json/test/moldavian.json',
            '../../../model/tokenizer',
            '../../../tf-record/moroco/md/test.tfrecord',
            block_size,
            'test')
    ]

    """
    create_dataset_record(
        lambda x, y, z: create_dataset_moroco(x, y, z, name_label='dialect'), write_tf_record_wrapper, data_info_all,
        map_features_binary, '../../../tf-record/moroco/dialect/info.json'
    )
    """

    create_dataset_record(
        lambda x, y, z: create_dataset_moroco(x, y, z, name_label='category'), write_tf_record_wrapper, data_info_ro,
        map_features_multi_class, '../../../tf-record/moroco/ro/info.json'
    )

    create_dataset_record(
        lambda x, y, z: create_dataset_moroco(x, y, z, name_label='category'), write_tf_record_wrapper, data_info_md,
        map_features_multi_class, '../../../tf-record/moroco/md/info.json'
    )
