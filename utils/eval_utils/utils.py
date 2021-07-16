import os
import json
from pathlib import Path

import tensorflow as tf

from collections import namedtuple
from typing import Callable, List

Dataset_Info = namedtuple('Dataset_Info', ['path_dataset', 'path_tokenizer', 'path_save', 'block_size', 'name_dataset'])


class BinaryF1Score(tf.keras.metrics.Metric):

    def __init__(self):
        super(BinaryF1Score, self).__init__(name='binary_f1_score')
        self._precision = tf.keras.metrics.Precision(thresholds=0.5)
        self._recall = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._precision.update_state(y_true, y_pred)
        self._recall.update_state(y_true, y_pred)

    def result(self):
        val_precision = self._precision.result()
        val_recall = self._recall.result()

        return 2 * tf.math.divide_no_nan((val_recall * val_precision), (val_recall + val_precision))

    def reset_states(self):
        self._precision.reset_states()
        self._recall.reset_states()


def int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def map_features_binary(features, output):
    return {
        'inputs': int64_feature_list(features),
        'labels': int64_feature(output)
    }


def map_features_multi_class(features, output):
    return {
        'inputs': int64_feature_list(features),
        'labels': int64_feature_list(output)
    }


def write_tf_record_wrapper(dataset: tf.data.Dataset, path_to_save_record: str, map_features: Callable):
    def serialize(feature, outputs):
        example_proto = tf.train.Example(features=tf.train.Features(feature=map_features(feature, outputs)))
        return example_proto.SerializeToString()

    def generator(dataset: tf.data.Dataset):
        for x, y in dataset:
            yield serialize(x, y)

    tf_dataset = tf.data.Dataset.from_generator(
        generator=lambda: generator(dataset),
        output_types=tf.string, output_shapes=()
    )

    if not os.path.exists(os.path.dirname(path_to_save_record)):
        Path(os.path.dirname(path_to_save_record)).mkdir(parents=True, exist_ok=True)

    writer = tf.data.experimental.TFRecordWriter(path_to_save_record)
    writer.write(tf_dataset)


def create_dataset_record(create_dataset: Callable, write_record: Callable, datasets_info: List[Dataset_Info],
                          map_features: Callable, path_to_info_file: str):
    if os.path.exists(path_to_info_file):
        with open(path_to_info_file, 'r') as input_file:
            info_dataset = json.load(input_file)
    else:
        if not os.path.exists(os.path.dirname(path_to_info_file)):
            Path(os.path.dirname(path_to_info_file)).mkdir(parents=True, exist_ok=True)
        info_dataset = {}

    for dataset_info in datasets_info:
        path_dataset = dataset_info.path_dataset
        path_tokenizer = dataset_info.path_tokenizer
        path_to_save = dataset_info.path_save
        block_size = dataset_info.block_size
        name_dataset = dataset_info.name_dataset

        dataset = create_dataset(path_dataset, path_tokenizer, block_size)
        info_dataset[name_dataset] = int(dataset.cardinality().numpy())
        write_record(dataset, path_to_save, map_features)

    with open(path_to_info_file, 'w+') as output_file:
        json.dump(info_dataset, output_file, indent=4)
