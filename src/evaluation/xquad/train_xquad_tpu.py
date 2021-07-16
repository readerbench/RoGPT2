import argparse

import math
from transformers import TFGPT2LMHeadModel
import tensorflow as tf
from typing import List


def get_dataset_from_recorder(path_local: List[str], block_size: int):
    raw_dataset = tf.data.TFRecordDataset(path_local)

    feature_description = {
        'inputs': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        'labels': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        return example['inputs'], example['labels']

    parsed_dataset = raw_dataset.map(_parse_function)

    return parsed_dataset


def get_model(pre_trained_gpt2: str, block_size: int, batch_size: int) -> tf.keras.models.Model:
    gpt2 = TFGPT2LMHeadModel.from_pretrained(pre_trained_gpt2)

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int32)
    gpt2_layer = gpt2(input_layer)
    output_layer = tf.keras.layers.Lambda(lambda x: x.logits)(gpt2_layer)

    model_train = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model_train


@tf.function
def loss_question_answer(y_true: tf.Tensor, y_pred: tf.Tensor, padding_ids: int = 50300) -> tf.Tensor:
    def replace_padding(vector, equal):
        return tf.where(equal, tf.zeros_like(vector), vector)

    equal = tf.math.equal(y_true, padding_ids)

    y_true = replace_padding(y_true, equal)  # replace paddings with 0

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.math.logical_not(equal)
    mask = tf.cast(mask, dtype=loss.dtype)

    loss = loss * mask

    return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)


class AccuracyQA(tf.keras.metrics.Metric):

    def __init__(self):
        super(AccuracyQA, self).__init__(name='accuracy_qa')
        self._correct = self.add_weight(name='correct', shape=1, initializer="zeros")
        self._nums_all = self.add_weight(name='nums_all', shape=1, initializer="zeros")

    def _reduce_sum(self, a):
        return tf.reduce_sum(tf.reduce_sum(a, axis=-1), axis=-1)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.int32)
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, dtype=tf.int32)

        predict = tf.argmax(y_pred, axis=-1)
        predict = tf.cast(predict, dtype=tf.int32)
        predict = predict * mask

        equal = tf.cast(tf.math.equal(predict, y_true), dtype=tf.int32)
        self._nums_all.assign_add(self._reduce_sum(mask))
        self._correct.assign_add(self._reduce_sum(equal) - self._reduce_sum(mask))

    def result(self):
        return self._correct / self._nums_all

    def reset_states(self):
        self._correct = self.add_weight(name='correct', shape=1, initializer="zeros")
        self._nums_all = self.add_weight(name='nums_all', shape=1, initializer="zeros")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu_name', help='Name for tpu for to run')
    args = parser.parse_args()

    block_size = 512
    path_ds_train = 'gs://pub-mihai-niculescu-gpt2/xquad/train'
    path_ds_dev = 'gs://pub-mihai-niculescu-gpt2/xquad/dev'
    total_size_train = 77138
    total_size_dev = 8571
    tpu_name = args.tpu_name

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    ds_train = get_dataset_from_recorder(tf.io.gfile.glob(f'{path_ds_train}/*.tfrecord'), block_size)
    ds_dev = get_dataset_from_recorder(tf.io.gfile.glob(f'{path_ds_dev}/*.tfrecord'), block_size)

    #TODO refeactorization
    #TODO refeactorization
    #TODO refeactorization
    for path_model, info in {
        '../../models/version2/base': {'batch_size': 144, 'epochs': 15},
        # '../../models/version2/medium': {'batch_size': 40, 'epochs': 10},
        # '../../models/version2/large': {'batch_size': 16, 'epochs': 9}
    }.items():
        name_model = path_model.split('/')[-1]
        ds_train_b = ds_train.shuffle(10_000).repeat().batch(info['batch_size'], drop_remainder=True)
        ds_dev_b = ds_dev.shuffle(10_000).repeat().batch(info['batch_size'], drop_remainder=True)

        with tpu_strategy.scope():
            model = get_model(path_model, block_size, info['batch_size'])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=loss_question_answer,
                metrics=['accuracy'
                         # AccuracyQA()
                         ]
            )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3,
                                                      restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, min_lr=4e-6,
                                                         min_delta=5e-3, factor=1 / math.exp(1))
        history = model.fit(
            ds_train_b, epochs=info['epochs'], steps_per_epoch=total_size_train // info['batch_size'],
            validation_data=ds_dev_b, validation_steps=total_size_dev // info['batch_size'],
            callbacks=[early_stop, reduce_lr]
        )

        model.layers[1].save_pretrained(f'model/{name_model}')

        with open(f'log/{name_model}.txt', 'w+') as output_file:
            for metric, values in history.history.items():
                output_file.write(f'{metric}: {values} \n')

        del model
        del ds_train_b, ds_dev_b
