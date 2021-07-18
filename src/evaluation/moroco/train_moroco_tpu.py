import argparse

import math

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow_addons.metrics import F1Score
from transformers import TFGPT2Model

from typing import List


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


def get_dataset_from_recorder_generic(path_local: List[str], batch_size: int, feature_description):
    raw_dataset = tf.data.TFRecordDataset(path_local)

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        return example['inputs'], example['labels']

    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.shuffle(10_000).repeat().batch(batch_size, drop_remainder=True)

    return parsed_dataset


def get_dataset_from_recorder_dialect(path_local: List[str], batch_size: int, block_size: int):
    feature_description = {
        'inputs': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        'labels': tf.io.FixedLenFeature([1], tf.int64, default_value=0)
    }

    return get_dataset_from_recorder_generic(path_local, batch_size, feature_description)


def get_dataset_from_recorder_category(path_local: List[str], batch_size: int, block_size: int):
    num_category = 6

    feature_description = {
        'inputs': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        'labels': tf.io.FixedLenFeature([num_category], tf.int64, default_value=[0] * num_category)
    }

    return get_dataset_from_recorder_generic(path_local, batch_size, feature_description)


def get_model(path_pretrained: str, block_size: int, batch_size: int, number_category: int):
    gpt2 = TFGPT2Model.from_pretrained(path_pretrained)

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int32)
    embeddings = gpt2.layers[0](input_layer).last_hidden_state
    docs_embedding = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(embeddings)
    if number_category == 2:
        output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(docs_embedding)
    else:
        output_layer = tf.keras.layers.Dense(units=number_category, activation='softmax')(docs_embedding)

    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model


def get_callbacks(val_monitor: str):
    early_stop = EarlyStopping(monitor=val_monitor, mode='max', patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor=val_monitor, mode='max', patience=2, min_lr=1e-6, min_delta=5e-3,
                                  factor=1 / math.exp(1))

    return [early_stop, reduce_lr]


def log_result(name_log_file: str, history_train, evaluate):
    with open(f'log/{name_log_file}.txt', 'w+') as output_file:
        for metric, values in history_train.history.items():
            output_file.write(f'{metric}: {values} \n')

        for msg, value in evaluate.items():
            output_file.write(f'{msg}: {value}\n')


def _get_total_dataset_task(name_task):
    """
    :param name_task: name task, this can be : dialect, ro, md
    :return: total size of dataset fo train, dev, train
    """

    if name_task == 'dialect':
        return 21719, 5921, 5924

    if name_task == 'ro':
        return 11751, 3205, 3205

    if name_task == 'md':
        return 9968, 2716, 2719

    return None, None, None


def _total_dataset_eval(name_task):
    """
    :param name_task: name task, this can be :  ro, md
    :return: total size of dataset fo dev, train for the other dataset
    """

    if name_task == 'ro':
        return 2716, 2719

    if name_task == 'md':
        return 3205, 3205

    return None, None


def run_task(tpu_strategy, block_size: int, task: str):
    total_size_train, total_size_dev, total_size_test = _get_total_dataset_task(task)
    total_size_eval_dev, total_size_eval_test = _total_dataset_eval(task)

    path_ds_train = f'gs://pub-mihai-niculescu-gpt2/eval/moroco/{task}/train'
    path_ds_dev = f'gs://pub-mihai-niculescu-gpt2/eval/moroco/moroco/{task}/dev'
    path_ds_test = f'gs://pub-mihai-niculescu-gpt2/eval/moroco/moroco/{task}/test'

    opus_task = 'md' if task == 'ro' else 'ro'
    path_eval_dev = None if task == 'dialect' else f'gs://pub-mihai-niculescu-gpt2/eval/moroco/moroco/{opus_task}/dev'
    path_eval_test = None if task == 'dialect' else f'gs://pub-mihai-niculescu-gpt2/eval/moroco/moroco/{opus_task}/test'

    num_category = 2 if task == 'dialect' else 6
    monitor_val = 'val_binary_f1_score' if task == 'dialect' else 'val_f1_score'

    for path_model, info in {
        '../../../model/models/base': {'batch_size': 128, 'epochs': 30},
        '../../../model/models/medium': {'batch_size': 40, 'epochs': 30},
        '../../../model/models/large': {'batch_size': 24, 'epochs': 30}
    }.items():
        batch_size = info['batch_size']
        name_model = path_model.split('/')[-1]
        get_dataset = get_dataset_from_recorder_dialect if task == 'dialect' else get_dataset_from_recorder_category

        files_train = tf.io.gfile.glob(f'{path_ds_train}/*.tfrecord')
        files_dev = tf.io.gfile.glob(f'{path_ds_dev}/*.tfrecord')
        files_test = tf.io.gfile.glob(f'{path_ds_test}/*.tfrecord')

        ds_train = get_dataset(files_train, batch_size, block_size)
        ds_dev = get_dataset(files_dev, batch_size, block_size)
        ds_test = get_dataset(files_test, batch_size, block_size)

        ds_eval_dev, ds_eval_test = None, None

        if task != 'dialect':
            files_eval_dev = tf.io.gfile.glob(f'{path_eval_dev}/*.tfrecord')
            files_eval_test = tf.io.gfile.glob(f'{path_eval_test}/*.tfrecord')

            ds_eval_dev = get_dataset(files_eval_dev, batch_size, block_size)
            ds_eval_test = get_dataset(files_eval_test, batch_size, block_size)

        with tpu_strategy.scope():
            model = get_model(path_model, block_size, batch_size, number_category=num_category)
            loss = BinaryCrossentropy() if task == 'dialect' else CategoricalCrossentropy()
            f1_score = BinaryF1Score() if task == 'dialect' else F1Score(num_classes=num_category, average='macro')

            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=loss,
                metrics=[f1_score]
            )

        callbacks = get_callbacks(monitor_val)

        history = model.fit(
            ds_train, epochs=info['epochs'], steps_per_epoch=total_size_train // batch_size,
            validation_data=ds_dev, validation_steps=total_size_dev // batch_size,
            callbacks=callbacks
        )

        _, f1_score_dev = model.evaluate(ds_dev, steps=total_size_dev // batch_size)
        _, f1_score_test = model.evaluate(ds_test, steps=total_size_test // batch_size)

        result_ds_eval = {}
        if task != 'dialect':
            _, f1_score_eval_dev = model.evaluate(ds_eval_dev, steps=total_size_eval_dev // batch_size)
            _, f1_score_eval_test = model.evaluate(ds_eval_test, steps=total_size_eval_test // batch_size)

            result_ds_eval = \
                {'F1 Score Dev Ro to MD: ': f1_score_eval_dev,
                 'F1 Score Test Ro to MD: ': f1_score_eval_test} if task == 'ro' else \
                    {'F1 Score Dev MD to Ro: ': f1_score_eval_dev, 'F1 Score Test MD to Ro: ': f1_score_eval_test}

        result = {'F1 Score Dev: ': f1_score_dev, 'F1 Score Test: ': f1_score_test}
        result.update(result_ds_eval)

        model.save(f'../../../model/evaluation/moroco/{name_model}/{name_model}-{task}.h5')

        log_result(f'{name_model}-{task}', history, result)

        del model
        del ds_train, ds_dev, ds_test
        if task != 'dialect':
            del ds_eval_dev, ds_eval_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu_name', type=str, required=True, help='Name of tpu for training')
    args = parser.parse_args()

    block_size = 275
    tpu_name = args.tpu_name

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    run_task(tpu_strategy, block_size, 'dialect')
    run_task(tpu_strategy, block_size, 'ro')
    run_task(tpu_strategy, block_size, 'md')
