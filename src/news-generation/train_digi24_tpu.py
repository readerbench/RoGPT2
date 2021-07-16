import argparse

import numpy as np
import tensorflow as tf
from transformers import TFGPT2LMHeadModel

from typing import List


def get_dataset_from_recorder(path_local: List[str], batch_size: int, block_size: int):
    raw_dataset = tf.data.TFRecordDataset(path_local)

    feature_description = {
        'inputs': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        'labels': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        return example['inputs'], example['labels']

    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.shuffle(10_000).repeat().batch(batch_size, drop_remainder=True)

    return parsed_dataset


def get_model(pre_trained_gpt2: str, block_size: int, batch_size: int) -> tf.keras.models.Model:
    gpt2 = TFGPT2LMHeadModel.from_pretrained(pre_trained_gpt2)

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int64)
    gpt2_layer = gpt2(input_layer)
    output_layer = tf.keras.layers.Lambda(lambda x: x.logits)(gpt2_layer)

    model_train = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model_train


class SaveBestCallBack(tf.keras.callbacks.Callback):
    def __init__(self, path_to_save: str, val_monitor: str, mode: str = 'min'):
        super(SaveBestCallBack, self).__init__()
        self.path_to_save = path_to_save
        self.val_monitor = val_monitor
        self.mode = mode
        self.best_value = np.Inf if mode == 'min' else np.NINF

    def on_epoch_end(self, epoch, logs=None):
        value_current = logs[self.val_monitor]
        if self.mode == 'min':
            if self.best_value > value_current:
                self.best_value = value_current
                self.model.layers[1].save_pretrained(self.path_to_save)

        if self.mode == 'max':
            if self.best_value < value_current:
                self.best_value = value_current
                self.model.layers[1].save_pretrained(self.path_to_save)


def run(strategy, path_model: str, block_size: int, batch_size: int, epochs: int, path_ds_train, path_ds_dev,
        path_save_best: str, path_save: str, path_log: str, total_size_train: int, total_size_dev: int
        ):
    with strategy.scope():
        model = get_model(path_model, batch_size=batch_size, block_size=block_size)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics='accuracy',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
    
    ds_train = get_dataset_from_recorder(tf.io.gfile.glob(f'{path_ds_train}/*.tfrecord'), batch_size=batch_size, block_size=block_size)
    ds_dev = get_dataset_from_recorder(tf.io.gfile.glob(f'{path_ds_dev}/*.tfrecord'), batch_size=batch_size, block_size=block_size)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, min_lr=4e-6, min_delta=5e-3)
    save_model = SaveBestCallBack(path_save_best, val_monitor='val_loss', mode='min')

    history = model.fit(
        ds_train, epochs=epochs, steps_per_epoch=total_size_train // batch_size, callbacks=[reduce_lr, save_model],
        validation_data=ds_dev, validation_steps=total_size_dev // batch_size
    )

    model.layers[1].save_pretrained(path_save)

    with open(path_log, 'w+') as output_file:
        for metric, values in history.history.items():
            output_file.write(f'{metric}: {values} \n')

    del model
    del ds_train, ds_dev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model', help='Path where is model')
    parser.add_argument('--tpu_name', help='Name for tpu for to run')
    parser.add_argument('--batch_size', help='Batch size')
    args = parser.parse_args()

    block_size = 512
    path_dataset_paragraph_train = 'gs://pub-mihai-niculescu-gpt2/eval/digi24-new/paragraph/train'
    path_dataset_paragraph_dev = 'gs://pub-mihai-niculescu-gpt2/eval/digi24-new/paragraph/dev'
    total_size_paragraph_train = 190645
    total_size_paragraph_dev = 10604
    path_dataset_title_train = 'gs://pub-mihai-niculescu-gpt2/eval/digi24-new/title/train'
    path_dataset_title_dev = 'gs://pub-mihai-niculescu-gpt2/eval/digi24-new/title/dev'
    total_size_title_train = 193992
    total_size_title_dev = 10779

    tpu_name = args.tpu_name
    batch_size = int(args.batch_size)
    name_model = args.path_model.split('/')[-1]

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    run(
        tpu_strategy, args.path_model, block_size, batch_size, 7, path_dataset_paragraph_train,
        path_dataset_paragraph_dev, f'model/paragraph/{name_model}-best', f'model/paragraph/{name_model}',
        f'logs/paragraph/{name_model}.txt', total_size_paragraph_train, total_size_paragraph_dev
    )

    run(
        tpu_strategy, args.path_model, block_size, batch_size, 7, path_dataset_title_train, path_dataset_title_dev,
        f'model/title/{name_model}-best', f'model/title/{name_model}', f'logs/title/{name_model}.txt',
        total_size_title_train, total_size_title_dev
    )
