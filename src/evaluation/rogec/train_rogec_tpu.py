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

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int64)
    gpt2_layer = gpt2(input_layer)
    output_layer = tf.keras.layers.Lambda(lambda x: x.logits)(gpt2_layer)

    model_train = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu_name', type=str, help='Name for tpu for to run', required=True)
    parser.add_argument('--total_size_train', type=int, required=True)

    args = parser.parse_args()

    block_size = 140
    path_ds_train = ''
    path_ds_dev = ''
    total_size_train = args.total_size_train
    total_size_dev = 1501
    tpu_name = args.tpu_name

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    ds_train = get_dataset_from_recorder(tf.io.gfile.glob(f'{path_ds_train}/*.tfrecord'), block_size)
    ds_dev = get_dataset_from_recorder(tf.io.gfile.glob(f'{path_ds_dev}/*.tfrecord'), block_size)

    for path_model, info in {
        '../../../model/models/base': {'batch_size': 144, 'epochs': 12},
        '../../../model/models/medium': {'batch_size': 72, 'epochs': 6},
        '../../../model/models/large': {'batch_size': 56, 'epochs': 3}
    }.items():
        name_model = path_model.split('/')[-1]
        ds_train_b = ds_train.shuffle(10_000).repeat().batch(info['batch_size'], drop_remainder=True)
        ds_dev_b = ds_dev.shuffle(10_000).repeat().batch(info['batch_size'], drop_remainder=True)

        with tpu_strategy.scope():
            model = get_model(path_model, block_size, info['batch_size'])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
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
