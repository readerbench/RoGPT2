import argparse
import math

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from transformers import TFGPT2Model
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from typing import List


class CustomerEarlyStop(tf.keras.callbacks.Callback):
    def __init__(self, ds_dev: tf.data.Dataset, patience=0):
        super(CustomerEarlyStop, self).__init__()

        self._ds = ds_dev
        self._patience = patience
        self._best_pearson = 0
        self._best_weights = None
        self._wait = 0

    def on_train_begin(self, logs=None):
        self._wait = 0
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        predicts = []
        values = []

        for x, y in self._ds:
            predict = model(x)
            values += [float(i[0]) for i in list(y.numpy())]
            predicts += list(predict.numpy())

        current_pearson_score = pearsonr(predicts, values)[0]

        self._wait += 1

        if current_pearson_score > self._best_pearson:
            self._best_pearson = current_pearson_score
            self._best_weights = self.model.get_weights()
            self._wait = 0
        else:
            if self._wait > self._patience:
                self.model.set_weights(self._best_weights)
                self.model.stop_training = True


class CosineSimilarity(tf.keras.layers.Layer):

    def __init__(self, name=None, **kwarg):
        super(CosineSimilarity, self).__init__(name=name, **kwarg)

    def build(self, input_shape):
        pass

    def call(self, a: tf.Tensor, b: tf.Tensor):
        a_norm = tf.math.l2_normalize(a, axis=-1)
        b_norm = tf.math.l2_normalize(b, axis=-1)

        return tf.reduce_sum(tf.multiply(a_norm, b_norm), axis=-1)


def get_dataset_from_recorder(path_local: List[str], batch_size: int, block_size: int, drop=True):
    raw_dataset = tf.data.TFRecordDataset(path_local)
    block_size = 512
    feature_description = {
        "sentence1": tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        "sentence2": tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        "attention_mask1": tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        "attention_mask2": tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        "similarity": tf.io.FixedLenFeature([1], tf.float32, default_value=0.0)
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        inputs = {
            'sentence1': example['sentence1'], 'sentence2': example['sentence2'],
            'attention_mask1': example['attention_mask1'], 'attention_mask2': example['attention_mask2']
        }

        return inputs, example['similarity']

    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.shuffle(10_000).repeat().batch(batch_size, drop_remainder=drop)

    return parsed_dataset


def get_model(path_pre_trained: str, batch_size: int, block_size: int) -> tf.keras.models.Model:
    gpt2 = TFGPT2Model.from_pretrained(path_pre_trained)

    sentence1_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, name='sentence1',
                                            dtype=tf.int32)
    sentence2_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, name='sentence2',
                                            dtype=tf.int32)

    mask1_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, name='attention_mask1', dtype=tf.int32)
    mask2_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, name='attention_mask2', dtype=tf.int32)

    sentence1_embedding = gpt2(input_ids=sentence1_layer, attention_mask=mask1_layer,
                               return_dict=True).last_hidden_state
    sentence2_embedding = gpt2(input_ids=sentence2_layer, attention_mask=mask2_layer,
                               return_dict=True).last_hidden_state


    vec_sentence1 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(sentence1_embedding)
    vec_sentence2 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(sentence2_embedding)

    output_layer = CosineSimilarity(name='similarity', )(vec_sentence1, vec_sentence2)

    return tf.keras.models.Model(inputs=[sentence1_layer, sentence2_layer, mask1_layer, mask2_layer], outputs=[output_layer])


def batched_datasets(datasets: List[tf.data.Dataset], batch_size: int) -> List[tf.data.Dataset]:
    dss = []

    for dataset in datasets:
        dss.append(dataset.shuffle(10_000).batch(batch_size, drop_remainder=True))

    return dss


def eval_model(model: tf.keras.models.Model, ds: tf.data.Dataset):
    predicts = []
    values = []

    for x, y in ds:
        predict = model(x)
        values += [float(i[0]) for i in list(y.numpy())]
        predicts += list(predict.numpy())

    print(len(values))

    return spearmanr(predicts, values)[0], pearsonr(predicts, values)[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu_name', help='Tpu name for training')
    args = parser.parse_args()

    block_size = 64 # best results for block size equal with 64 and without attention mask
    total_size_train = 5749
    total_size_dev = 1500
    total_size_test = 1379
    tpu_name = args.tpu_name

    path_ds_train = 'gs://pub-mihai-niculescu-gpt2/eval/Ro-STS/train'
    path_ds_dev = 'gs://pub-mihai-niculescu-gpt2/eval/Ro-STS/dev'
    path_ds_test = 'gs://pub-mihai-niculescu-gpt2/eval/Ro-STS/test'

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    for path_model, info in {
        '../../../model/models/base': {'batch_size': 144, 'epochs': 25},
        '../../../model/models/medium': {'batch_size': 64, 'epochs': 20},
        '../../../model/models/large': {'batch_size': 16, 'epochs': 5}
    }.items():
        name_model = path_model.split('/')[-1]
        batch_size = info['batch_size']

        files_train = tf.io.gfile.glob(f'{path_ds_train}/*.tfrecord')
        ds_train = get_dataset_from_recorder(files_train, batch_size=batch_size, block_size=block_size)
        files_dev = tf.io.gfile.glob(f'{path_ds_dev}/*.tfrecord')
        ds_dev = get_dataset_from_recorder(files_dev, batch_size=batch_size, block_size=block_size)
        
        files_test = tf.io.gfile.glob(f'{path_ds_test}/*.tfrecord')
        ds_test = get_dataset_from_recorder(files_test, batch_size=197, block_size=block_size)
        ds_dev_callBack = get_dataset_from_recorder(files_dev, batch_size=150, block_size=block_size)
        
        
        with tpu_strategy.scope():
            model = get_model(path_model, block_size=block_size, batch_size=batch_size)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                metrics=[tf.keras.metrics.MeanSquaredError()],
                loss='mse'
            )
        
        early_stop = CustomerEarlyStop(patience=3, ds_dev=ds_dev_callBack.take(10))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, min_lr=4e-6, min_delta=5e-3,
                                      factor=1 / math.exp(1))

        history = model.fit(
            ds_train, epochs=info['epochs'], steps_per_epoch=total_size_train // batch_size,
            validation_data=ds_dev, validation_steps=total_size_dev // batch_size,
            callbacks=[early_stop, reduce_lr]
        )

        model.layers[4].save_pretrained(f'../../../model/evaluation/ro-sts/{name_model}')
    
        spearman_score_dev, pearson_score_dev = eval_model(model, ds_dev_callBack.take(10))
        spearman_score_test, pearson_score_test = eval_model(model, ds_test.take(7))
        
        with open(f'log/{name_model}-mask-512.txt', 'w+') as output_file:
            output_file.write(
                f'For dev, spearman score: {spearman_score_dev}, pearson score: {pearson_score_dev}\n')
            output_file.write(
                f'For test, spearman score: {spearman_score_test}, pearson score: {pearson_score_test}\n')
        
        del ds_train, ds_test, ds_dev, ds_dev_callBack
        del model
    
