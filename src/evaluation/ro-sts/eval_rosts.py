import os
from pathlib import Path

import tensorflow as tf
from transformers import GPT2Tokenizer, TFAutoModel
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

from typing import List


def create_dataset(path_file: str, name_model: str, block_size: int, batch_size: int) -> tf.data.Dataset:
    sentences_1 = []
    sentences_2 = []
    similarities = []
    tokenizer = GPT2Tokenizer.from_pretrained(name_model)
    tokenizer.pad_token = tokenizer.eos_token

    with open(path_file, 'r') as input_file:

        for line in input_file.readlines():
            if line.strip() == '':
                continue

            similarity, sentence1, sentence2 = line.strip().split('\t')
            sentences_1.append(sentence1)
            sentences_2.append(sentence2)
            similarities.append(float(similarity) / 5)

    sentences_1 = tokenizer(sentences_1, padding='max_length', max_length=block_size, truncation=True,
                            return_tensors='tf')
    sentences_2 = tokenizer(sentences_2, padding='max_length', max_length=block_size, truncation=True,
                            return_tensors='tf')
    similarities = tf.convert_to_tensor(similarities, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((
        {'sentence1_ids': sentences_1['input_ids'], 'sentence2_ids': sentences_2['input_ids'], }, similarities))
    dataset = dataset.shuffle(10_000).batch(batch_size=batch_size, drop_remainder=True)

    return dataset


class CosineSimilarity(tf.keras.layers.Layer):

    def __init__(self, name=None, **kwarg):
        super(CosineSimilarity, self).__init__(name=name, **kwarg)

    def build(self, input_shape):
        pass

    def call(self, a: tf.Tensor, b: tf.Tensor):
        a_norm = tf.math.l2_normalize(a, axis=-1)
        b_norm = tf.math.l2_normalize(b, axis=-1)

        return tf.reduce_sum(tf.multiply(a_norm, b_norm), axis=-1)


def create_model(path_pre_trained: str, block_size: int, batch_size: int) -> tf.keras.models.Model:
    gpt2 = TFAutoModel.from_pretrained(path_pre_trained)

    sentence1_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, name='sentence1_ids',
                                            dtype=tf.int32)
    sentence2_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, name='sentence2_ids',
                                            dtype=tf.int32)

    sentence1_embedding = gpt2(sentence1_layer, return_dict=True).last_hidden_state
    sentence2_embedding = gpt2(sentence2_layer, return_dict=True).last_hidden_state

    vec_sentence1 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(sentence1_embedding)
    vec_sentence2 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(sentence2_embedding)

    output_layer = CosineSimilarity(name='similarity')(vec_sentence1, vec_sentence2)

    return tf.keras.models.Model(inputs=[sentence1_layer, sentence2_layer], outputs=[output_layer])


def _eval_model(model: tf.keras.models.Model, ds: tf.data.Dataset):
    predicts = []
    values = []

    for x, y in ds:
        predict = model(x)
        values += [float(i) for i in list(y.numpy())]  # in case of error use this i[0] :)
        predicts += list(predict.numpy())

    return spearmanr(predicts, values)[0], pearsonr(predicts, values)[0]


def eval_rosts(model: tf.keras.Model, ds_dev: tf.data.Dataset, ds_test: tf.data.Dataset, path_log: str):
    spearman_dev, pearson_dev = _eval_model(model, ds_dev)
    spearman_test, pearson_test = _eval_model(model, ds_test)

    Path(os.path.dirname(path_log)).mkdir(parents=True, exist_ok=True)
    with open(f'{path_log}', 'w+') as output_file:
        output_file.write(f'For dev: spearman: {spearman_dev}, pearson: {pearson_dev}\n')
        output_file.write(f'For test: spearman: {spearman_test}, pearson: {pearson_test}\n')


if __name__ == '__main__':
    block_size = 64
    batch_size_dev = 15  # total size: 1500
    batch_size_test = 7  # total size: 1379
    path_tokenizer = '../../../model/tokenizer'
    path_ds_dev = '../../../dataset/ro-sts/raw/RO-STS.dev.tsv'
    path_ds_test = '../../../dataset/ro-sts/raw/RO-STS.test.tsv'

    ds_dev = create_dataset(path_ds_dev, path_tokenizer, block_size, batch_size_dev)
    ds_test = create_dataset(path_ds_test, path_tokenizer, block_size, batch_size_test)

    for path_model in [
        '../../../model/evaluation/ro-sts/base',
        '../../../model/evaluation//ro-sts/medium',
        '../../../model/evaluation/ro-sts/large'
    ]:
        name_model = path_model.split('/')[-1]
        model = create_model(path_model, block_size, max(batch_size_test, batch_size_dev))

        eval_rosts(model, ds_dev, ds_test, f'../../../log/ro-sts/{name_model}.txt')
