import argparse
from pathlib import Path

from transformers import TFGPT2LMHeadModel
import tensorflow as tf

from typing import List


def get_dataset_from_recorder(path_local: List[str], batch_size: int, block_size: int):
    assert len(path_local) != 0, f'Non-file for dataset'
    raw_dataset = tf.data.TFRecordDataset(path_local)

    feature_description = {
        'inputs': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
        'labels': tf.io.FixedLenFeature([block_size], tf.int64, default_value=[0] * block_size),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        return example['inputs'][:512], example['labels'][:512]

    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset = parsed_dataset.shuffle(10_000).repeat().batch(batch_size)

    return parsed_dataset


def get_model(pre_trained_gpt2: str, block_size: int, batch_size: int) -> tf.keras.models.Model:
    gpt2 = TFGPT2LMHeadModel.from_pretrained(pre_trained_gpt2)

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int64)
    gpt2_layer = gpt2(input_layer)
    output_layer = tf.keras.layers.Lambda(lambda x: x.logits)(gpt2_layer)

    model_train = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model_train


class SaveModelCallBack(tf.keras.callbacks.Callback):
    def __init__(self, name_model: str, log_director: str, model_save_dir: str):
        super(SaveModelCallBack, self).__init__()

        self._name_model = name_model
        self._log_director = log_director
        self._model_save_dir = model_save_dir

    def on_train_begin(self, logs=None):
        Path(self._log_director).mkdir(parents=True, exist_ok=True)
        self._file_log = open(f'{self._log_director}/log_{self._name_model}.txt', 'a+')

    def on_epoch_end(self, epoch, logs=None):
        self._file_log.write(f'Epoch: {epoch}, Loss: {logs["loss"]}, Accuracy: {logs["accuracy"]}\n')
        self.model.layers[1].save_pretrained(
            f'{self._model_save_dir}/check_point_{self._name_model}/{epoch}_{logs["loss"]}')

    def on_train_end(self, logs=None):
        self._file_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--path_model', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    parser.add_argument('--block_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--tpu_name', type=str, required=True)
    args = parser.parse_args()

    address_dataset = ''  # we used tf-record store on google cloud platform storage
    block_size = args.block_size
    batch_size = args.batch_size
    total_size = 3494714 if block_size == 1024 else 7019631  # total size
    tpu_name = args.tpu_name
    path_log = 'log'
    name_model = args.path_model.split('/')[-1]
    model_save_dir_check_point = '/'.join(args.path_save.split('/'))

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    with tpu_strategy.scope():
        model = get_model('base', block_size, batch_size)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            metrics='accuracy',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

    model_check_point = SaveModelCallBack(name_model, path_log, model_save_dir_check_point)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, min_lr=4e-6)

    files = tf.io.gfile.glob(f'{address_dataset}/*.tfrecord')
    tf_dataset = get_dataset_from_recorder(files, batch_size, block_size)

    model.fit(tf_dataset,
              epochs=args.epochs,
              steps_per_epoch=total_size // batch_size,
              callbacks=[model_check_point, reduce_lr]
    )

    model.layers[1].save_pretrained(args.path_save)
