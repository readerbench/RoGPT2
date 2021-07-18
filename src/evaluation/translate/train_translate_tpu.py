import argparse

import datasets
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def create_dataset_v1(path_to_file: str, path_tokenizer: str, block_size: int, mode: str) -> tf.data.Dataset:
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    ds = datasets.load_dataset('wmt16', 'ro-en', split=path_to_file)
    inputs = []
    labels = []
    for example in ds['translation']:
        if mode == 'ro-en':
            text = f'Romanian: {example["ro"]} English: {example["en"]} <|endoftext|>'
        else:
            text = f'English: {example["en"]} Romanian: {example["ro"]} <|endoftext|>'

        tokens = tokenizer.encode(text)
        if len(tokens) > block_size:
            continue

        inputs.append(tokens[:-1])
        labels.append(tokens[1:])

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=block_size, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=block_size, padding='post')

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def create_dataset_v2(path_to_file: str, path_tokenizer: str, block_size: int, mode: str) -> tf.data.Dataset:
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    ds = datasets.load_dataset('wmt16', 'ro-en', split=path_to_file)
    inputs = []
    labels = []
    for example in ds['translation']:
        if mode == 'ro-en':
            inp_text = f'Romanian: {example["ro"]} English: '
            inp_tokens = tokenizer.encode(inp_text)
            lab_text = f'{example["en"]}<|endoftext|>'
            lab_tokens = tokenizer.encode(lab_text)

        else:
            inp_text = f'English: {example["en"]} Romanian: '
            inp_tokens = tokenizer.encode(inp_text)
            lab_text = f'{example["ro"]}<|endoftext|>'
            lab_tokens = tokenizer.encode(lab_text)

        tokens = inp_tokens + lab_tokens
        if len(tokens) > block_size:
            continue

        inputs.append(tokens[:-1])
        labels.append(([50300] * (len(inp_tokens) - 1)) + lab_tokens)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=block_size, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=block_size, padding='post', value=50300)

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def get_model(pre_trained_gpt2: str, block_size: int, batch_size: int) -> tf.keras.models.Model:
    gpt2 = TFGPT2LMHeadModel.from_pretrained(pre_trained_gpt2)

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int64)
    gpt2_layer = gpt2(input_layer)
    output_layer = tf.keras.layers.Lambda(lambda x: x.logits)(gpt2_layer)

    model_train = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model_train


@tf.function
def loss_translate(y_true: tf.Tensor, y_pred: tf.Tensor, padding_ids: int = 50300) -> tf.Tensor:
    def replace_padding(vector, equal):
        return tf.where(equal, tf.zeros_like(vector), vector)

    equal = tf.math.equal(y_true, padding_ids)

    y_true = replace_padding(y_true, equal)  # replace paddings with 0

    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.math.logical_not(equal)
    mask = tf.cast(mask, dtype=loss.dtype)

    loss = loss * mask

    return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='Mode for translation: en-ro or ro-en')
    parser.add_argument('--tpu_name', type=str, required=True, help='Name of Tpu for training')
    parser.add_argument('--strategy', type=str, required=True, help='v1 or v2')

    args = parser.parse_args()

    path_tokenizer = '../../../model/tokenizer'
    block_size = 250
    mode = args.mode
    tpu_name = args.tpu_name
    create_dataset = create_dataset_v1 if args.strategy == 'v1' else create_dataset_v2

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)

    ds_train = create_dataset('train', path_tokenizer, block_size, mode)
    ds_dev = create_dataset('validation', path_tokenizer, block_size, mode)

    for model_path, info in {
        '../../../model/models/base': {'batch_size': 144, 'epochs': 12},
        '../../../model/models/medium': {'batch_size': 40, 'epochs': 8},
        '../../../model/models/large': {'batch_size': 24, 'epochs': 4}
    }.items():
        name_model = model_path.split('/')[-1]
        ds_train_batched = ds_train.batch(info['batch_size'], drop_remainder=True)
        ds_dev_batched = ds_dev.batch(info['batch_size'], drop_remainder=True)

        with tpu_strategy.scope():
            model = get_model(model_path, block_size, info['batch_size'])
            loss = SparseCategoricalCrossentropy(from_logits=True) if args.strategy == 'v1' else loss_translate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics='accuracy',
                loss=loss
            )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, min_lr=4e-6,
                                                         min_delta=5e-3)

        history = model.fit(
            ds_train_batched, epochs=info['epochs'],
            validation_data=ds_dev_batched,
            callbacks=[early_stop, reduce_lr],
        )

        model.layers[1].save_pretrained(
            f'../../../model/evaluation/translate/{args.strategy}/{name_model}/{name_model}-{mode}')

        with open(f'log/{model_path.split("/")[-1]}-{mode}.txt', 'w+') as output_file:
            for metric, values in history.history.items():
                output_file.write(f'{metric}: {values} \n')

        del model
        del ds_train_batched, ds_dev_batched
