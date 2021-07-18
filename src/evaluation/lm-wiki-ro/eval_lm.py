import math

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel


def get_model(pre_trained_gpt2: str, block_size: int, batch_size: int) -> tf.keras.models.Model:
    gpt2 = TFGPT2LMHeadModel.from_pretrained(pre_trained_gpt2)

    input_layer = tf.keras.layers.Input(shape=block_size, batch_size=batch_size, dtype=tf.int64)
    gpt2_layer = gpt2(input_layer)
    output_layer = tf.keras.layers.Lambda(lambda x: x.logits)(gpt2_layer)

    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

    return model


def get_dataset(path_dataset: str, path_tokenizer: str, block_size: int, batch_size: int):
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    inputs = []
    labels = []

    with open(path_dataset, 'r') as input_file:
        data = input_file.read()
        tokens = tokenizer.encode(data)
        len_tokens = len(tokens)

    for i in range(0, len_tokens // (block_size + 1)):
        tokens_chunk = tokens[i * (block_size + 1): (i + 1) * (block_size + 1)]
        inputs.append(tokens_chunk[:-1])
        labels.append(tokens_chunk[1:])

    return tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(batch_size, drop_remainder=True)


if __name__ == '__main__':
    path_tokenizer = '../../../model/tokenizer'
    path_dev = '../../../dataset/wiki-ro/wiki.txt.valid'
    path_test = '../../../dataset/wiki-ro/wiki.txt.test'
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)
    block_size = 1024

    ds_dev = get_dataset(path_dev, path_tokenizer, block_size, 1)
    ds_test = get_dataset(path_test, path_tokenizer, block_size, 1)

    model = get_model('../../../model/models/base', block_size, 1)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    loss_dev = model.evaluate(ds_dev)
    loss_test = model.evaluate(ds_test)

    print("PPL dev:", math.exp(loss_dev))
    print("PPL test:", math.exp(loss_test))

    # with open(path_test, 'r') as input_file:
    #     data = input_file.read()
    #     tokens = tokenizer.encode(data)
    # model = TFGPT2LMHeadModel.from_pretrained('../../../model/models/large')
    # lls = []
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #
    # for i in range(0, len(tokens), block_size + 1):
    #     tokens_chunk = tokens[i:i + block_size + 1]
    #     inputs = tf.convert_to_tensor([tokens_chunk[:-1]])
    #     labels = tf.convert_to_tensor([tokens_chunk[1:]])
    #
    #     outputs = model(inputs)
    #     lls.append(loss(labels, outputs['logits']).numpy())
    #
    # lls_batch = tf.reduce_mean(lls).numpy()
    # print(lls_batch)
    # print(math.exp(lls_batch))
