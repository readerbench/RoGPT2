import json
import argparse
import tensorflow as tf
from transformers import GPT2Tokenizer
from threading import Thread

from eval_utils.utils import Dataset_Info, write_tf_record_wrapper, int64_feature_list, create_dataset_record


def create_dataset_paragraph(path_to_files: str, path_tokenizer: str, block_size: int) -> tf.data.Dataset:
    inputs = []
    labels = []
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)

    with open(path_to_files, 'r') as input_file:
        data = json.load(input_file)

    for example in data:
        title = example['title']
        paragraphs = example['text']

        if len(paragraphs) < 2:
            continue

        news_text = ' '.join(paragraphs[:-1])
        last_paragraph = paragraphs[-1]

        text_news = f'Text: {title} {news_text} '
        text_last_paragraph = f'Continuation: {last_paragraph} <|endoftext|>'
        news_tokens = tokenizer.encode(text_news)
        last_paragraph_tokens = tokenizer.encode(text_last_paragraph)

        if len(last_paragraph_tokens) > int(0.7 * (block_size + 1)):
            continue

        if len(news_tokens) + len(last_paragraph_tokens) > block_size + 1:
            skip_tokens = len(news_tokens) + len(last_paragraph_tokens) - (block_size + 1)
            news_tokens = news_tokens[:-skip_tokens]

        tokens = news_tokens + last_paragraph_tokens
        assert len(tokens) <= block_size + 1, f'Wrong len of input and label, length is: {len(tokens)}'

        inputs.append(tokens[:-1])
        labels.append(tokens[1:])

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=block_size, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=block_size, padding='post')

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def create_dataset_title(path_to_files: str, path_tokenizer: str, block_size: int) -> tf.data.Dataset:
    inputs = []
    labels = []
    tokenizer = GPT2Tokenizer.from_pretrained(path_tokenizer)

    with open(path_to_files, 'r') as input_file:
        data = json.load(input_file)

    for example in data:
        title = example['title']
        paragraphs = example['text']
        news_text = ' '.join(paragraphs)

        text_news = f'Text: {news_text} '
        text_title = f'Title: {title} <|endoftext|>'
        news_tokens = tokenizer.encode(text_news)
        title_tokens = tokenizer.encode(text_title)

        if len(title_tokens) > int(0.7 * (block_size + 1)):
            continue

        if len(news_tokens) + len(title_tokens) > block_size + 1:
            skip_tokens = len(news_tokens) + len(title_tokens) - (block_size + 1)
            news_tokens = news_tokens[:-skip_tokens]

        tokens = news_tokens + title_tokens
        assert len(tokens) <= block_size + 1, f'Wrong len of input and label, length is: {len(tokens)}'

        inputs.append(tokens[:-1])
        labels.append(tokens[1:])

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=block_size, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=block_size, padding='post')

    return tf.data.Dataset.from_tensor_slices((inputs, labels))


def map_features_digi24(features, outputs):
    return {
        'inputs': int64_feature_list(features),
        'labels': int64_feature_list(outputs),
    }


if __name__ == '__main__':
    path_file = '/home/mihai/Documents/EvalGPT2/dataset/digi24/split/train.json'
    path_tokenizer = '/home/mihai/Documents/GPT2Model/tokenizer'

    ds = create_dataset_paragraph(path_file, path_tokenizer, 512)
    print(f'For paragraph: {ds.cardinality()}')

    ds = create_dataset_title(path_file, path_tokenizer, 512)
    print(f'For title: {ds.cardinality()}')


    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_dataset', help='Name of file for dataset')
    args = parser.parse_args()

    name_dataset = args.name_dataset
    path_dataset = f'/home/mihai/Documents/EvalGPT2/dataset/digi24/split/{name_dataset}.json'
    path_save = '/home/mihai/Documents/EvalGPT2/tf-record/digi24/'
    path_tokenizer = '/home/mihai/Documents/GPT2Model/tokenizer'
    block_size = 512

    data_info_paragraph = [
        Dataset_Info(path_dataset, path_tokenizer, f'{path_save}paragraph/{name_dataset}.tfrecord', block_size,
                     name_dataset)
    ]

    data_info_title = [
        Dataset_Info(path_dataset, path_tokenizer, f'{path_save}title/{name_dataset}.tfrecord', block_size,
                     name_dataset)
    ]

    create_dataset_record(
        create_dataset, write_tf_record_wrapper,  data_info_paragraph, map_features_digi24,
        '/home/mihai/Documents/EvalGPT2/tf-record/digi24/paragraph/info.json'
    )

    create_dataset_record(
        create_dataset, write_tf_record_wrapper, data_info_title, map_features_digi24,
        '/home/mihai/Documents/EvalGPT2/tf-record/digi24/title/info.json'
    )
    """