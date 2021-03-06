import json

from transformers import GPT2Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split

from eval_utils.utils import write_tf_record_wrapper, create_dataset_record, Dataset_Info, int64_feature_list


def map_features_xquad(features, output):
    return {
        'inputs': int64_feature_list(features),
        'labels': int64_feature_list(output)
    }


def unpack_dataset():
    path_raw_dataset = '../../../dataset/xquad/xquad.ro.json'
    path_unpack_dataset = '../../../dataset/xquad/test.json'
    examples = []

    with open(path_raw_dataset, 'r+') as input_files:
        data = json.load(input_files)

    for i in range(len(data['data'])):
        title = data['data'][i]['title']

        for j in range(len(data['data'][i]['paragraphs'])):
            context = data['data'][i]['paragraphs'][j]['context']

            for k in range(len(data['data'][i]['paragraphs'][j]['qas'])):
                question = data['data'][i]['paragraphs'][j]['qas'][k]['question']
                assert len(data['data'][i]['paragraphs'][j]['qas'][k]['answers']) == 1
                answer = data['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text']

                examples.append({'title': title, 'context': context, 'question': question, 'answer': answer})

    # assert len(examples) == 1190

    with open(path_unpack_dataset, 'w+') as output_file:
        json.dump(examples, output_file, ensure_ascii=False, indent=4)


def get_dataset(path_to_file: str, path_to_tokenizer: str, block_size: int) -> tf.data.Dataset:
    inputs = []
    labels = []
    tokenizer = GPT2Tokenizer.from_pretrained(path_to_tokenizer)
    padding_token_id = 50300

    with open(path_to_file, 'r') as input_file:
        data = json.load(input_file)

    for example in data:
        input_text = f'C: {example["context"]} Q: {example["question"]} A: '
        labels_text = f'{example["answer"]}<|endoftext|>'
        input_tokens = tokenizer.encode(input_text)
        labels_tokens = tokenizer.encode(labels_text)

        inputs.append(input_tokens + labels_tokens[:-1])
        labels.append([padding_token_id] * (len(input_tokens) - 1) + labels_tokens)

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=block_size, padding='post')
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=block_size, padding='post',
                                                           value=padding_token_id)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    return dataset


def split_train_dev():
    path_dataset_train = '../../../dataset/xquad/train-v1.1.json'
    tokenizer = GPT2Tokenizer.from_pretrained('../../../model/tokenizer')
    examples = []

    with open(path_dataset_train, 'r') as input_file:
        data = json.load(input_file)

    for example in data:
        text = f'C: {example["context"]} Q: {example["question"]} A: {example["answer"]} <|endoftext|>'
        tokens = tokenizer.encode(text)

        if len(tokens) - 1 > 512:
            continue

        examples.append(example)

    train, dev = train_test_split(examples, test_size=0.1, random_state=42)

    for partition in ['train', 'dev']:
        with open(f'../../../dataset/xquad/split/{partition}.json', 'w+') as output:
            json.dump(dev if partition == 'dev' else train, output, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # split_train_dev()

    block_size = 512

    data_info = [
        Dataset_Info(
            '../../../dataset/xquad/split/train.json',
            '../../../model/tokenizer',
            '../../../tf-record/xquad/train/train.tfrecord',
            block_size,
            'train'
        ),

        Dataset_Info(
            '../../../dataset/xquad/split/dev.json',
            '../../../model/tokenizer',
            '../../../tf-record/xquad/dev/dev.tfrecord',
            block_size, 'dev'
        )
    ]

    create_dataset_record(
        get_dataset, write_tf_record_wrapper, data_info, map_features_xquad,
        '../../../tf-record/xquad/info.json'
    )
