import os
import json
import argparse
from pathlib import Path

from transformers import GPT2Tokenizer
from datasets import load_dataset, load_from_disk, Dataset
import tensorflow as tf

from typing import List, Union


def processing_dataset_save(dirs: List[str], tokenizer: GPT2Tokenizer, block_size: int, save: bool = False,
                            path_save: str = None, num_work: int = 10) -> Dataset:
    # source: https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb
    text_column_name = "text"

    """ 
    This function create local on disk dataset after this is processing using hugging face datasets
    """

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    files = []
    for director in dirs:
        path_file = Path(director).glob('**/*.txt')
        files += [str(x) for x in path_file if x.is_file()]

    dataset_test = load_dataset('text', data_files=files)

    tokenized_datasets = dataset_test.map(
        tokenize_function,
        batched=True,
        num_proc=num_work,
        remove_columns=[text_column_name]
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        num_proc=num_work,
        batched=True
    )

    if save:
        lm_datasets['train'].save_to_disk(path_save)

    return lm_datasets['train']


def write_tf_record_from_hugging_face_dataset(path_to_save_record: str, hugging_face_dataset: Union[str, Dataset],
                                              name_dataset: str, path_to_info: str, from_local: bool = False):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def serialize(feature, outputs):
        map_inputs = {
            "inputs": _int64_feature(feature),
            "labels": _int64_feature(outputs)
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=map_inputs))
        return example_proto.SerializeToString()

    def generator(hf_dataset: Dataset):
        for example in hf_dataset:
            yield serialize(example['input_ids'][:-1], example['input_ids'][1:])

    if from_local:
        lm_dataset = load_from_disk(hugging_face_dataset)
    else:
        lm_dataset = hugging_face_dataset

    total_size = len(lm_dataset)

    if os.path.exists(path_to_info):
        with open(path_to_info, 'r') as input_file:
            json_data = json.load(input_file)
    else:
        json_data = {}

    json_data[name_dataset] = total_size

    with open(path_to_info, 'w+') as output_file:
        json.dump(json_data, output_file, indent=4)

    tf_dataset = tf.data.Dataset.from_generator(
        generator=lambda: generator(lm_dataset),
        output_types=tf.string, output_shapes=()
    )

    if not os.path.exists(os.path.dirname(path_to_save_record)):
        Path(os.path.dirname(path_to_save_record)).mkdir(parents=True, exist_ok=True)

    writer = tf.data.experimental.TFRecordWriter(path_to_save_record)
    writer.write(tf_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--director_datasets', type=str, required=True)
    parser.add_argument('--path_tokenizer', type=str, required=True)
    parser.add_argument('--path_save', type=str, required=True)
    args = parser.parse_args()

    director_datasets = args.director_datasets
    dirs_dataset = [str(x) for x in Path(director_datasets).glob("*/**") if x.is_dir()]

    block_sizes = [1024, 512]  # completely [1024, 768, 512, 256, 128]
    tokenizer = GPT2Tokenizer.from_pretrained(args.path_tokenizer)

    for block_size in block_sizes:
        for dir_dataset in dirs_dataset:
            hg_dataset = processing_dataset_save(dirs=[dir_dataset], tokenizer=tokenizer, block_size=block_size + 1)
            path_to_save_record = f'{args.path_save}/{block_size}/{dir_dataset.split("/")[-1]}.tfrecord'

            write_tf_record_from_hugging_face_dataset(
                path_to_save_record=path_to_save_record, hugging_face_dataset=hg_dataset,
                name_dataset=f'{dir_dataset.split("/")[-1]}',
                path_to_info=f'{args.path_save}/{block_size}/info_datasets.json'
            )

        print(f'Finish to write dataset with block size: {block_size}')
