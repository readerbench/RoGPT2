from pathlib import Path

from transformers import GPT2Tokenizer
import tensorflow as tf

from eval_utils.utils import float_feature, int64_feature_list, Dataset_Info, create_dataset_record, \
    write_tf_record_wrapper


def map_features_ro_sts(features, output):
    return {
        'sentence1': int64_feature_list(features['sentence1_ids']),
        'sentence2': int64_feature_list(features['sentence2_ids']),
        'attention_mask1': int64_feature_list(features['attention_mask1']),
        'attention_mask2': int64_feature_list(features['attention_mask2']),
        'similarity': float_feature(output)

    }


def create_dataset_ro_sts(path_file: str, name_tokenizer: str, block_size: int) -> tf.data.Dataset:
    sentences_1 = []
    sentences_2 = []
    similarities = []
    tokenizer = GPT2Tokenizer.from_pretrained(name_tokenizer)

    with open(path_file, 'r') as input_file:

        for line in input_file.readlines():
            if line.strip() == '':
                continue

            similarity, sentence1, sentence2 = line.strip().split('\t')
            sentences_1.append(sentence1)
            sentences_2.append(sentence2)
            similarities.append(float(similarity) / 5)

    sentences_1 = tokenizer(sentences_1, padding='max_length', max_length=block_size, return_tensors='tf')
    sentences_2 = tokenizer(sentences_2, padding='max_length', max_length=block_size, return_tensors='tf')
    similarities = tf.convert_to_tensor(similarities, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((
        {'sentence1_ids': sentences_1['input_ids'], 'sentence2_ids': sentences_2['input_ids'],
         'attention_mask1': sentences_1['attention_mask'], 'attention_mask2': sentences_2['attention_mask']},
        similarities))

    return dataset


def map_features_ro_sts(features, output):
    return {
        'sentence1': int64_feature_list(features['sentence1_ids']),
        'sentence2': int64_feature_list(features['sentence2_ids']),
        'attention_mask1': int64_feature_list(features['attention_mask1']),
        'attention_mask2': int64_feature_list(features['attention_mask2']),
        'similarity': float_feature(output)

    }


if __name__ == '__main__':
    block_size = 64

    data_info = [
        Dataset_Info(
            '../../../dataset/ro-sts/raw/RO-STS.train.tsv',
            '../../../model/tokenizer',
            '../../../tf-record/ro-sts/train/train.tfrecord',
            block_size, 'train'
        ),

        Dataset_Info(
            '../../../dataset/ro-sts/raw/RO-STS.dev.tsv',
            '../../../model/tokenizer',
            '../../../tf-record/ro-sts/dev/dev.tfrecord',
            block_size, 'dev'
        )
    ]


    create_dataset_record(
        create_dataset_ro_sts, write_tf_record_wrapper, data_info, map_features_ro_sts,
        f'../../../tf-record/ro-sts/info.json'
    )
