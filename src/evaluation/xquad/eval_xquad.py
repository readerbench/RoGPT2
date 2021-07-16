from __future__ import print_function, division, unicode_literals
from collections import Counter
from pathlib import Path
import string
import re
import json

import spacy

nlp = spacy.load('ro_core_news_lg', disable=["tagger", "attribute_ruler", "tok2vec", "ner"])


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace and for romanian language lemmatisation"""

    def lemma(text):
        my_doc = nlp(text)

        return ' '.join([token.lemma_ for token in my_doc])

    def remove_articles(text):
        return re.sub(r'\b(a|an|the|un|o)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return lemma(white_space_fix(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dir_generate):
    type_model = dir_generate.split('/')[-1]
    results = {}

    for i in [str(x) for x in Path(dir_generate).glob("*") if x.is_dir()]:
        for path_file in [str(x) for x in Path(i).glob('**/*.json') if x.is_file()]:
            f1, exact_match, total = 0, 0, 0
            name_eval = path_file.split('/')[-1].replace('.json', '')

            with open(path_file, 'r') as input:
                data = json.load(input)

            for example in data:
                ground_truths = [example['original']]
                prediction = example['predict']

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                total += 1

            exact_match = 100.0 * exact_match / total
            f1 = 100.0 * f1 / total

            results[name_eval] = {'exact_match': exact_match, 'f1': f1}

    Path('../../../log/xquad/').mkdir(exist_ok=True, parents=True)
    with open(f'../../../log/xquad/{type_model}.txt', 'w+') as output_file:
        for k, v in results.items():
            output_file.write(f'{k}: {v}\n')


if __name__ == '__main__':
    evaluate('../../../generate/xquad/normal')
    evaluate('../../../generate/xquad/translate')
