import re
from pathlib import Path
import json
from sklearn.model_selection import train_test_split


def clean_line(line: str) -> str:
    line = line.strip()

    for remove_text in [
        'UPDATE.', 'UPDATE', 'GRAFICĂ. ', 'GRAFICĂ.', 'VIDEO.', 'VIDEO |', 'VIDEO', 'REPORTAJ', 'REPORTAJ.', 'AUDIO.',
        'AUDIO', 'Titlu:', 'TITLU:', 'Titlu', 'TITLU', 'FOTO.', 'FOTO', 'VIRAL. '
    ]:
        line = line.replace(remove_text, '')

    if len(line) == 0:
        return line

    if line[0] == ' ':
        line = line[1:]

    line = re.sub(' +', ' ', line)

    return line


def clean_dataset(path_dataset_raw: str, path_dataset_clean: str):
    dirs_dataset_raw = [str(x) for x in Path(path_dataset_raw).glob("*") if x.is_dir()]

    for director in dirs_dataset_raw:
        for sub_dir in [str(x) for x in Path(director).glob('**/*') if x.is_dir()]:
            category = director.split('/')[-1]
            sub_category = sub_dir.split('/')[-1]
            articles = []

            with open(f'{sub_dir}/articles.json', 'r', encoding='utf8') as input_file:
                data = json.load(input_file)

            for article in data:
                title = article['title']
                title = clean_line(title)
                title = re.sub('.*\|', '', title)
                text = []

                for line in article['text']:
                    line_processed = line.strip()
                    non_info_line = False

                    for non_info in [
                        'Citiți și', 'sursa:', 'Sursa:', 'Editor web', 'editor web', 'Editor', 'editor',
                        'Redactor:', 'redactor:', 'Reporter:' 'reporter:', '---------', "FOTO"
                    ]:
                        if non_info in line_processed:
                            non_info_line = True
                            break

                    if non_info_line:
                        continue

                    # clean non-info text
                    line_processed = clean_line(line_processed)

                    # if exist multiple lines
                    lines = line_processed.split('\n')
                    if len(lines) > 1:
                        lines = [i.strip() for i in lines if len(i) >= 1]
                        text = text + lines
                    else:
                        if len(line_processed) == 0:
                            continue
                        text.append(line_processed)

                articles.append({'title': title, 'text': text})

            with open(f'{path_dataset_clean}{category}/{sub_category}/articles.json', 'w+', encoding='utf8') as o_file:
                json.dump(articles, o_file, ensure_ascii=False, indent=4)


def split_dataset(path_dataset: str, path_split: str):
    train_articles = []
    dev_articles = []
    test_articles = []

    for director in [str(x) for x in Path(path_dataset).glob('**/*') if x.is_dir()]:
        for sub_director in [str(x) for x in Path(director).glob('**/*') if x.is_dir()]:
            with open(f'{sub_director}/articles.json', 'r') as input_file:
                articles = json.load(input_file)

            train, test = train_test_split(articles, test_size=0.1, random_state=42)
            dev, test = train_test_split(test, test_size=0.5, random_state=42)

            train_articles.extend(train)
            dev_articles.extend(dev)
            test_articles.extend(test)

    files_name = {'train.json': train_articles, 'dev.json': dev_articles, 'test.json': test_articles}
    for file_name, dataset in files_name.items():
        with open(f'{path_split}{file_name}', 'w+') as output_file:
            json.dump(dataset, output_file, ensure_ascii=False, indent=4)


def split_train(path_train_file: str, path_save_dir: str, split_nums: int):
    with open(path_train_file, 'r') as input_file:
        data = json.load(input_file)

    len_data = len(data)
    chunk_size = len_data // split_nums + int(len_data % split_nums > 0)
    index_split = [i for i in range(0, len_data, chunk_size)]
    if len_data - 1 not in index_split:
        index_split.append(len_data - 1)

    for i in range(len(index_split) - 1):
        data_chunk = data[index_split[i]: index_split[i + 1]]

        with open(f'{path_save_dir}train-{i}.json', 'w+') as output_file:
            json.dump(data_chunk, output_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    clean_dataset(
        '../../dataset/digi24/original/',
        '../../dataset/digi24/original/'
    )

    split_dataset(
        '../../dataset/digi24/original/', '../../dataset/digi24/split'
    )

    """
    split_train(
        '../../dataset/digi24/split/train.json',
        '../../dataset/digi24/split/split-train/', 20
    )
    """