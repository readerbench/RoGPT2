import json
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    path_ds = '../../../dataset/laroseda/raw/laroseda_train.json'

    with open(path_ds, 'r') as input_file:
        data = json.load(input_file)

    rating_1 = []
    rating_2 = []
    rating_4 = []
    rating_5 = []

    for example in data['reviews']:
        rating = int(example["starRating"])
        if rating == 1:
            rating_1.append(example)

        if rating == 2:
            rating_2.append(example)

        if rating == 4:
            rating_4.append(example)

        if rating == 5:
            rating_5.append(example)

    rating_1_train, rating_1_dev = train_test_split(rating_1, test_size=0.1, random_state=42)
    rating_2_train, rating_2_dev = train_test_split(rating_2, test_size=0.1, random_state=42)
    rating_4_train, rating_4_dev = train_test_split(rating_4, test_size=0.1, random_state=42)
    rating_5_train, rating_5_dev = train_test_split(rating_5, test_size=0.1, random_state=42)

    train = rating_1_train + rating_2_train + rating_4_train + rating_5_train
    dev = rating_1_dev + rating_2_dev + rating_4_dev + rating_5_dev

    with open('../../../dataset/laroseda/split/train.json', 'w+') as output:
        json.dump(train, output, ensure_ascii=False, indent=4)

    with open('../../../dataset/laroseda/split/dev.json', 'w+') as output:
        json.dump(dev, output, ensure_ascii=False, indent=4)
