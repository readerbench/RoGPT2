# RoGPT2: Romanian GPT2 for text generation

---

This is the Romanian language version of the GPT2 model. There are 3 trained versions, they are available on the HuggingFace Hub:

* [base](https://huggingface.co/readerbench/RoGPT2-base)
* [medium](https://huggingface.co/readerbench/RoGPT2-medium)
* [large](https://huggingface.co/readerbench/RoGPT2-large)

## Training

---

### Corpus Statistics

| Corpus | Total size | Number of words | Number of sentences |
|:------:|:----------:|:---------------:|:-------------------:|
|OSCAR| 11.54 GB | 1745M | 48.46M |
|Wiki-Ro | 0.46 GB | 68M | 1.79M |
|Debates | 0.5 GB | 73M | 3.61M |
|Books | 4.37 GB | 667M | 37.39M |
|News | 0.15 GB | 23M | 0.77M |

### Training Statistics

| Version | Number of parameters | Number of epoch | Duration of an epoch | Context size | Batch size | PPL |
|:-------:|:--------------------:|:---------------:|:--------------------:|:----------:|:----------:|:---:|
| Base   | 124M | 15 | 7h      | 1024 | 72 | 22.96    |
| Medium | 354M | 10 | 22h     | 1024 | 24 | 17.64    |
| Large  | 774M | 5  | **45h** | 512  | 16 | **16.77**|

## Install Dependencies

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
wget https://nextcloud.readerbench.com/index.php/s/2jasc6H79F4ANkD/download -O dataset.zip
unzip dataset.zip
rm -fr dataset.zip
wget https://nextcloud.readerbench.com/index.php/s/94EKKmTCt9CjTXf/download -O model.zip
unzip model.zip
rm -fr model.zip
```

The training corpus can be found at the [link](https://nextcloud.readerbench.com/index.php/s/EMpPgRJtMPexcRt). <br /> 
The datasets for evaluation can be found at the [link](https://nextcloud.readerbench.com/index.php/s/2jasc6H79F4ANkD). <br />
The downstream models can be found at the [link](https://nextcloud.readerbench.com/index.php/s/94EKKmTCt9CjTXf). <br />

## Evaluation

---

### 1. [MOROCO](src/evaluation/moroco/Eval_MOROCO.ipynb)

|      Model        | Dialect | Md to Ro | Ro to Md |
|:-----------------:|:-------:|:--------:|:--------:|
| KRR + SK          | 94.06   | 67.59    | 75.47    |
| BERT-base-ro      | 95.98   | 69.90    | 78.08    |
| RoBERT-small      | 95.76   | 69.05    | 80.15    |
| RoBERT-base       |**97.24**| 68.80    | 82.37    |
| RoBERT-large      | 97.21   | 69.50    | **83.26**|
| RoGPT2-base       | 96.69   | 69.82    | 77.55    |
| RoGPT2-medium     | 96.42   | 69.77    | 80.51    |
| RoGPT2-large      | 96.93   |**71.07** | 82.56    |

### 2. [LaRoSeDa](src/evaluation/laroseda/Eval_LaRoSeDa.ipynb)

| Model        | Binary: Accuracy | Binary: F1-Score | Multi-Class: Accuracy | Multi-Class: F1-Score |
|:------------:|:----------------:|:----------------:|:---------------------:|:---------------------:|
|BERT-base-ro  | 98.07            | 97.94            | -                     |79.61                  |
| RoDiBERT     |**98.40**         |**98.31**         | -                     |83.01                  |
| RoBERT-small | 97.44            | 97.43            | 89.30                 |84.23                  |
| RoBERT-base  | 98.27            | 98.26            | 90.59                 |86.27                  |
| RoBERT-large | 98.20            | 98.19            |**90.93**              |**86.63**              |
| RoGPT2-base  | 97.89            | 97.88            |89.65                  |84.68                  |
|RoGPT2-medium | 98.03            |98.04             | 90.29                 | 85.37                 |
| RoGPT2-large | 98.06            |98.07             | 90.26                 | 84.89                 |

### 3. [RoSTS](src/evaluation/ro-sts/Eval_RoSTS.ipynb)

| Model        | Spearman dev-set | Spearman test-set | Pearson dev-set | Pearson test-set |
|:------------:|:----------------:|:-----------------:|:---------------:|:----------------:|
|BERT-base-ro  | 84.26            | 80.86             | 84.59           | 81.59            |
|RoDiBERT      | 77.07            | 71.47             | 77.13           | 72.25            |
|RoBERT-small  | 82.06            | 78.06             | 81.66           | 78.49            |
|RoBERT-base   | 84.93            | 80.39             | 85.03           | 80.39            |
|RoBERT-large  |**86.25**         |**83.15**          |**86.58**        |**83.76**         |
|RoGPT2-base   | 83.51            | 79.77             | 83.74           | 80.56            |
|RoGPT2-medium | 85.75            | 82.25             | 86.04           | 83.16            |
|RoGPT2-large  | 85.70            | 82.64             | 86.14           | 83.46            |

### 4. [WMT16](src/evaluation/translate/Eval_Translate.ipynb)

| Model        | Decoder method | Ro-En  | En-Ro  |
|:------------:|:--------------:|:------:|:------:|
|mBART         | -              |**38.5**|**38.5**|
|OpenNMT       | -              |  -     | 24.7   |
|RoGPT2-base   |Greedy          | 30.37  | 20.27  |
|RoGPT2-base   |Beam-search-4   | 31.26  | 22.31  |
|RoGPT2-base   |Beam-search-8   | 31.39  | 22.95  |
|RoGPT2-medium |Greedy          | 32.48  | 22.18  |
|RoGPT2-medium |Beam-search-4   | 34.08  | 24.03  |
|RoGPT2-medium |Beam-search-8   | 34.16  | 24.13  |
|RoGPT2-large  |Greedy          | 33.69  | 23.31  |
|RoGPT2-large  |Beam-search-4   |34.40   |24.23   |
|RoGPT2-large  |Beam-search-8   |34.51   |24.32   | 

### 5. [XQuAD](src/evaluation/xquad/Eval_XQuAD.ipynb)
| Model        |Decoder method |  EM   | F1-Score |
|:------------:|:-------------:|:-----:|:--------:|
|BERT-base-ro  | -             | 47.89 | 63.74    |
|RoDiBERT      | -             | 21.76 | 34.57    |
|RoBERT-small  | -             | 30.84 | 45.17    |
|RoBERT-base   | -             | 53.52 | 70.04    |
|RoBERT-large  | -             | 55.46 |  69.64   |
|mBERT         | -             | 59.9  |  72.7    |
|XLM-R Large   | -             |**69.7**| **83.6**|
|RoGPT2-base   |  Greedy       | 23.69 | 35.97    |
|RoGPT2-base   | Beam-search-4 | 24.11 | 35.27    |
|RoGPT2-medium | Greedy        | 29.66 | 44.74    |
|RoGPT2-medium | Beam-search-4 | 31.59 | 45.32 |
|RoGPT2-large | Greedy | 29.74 | 42.98 |
|RoGPT2-large | Beam-search-4 | 29.66 | 43.05 |
|RoGPT2-base-en-ro | Greedy | 23.86 | 34.27 |
|RoGPT2-base-en-ro | Beam-search-4 | 25.04 | 34.51 |
|RoGPT2-medium-en-ro | Greedy | 27.05 | 39.75 |
|RoGPT2-medium-en-ro | Beam-search-4 | 27.64 | 39.11 |
|RoGPT2-large-en-ro | Greedy | 28.40 | 39.79 |
|RoGPT2-large-en-ro | Beam-search-4 | 28.73 | 39.71 |
|RoGPT2-large-en-ro-mask | Greedy | 31.34 | 44.71 |
|RoGPT2-large-en-ro-mask|  Beam-search-4 | 31.59 | 43.53 |

### 6. [Wiki-Ro: LM](src/evaluation/lm-wiki-ro/Eval_LM.ipynb)

| Model        | PPL dev | PPL test |
|:------------:|:-------:|:--------:|
|BERT-base-ro | 29.0897 | 28.0043|
|RoGPT2-base | 34.3795 | 33.7460|
|RoGPT2-medium | 23.7879 | 23.4581|
|RoGPT2-large | **21.7491** | **21.5200** |

### 7. [RoGEC](src/evaluation/rogec/Eval_RoGEC.ipynb)

| Model | Decoder mothod |  P  |  R  |  F<sub>0.5</sub>   |
|:-----:|:--------------:|:---:|:---:|:------:|
|Transformer-tiny | Beam-search | 53.53 | 26.36 | 44.38 |
|Transformer-base Finetuning | Beam-search | 56.05 | 46.19 | 53.76 |
|Transformer-base Finetuning | Beam-search-LM | 50.68 | 45.39 | 49.52 |
|Transformer-base Finetuning | Beam-search-norm-LM | 51.06 | 45.43 | 49.83 |
|RoGPT2-base | Greedy | 59.02 | 49.35 | 56.80 |
|RoGPT2-base | Beam-search-4 | 65.23 | 49.26 | 61.26 |
|RoGPT2-base  |Beam-search-8 | 65.88 | 49.64 | 61.84 |
|RoGPT2-medium | Greedy | 69.97 | 57.94 | 67.18 |
|RoGPT2-medium | Beam-search-4 | **72.46** | **57.99** | **69.01** |
|RoGPT2-medium | Beam-search-8 | 72.24 | 57.69 | 68.77 |
|RoGP2-large | Greedy | 61.90 | 49.09 | 58.83 |
|RoGP2-large | Beam-search-4 | 65.24 | 49.43 | 61.32 |
|RoGP2-large | Beam-search-8 | 64.96 | 49.22 | 61.06 |
|RoGPT2-base* | Greedy | 68.67 | 49.60 | 63.77 |
|RoGPT2-base* | Beam-search-4 | 71.16 | 50.53 | 65.79 |
|RoGPT2-base* | Beam-search-8 | 71.68 | 50.65 | 66.18 |
|RoGPT2-medium* | Greedy | 58.21 | 43.32 | 54.47 |
|RoGPT2-medium* | Beam-search-4 | 68.31 | 43.78 | 61.43 |
|RoGPT2-medium* | Beam-search-8 | 68.68 | 43.99 | 61.75 |
|RoGPT2-large* | Greedy | 64.86 | 41.30 | 58.22 |
|RoGPT2-large* | Beam-search-4 | 65.57 | 41.00 | 58.55 |
|RoGPT2-large* | Beam-search-8 | 65.44 | 41.09 | 58.50 |

**__Note__**: * the models were trained using the dataset of 3,000,000 artificially generated pairs

## Practical Application

--- 

### [Continuation and Title generation](src/news-generation/Practical_Application.ipynb)

##  Acknowledgments

---
Research supported with [Cloud TPUs](https://cloud.google.com/tpu/) from Google's [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc)

## How to cite

---

@inproceedings{niculescu2021rogpt2, <br>
  title={RoGPT2: Romanian GPT2 for Text Generation}, <br>
  author={Niculescu, Mihai Alexandru and Ruseti, Stefan and Dascalu, Mihai}, <br>
  booktitle={2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI)}, <br>
  pages={1154--1161}, <br>
  year={2021}, <br>
  organization={IEEE} <br>
}
