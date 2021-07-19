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

| Version | Number of parameters | Number of epoch | Duration of an epoch | Block size | Batch size | PPL |
|:-------:|:--------------------:|:---------------:|:--------------------:|:----------:|:----------:|:---:|
| Base   | 124M | 15 | 7h      | 1024 | 72 | 22.96    |
| Medium | 354M | 10 | 22h     | 1024 | 24 | 17.64    |
| Large  | 774M | 5  | **45h** | 512  | 16 | **16.77**|

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

### 4. WMT16

| Model        | Decoder method | Ro-En | En-Ro |
|:------------:|:--------------:|:-----:|:-----:|
|RoGPT2-base   |Greedy          | 30.37 | 20.27 |
|RoGPT2-base   |Beam-search-4   | 31.26 | 22.31 |
|RoGPT2-base   |Beam-search-8   | 31.39 | 22.95 |
|RoGPT2-medium |Greedy          | 32.48 | 22.18 |
|RoGPT2-medium |Beam-search-4   | 34.08 | 24.03 |
|RoGPT2-medium |Beam-search-8   | 34.16 | 24.13 |
|RoGPT2-large  |Greedy          | 33.69 | 23.31 |
|RoGPT2-large  |Beam-search-4   |34.40  |24.23  |
|RoGPT2-large  |Beam-search-8   |34.51  |24.32  | 

### 5. XQuAD
| Model        |Decoder method |  EM   | F1-Score |
|:------------:|:-------------:|:-----:|:--------:|
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

### 7. RoGEC

