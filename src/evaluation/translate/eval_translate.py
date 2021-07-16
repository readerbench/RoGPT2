import json
from pathlib import Path

from sacrebleu import corpus_bleu

if __name__ == '__main__':
    modes = ['ro-en', 'en-ro']
    versions = ['base', 'medium', 'large']

    for log_dir in [str(x) for x in Path(f'../../../generate/translate').glob('*') if x.is_dir()]:
        trained_version = log_dir.split('/')[-1]
        print(f'Trained version {trained_version}\n')

        for mode in modes:
            for version in versions:
                result = {}

                for path_log in [str(x) for x in Path(f'{log_dir}/{mode}/{version}').glob("**/*.json")]:

                    hyp = []
                    refs = []
                    name_eval = path_log.split('/')[-1].replace('.json', '')
                    print(path_log)

                    with open(path_log, 'r') as input_file:
                        data = json.load(input_file)

                    for example in data:
                        refs.append(example['original'])
                        hyp.append(
                            example['predict'].replace('Romanian:', '').replace('English:', ''))  # remove artefact

                    refs = [refs]
                    bleu_score = corpus_bleu(hyp, refs)
                    result[name_eval] = bleu_score

                Path(f'../../../log/translate/{trained_version}').mkdir(parents=True, exist_ok=True)
                with open(f'../../../log/translate/{trained_version}/{version}-{mode}.txt', 'w+') as output_file:
                    for k, v in result.items():
                        output_file.write(f'{k}: {v}\n')