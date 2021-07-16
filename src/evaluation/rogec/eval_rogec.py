import re
from pathlib import Path
import json
import os


def fix_quotation_marks(line: str) -> str:
    line = re.sub(chr(8222), chr(34), line)
    line = re.sub(chr(8221), chr(34), line)
    line = line.replace("\n", "")
    line = re.sub("\s+", ' ', line)

    return line


if __name__ == '__main__':
    path_generate = '../../../generate/rogec'
    path_log = '../../../log/rogec'

    for path_dirs in [str(x) for x in Path(path_generate).glob("*") if x.is_dir()]:
        name_ds = path_dirs.split('/')[-1]

        for path_version_dirs in [str(x) for x in Path(path_dirs).glob('*') if x.is_dir()]:
            name_version = path_version_dirs.split('/')[-1]

            Path(f'{path_log}/{name_ds}/{name_version}').mkdir(exist_ok=True, parents=True)
            for file in [str(x) for x in Path(path_version_dirs).glob("*.json") if x.is_file()]:
                print(f"Evaluate : {file}\n")
                generate_method = file.split('/')[-1].replace('.json', '')

                originals = []
                predicts = []
                inputs = []

                # extract from generate
                with open(file, 'r') as input_file:
                    data = json.load(input_file)

                for i in data:
                    originals.append(fix_quotation_marks(i['original']))
                    predicts.append(fix_quotation_marks(i['predict']))
                    inputs.append(fix_quotation_marks(i['input']))

                with open('correct.txt', 'w+') as output_file:
                    for line in originals:
                        output_file.write(f'{line}\n')

                with open('predict.txt', 'w+') as output_file:
                    for line in predicts:
                        output_file.write(f'{line}\n')

                with open('input.txt', 'w+') as output_file:
                    for line in inputs:
                        output_file.write(f'{line}\n')

                # eval input
                os.system(
                    f'python3.8 ERRANT/parallel_to_m2.py -orig input.txt -cor correct.txt -out out_ref.txt -lang ro')

                # eval predict
                os.system(
                    f'python3.8 ERRANT/parallel_to_m2.py -orig predict.txt -cor correct.txt -out out_hyp.txt -lang ro')

                if os.path.exists(f'{path_log}/{name_ds}/{name_version}/{generate_method}.txt'):
                    os.remove(f'{path_log}/{name_ds}/{name_version}/{generate_method}.txt')

                # logging results
                os.system(
                    f'python3 ERRANT/compare_m2.py -hyp out_hyp.txt -ref out_ref.txt >> {path_log}/{name_ds}/{name_version}/{generate_method}.txt'
                )

                os.system(
                    f'python3 ERRANT/compare_m2.py -hyp out_hyp.txt -ref out_ref.txt -cse >> {path_log}/{name_ds}/{name_version}/{generate_method}.txt'
                )

                os.system(
                    f'python3 ERRANT/compare_m2.py -hyp out_hyp.txt -ref out_ref.txt -ds >> {path_log}/{name_ds}/{name_version}/{generate_method}.txt'
                )

                os.system(
                    f'python3 ERRANT/compare_m2.py -hyp out_hyp.txt -ref out_ref.txt -dt >> {path_log}/{name_ds}/{name_version}/{generate_method}.txt'
                )

    # clean-up
    os.remove('correct.txt')
    os.remove('predict.txt')
    os.remove('input.txt')
    os.remove('out_ref.txt')
    os.remove('out_hyp.txt')
