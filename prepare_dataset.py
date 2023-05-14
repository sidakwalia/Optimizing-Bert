import argparse
import filecmp
import os
import pathlib
import shutil
import subprocess
import sys
import time


def get_default_config():
    config = {
        'system': {
            # 'num_gpus': 4,
            'num_gpus': 2,
            'max_memory_in_gb': 16
        },
        'wandb': {
            'api_key': '...'
        },
        'dataset': {
            'enabled': True,
            # 'huggingface_datasets': [],
            'huggingface_datasets': [('wikipedia', '20220301.simple')],
            'customized_datasets': [],
            'id': None,
            'tmp_dir': None,
            'output_dir': None,
            'log_dir': 'log/dataset'
        },
        'result_collection': {
            'enabled': True
        },
        'tokenizer': {
            'name_or_path': 'bert-large-uncased'
        }
    }

    return config


def get_dataset_list(config):
    customized_data_list = []
    for dataset_dir in config['dataset']['customized_datasets']:
        file_list = sorted(os.listdir(dataset_dir))
        for data_file in file_list:
            if data_file.endswith('.txt'):
                data_path = os.path.join(dataset_dir, data_file)
                customized_data_list.append(f'{data_path}')

    huggingface_data_list = []
    for dataset_name, subset_name in config['dataset']['huggingface_datasets']:
        huggingface_data_list.append(f'{dataset_name}.{subset_name}')

    return customized_data_list, huggingface_data_list


def read_str_from_file(file_path):
    file_content = ''
    if file_path.is_file():
        with open(file_path, 'r') as fin:
            file_content = fin.read()
    return file_content


def set_dataset_id(config):
    counter = -1    # Adds counter until there is no conflict IDs
                    # Normally the hash conflict probability is extremely low

    while True:
        counter += 1

        # Get customized & huggingface dataset list
        customized_data_list, huggingface_data_list = get_dataset_list(config)
        customized_data_str = ';'.join(customized_data_list)
        huggingface_data_str = ';'.join(huggingface_data_list)

        # Gets auto-generated ID

        # hash_list.append(get_md5sum_of_str(huggingface_data_str))
        # hash_list.append(get_md5sum_of_str(str(counter)))   # Avoid conflicts

        # hash_str = ''.join(hash_list)
        # final_hash = get_md5sum_of_str(hash_str)        # Two-layer md5sum

        final_hash = "abcdefg"

        config['dataset']['id'] = final_hash
        config['dataset']['tmp_dir'] = f'tmp/dataset/{config["dataset"]["id"]}'
        config['dataset']['output_dir'] = f'data/{config["dataset"]["id"]}'

        # Checks if the ID is already used
        project_dir = os.path.dirname(os.path.realpath(__file__))
        tmp_dir = pathlib.Path(project_dir, config['dataset']['tmp_dir'])
        output_dir = pathlib.Path(project_dir, config['dataset']['output_dir'])


        huggingface_file = pathlib.Path(tmp_dir, 'huggingface_dataset_list.txt')
        customized_file = pathlib.Path(tmp_dir, 'customized_dataset_list.txt')

        if not tmp_dir.is_dir() and not output_dir.is_dir():
            # Not used, new dataset group, generates dataset list file
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with open(huggingface_file, 'w') as fout:
                fout.write(huggingface_data_str)
            with open(customized_file, 'w') as fout:
                fout.write(customized_data_str)
            break

        # ID used, check if it was used by the same dataset group
        old_datalist_str = read_str_from_file(huggingface_file)
        if old_datalist_str != huggingface_data_str:
            continue        # ID used by a different dataset group

        old_datalist_str = read_str_from_file(customized_file)
        old_data_list = old_datalist_str.split(';')
        old_data_list = [ path for path in old_data_list if path != '' ]

        is_same = None
        if len(old_data_list) != len(customized_data_list):
            is_same = False
        else:
            for file_a, file_b in zip(old_data_list, customized_data_list):
                is_file_same = filecmp.cmp(file_a, file_b, shallow=False)
                if not is_file_same:
                    is_same = False
                    break
            if is_same == None:
                is_same = True

        if not is_same:
            continue        # ID used by a different dataset group
        else:
            break           # ID used by the same dataset group

    return config


def setup_config():
    config = get_default_config()
    # config.update(yaml.load(open(args.config_file)))

    if config['dataset']['id'] is None:
        config = set_dataset_id(config)
    else:
        config['dataset']['tmp_dir'] = f'tmp/dataset/{config["dataset"]["id"]}'
        config['dataset']['output_dir'] = f'data/{config["dataset"]["id"]}'

    if config['wandb']['api_key'] is None:
        raise ValueError(
            'WANDB.API_KEY not provided, '
            'please see "https://docs.wandb.ai/quickstart" for more details'
        )

    return config


def get_date():
    return subprocess.check_output('date').decode(sys.stdout.encoding).strip()


def logging(message):
    print(f'{get_date()}: ' + message, flush=True)


def run_bash(command):
    process = subprocess.run(
        command,
        shell=True,
        executable='/bin/bash',
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True
    )
    return process


def prepare_dataset(config, args=None):
    if not config['dataset']['enabled']:
        return

    logging('==== prepare dateset starts here =====')
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # The prepared dataset for pretraining is stored in {output_dir}
    output_dir = pathlib.Path(project_dir, config['dataset']['output_dir'])
    tmp_dir = pathlib.Path(project_dir, config['dataset']['tmp_dir'], 'content')
    log_dir = pathlib.Path(project_dir, config['dataset']['log_dir'])


    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # If the dataset file is prepared, skip
    skip_mark_file = pathlib.Path(tmp_dir, f"{config['dataset']['id']}.mark")
    if skip_mark_file.is_file():
        if args is None or not args.clear_cache:
            logging(
                f"Dataset for this ID {config['dataset']['id']}"
                ' has already been prepared, skip preprocessing...'
            )
            logging('===== prepare dateset ends here =====')
            return

    # Clear the temporary directory and output directory
    shutil.rmtree(output_dir)
    shutil.rmtree(tmp_dir)

    os.chdir('dataset')     # Goto {project_dir}/dataset

    logging('===== data sharding start...')
    shard_data_command = [
        'python shard_data.py',
        '  --num_train_shards 256',
        '  --num_test_shards 128',
        '  --frac_test 0.1',
        f'  --output_dir {tmp_dir}',
        f"  --max_memory {config['system']['max_memory_in_gb']}",
    ]
    for dataset_name, subset_name in config['dataset']['huggingface_datasets']:
        shard_data_command.append(f'  --dataset {dataset_name} {subset_name}')
    for dataset_dir in config['dataset']['customized_datasets']:
        shard_data_command.append(f'  --dataset custom {dataset_dir}')

    shard_data_command.extend([
        f'  > {log_dir}/shard_data.log',
        f'  2> {log_dir}/shard_data.err'
    ])

    logging(f'See {log_dir}/shard_data.[log|err] for detailed stdout/stderr')
    run_bash(''.join(shard_data_command))
    logging('===== data sharding end...')

    logging('===== sample generation start...')
    tokenizer_name = config['tokenizer']['name_or_path']

    generate_sample_command = [
        'python generate_samples.py',
        '  --dup_factor 10',
        '  --seed 42',
        '  --do_lower_case 1',
        '  --masked_lm_prob 0.15',
        '  --max_seq_length 128',
        '  --model_name bert-large-uncased',
        '  --max_predictions_per_seq 20',
        '  --n_processes 8',
        f'  --dir {tmp_dir}',
        f'  -o {output_dir}',
        f'  --tokenizer_name {tokenizer_name}',
        f'  > {log_dir}/generate_sample.log',
        f'  2> {log_dir}/generate_sample.err',
    ]

    logging(f'See {log_dir}/generate_sample.[log|err] for detailed'
             ' stdout/stderr')
    run_bash(''.join(generate_sample_command))
    logging('===== sample generation end')

    os.chdir('..')     # Goes back {project_dir}

    # Creates skip mark
    skip_mark_file.touch()
    logging('########## prepare dateset end')


def main():
    config = setup_config()
    prepare_dataset(config)

main()
