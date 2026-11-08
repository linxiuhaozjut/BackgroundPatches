"""Show stage commands without loading models or importing training dependencies.

Read stage configuration files and display their command-line equivalents.
Provide datasets, weights, and dependencies before executing a displayed command.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def options(values):
    result = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        result.append('--' + key.replace('_', '-'))
        if value is not True:
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def commands(stage):
    paths = load('paths.json')
    def cmd(script, values):
        return ['python', script, *options(values)]
    if stage == 'prepare':
        return [cmd('generator/paired_dit_data.py', dict(load('prepare.json'),
            dataset_root=paths['generator_dataset'], output_root=paths['generator_cache'],
            vae=paths['vae']))]
    if stage in ('train-base', 'train-prototype'):
        variant = 'base' if stage == 'train-base' else 'prototype'
        args = dict(load('generator_' + variant + '.json'),
                    cache_root=paths['generator_cache'],
                    output_root=str(Path('outputs/generator') / variant))
        args['init_ckpt' if variant == 'base' else 'warm_start'] = paths[
            'dit_backbone' if variant == 'base' else 'base_checkpoint']
        return [cmd('generator/train_paired_dit.py', args)]
    if stage == 'generate':
        return [cmd('generator/sample_paired_dit.py', dict(load('sampling.json'),
            ckpt=paths['generator_checkpoint'], output_root=paths['initial_patch'], vae=paths['vae']))]
    if stage in ('optimize', 'optimize-pixel'):
        coherent = stage == 'optimize'
        args = dict(load('optimization_coherent.json' if coherent else 'optimization_pixel.json'),
                    dataset_root=paths['carla_dataset'],
                    mask_root=str(Path(paths['carla_dataset']) / 'mask'),
                    patch_dir=paths['initial_patch'], checkpoint=paths['tracker_checkpoint'],
                    placement_script='preprocessing/apply_paired_patches.py',
                    output_root=str(Path(paths['optimization_output']) / ('coherent' if coherent else 'pixel')),
                    sequence_list=paths['development_split'], optimization_list=paths['train_split'])
        filename = 'optimize_dataset4002_coherent.py' if coherent else 'optimize_dataset4002_blackbox.py'
        return [cmd('optimization/' + filename, args)]
    if stage == 'annotate':
        return [cmd('preprocessing/build_fixed_camera_annotations.py', dict(
            dataset_root=paths['carla_dataset'], weights=paths['segmentation_weights']))]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['annotate', 'prepare', 'train-base',
        'train-prototype', 'generate', 'optimize', 'optimize-pixel', 'all'])
    args = parser.parse_args()
    stages = ['prepare', 'train-base', 'train-prototype', 'generate', 'optimize'] if args.stage == 'all' else [args.stage]
    print('COMMAND PREVIEW ONLY - no training, inference, or optimization is started.')
    print('Run commands from the repository root.')
    print('Additional Python import directory: tracker/')
    for stage in stages:
        print('\n[' + stage + ']')
        for command in commands(stage):
            print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command))


if __name__ == '__main__':
    main()
