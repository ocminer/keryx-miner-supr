#!/usr/bin/env python3
"""Prepare the pinned Vulkan trace backend identically on Linux and Windows.

Only pristine checkouts or an exact previously generated tree are accepted. CUDA
keeps its own pin and patch series. Recreate this disposable checkout when the
Vulkan patch/export inputs change; arbitrary changes are never reset or removed.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PIN = 'd81235049384534c167caea52b85a694f6103d14'
MARK = '// ---- KERYX VULKAN GENERATED EXPORTS ----'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(target):
    target = target.resolve()
    patches = sorted((ROOT / 'tools/keryx-llama/vulkan-patches').glob('*.patch'))
    assert patches, 'Vulkan trace patch series is empty'
    # Normalize Git checkout line endings on Windows before binding source inputs.
    patch_data = [p.read_text(encoding='utf-8').encode() for p in patches]
    exports = (ROOT / 'tools/keryx-llama/keryx_vk_exports.inc.cpp').read_text(encoding='utf-8')
    inputs = {p.name: sha(data) for p, data in zip(patches, patch_data)}
    inputs['exports'] = sha(exports.encode())
    inputs['pin'] = PIN
    stamp = target / '.keryx-vulkan-prepared.json'

    def git(*args, **kwargs):
        return subprocess.check_output(['git', '-C', str(target), *args], **kwargs)

    if not target.exists():
        target.mkdir(parents=True)
        subprocess.run(['git', 'init', '--quiet', str(target)], check=True)
        git('config', 'core.autocrlf', 'false')
        git('remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp')
        git('fetch', '--quiet', '--depth=1', 'origin', PIN)
        git('checkout', '--quiet', '--detach', 'FETCH_HEAD')
    assert git('rev-parse', 'HEAD').decode().strip() == PIN, 'Wrong Vulkan llama.cpp pin'
    git('config', 'core.autocrlf', 'false')

    def status():
        return git('status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules').decode().splitlines()

    if stamp.exists():
        old = json.loads(stamp.read_text())
        actual = {line[3:] for line in status() if line[3:] != stamp.name}
        assert actual == set(old['files']), 'Unexpected files in the prepared Vulkan checkout'
        for name, expected in old['files'].items():
            assert sha((target / name).read_bytes()) == expected, 'Modified native input: '+name
        if old['inputs'] != inputs:
            # Export regeneration is safe only after every old native file has matched
            # the recorded bytes. Changes to the native patch series need a fresh tree.
            without_exports = lambda x: {k: v for k, v in x.items() if k != 'exports'}
            assert without_exports(old['inputs']) == without_exports(inputs), 'Vulkan patches changed; recreate the disposable native checkout'
            source = target / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
            text = source.read_text(encoding='utf-8')
            assert text.count(MARK) == 1
            with source.open('w', encoding='utf-8', newline='\n') as f:
                f.write(text.split(MARK)[0]+MARK+'\n'+exports+'\n')
            old['files'][source.relative_to(target).as_posix()] = sha(source.read_bytes())
            old['inputs'] = inputs
            stamp.write_text(json.dumps(old, indent=2)+'\n')
            print('Vulkan exports refreshed after exact prepared-tree verification')
        print('Vulkan source: exact prepared pin/patch/export match')
        return

    assert not status(), 'Vulkan checkout must be pristine before preparation'
    for data in patch_data:
        subprocess.run(['git', '-C', str(target), 'apply', '--check', '-'], input=data, check=True)
        subprocess.run(['git', '-C', str(target), 'apply', '-'], input=data, check=True)
    source = target / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
    with source.open('a', encoding='utf-8', newline='\n') as f:
        f.write('\n'+MARK+'\n'+exports+'\n')
    files = {line[3:]: sha((target / line[3:]).read_bytes()) for line in status()}
    stamp.write_text(json.dumps({'inputs': inputs, 'files': files}, indent=2)+'\n')
    print('Vulkan source prepared:', PIN, len(files), 'patched/generated files')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: prepare-llama-vulkan.py <disposable-source-directory>')
    prepare(Path(sys.argv[1]))
