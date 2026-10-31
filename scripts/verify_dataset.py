"""Verify downloaded archives or extracted dataset bytes. Python 3.12, no packages.

Never repairs, rewrites or deletes data. Exit 0 means matching bytes; exit 1
means validation failed. This is an integrity check, not scientific validation.
"""
import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys

ARCHIVES = {f'dataset-v1.0.0-{part}.zip' for part in ('train', 'val', 'test')}


def safe_path(root, relative):
    """Accept only portable relative paths within the explicitly selected root."""
    parts = relative.split('/')
    if (not relative or '\\' in relative or ':' in relative or PurePosixPath(relative).is_absolute()
            or any(p in ('', '.', '..') for p in parts)):
        raise ValueError(f'Unsafe manifest path: {relative!r}')
    root = root.resolve()
    target = root.joinpath(*parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Manifest path escapes the selected directory: {relative}')
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise ValueError(f'Links and junctions are not accepted: {relative}')
    return target


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_manifest(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['relative_path', 'size_bytes', 'sha256']:
            raise ValueError('Use metadata/raster_manifest.csv, not the source inventory CSV.')
        rows = list(reader)
    if not rows:
        raise ValueError('The raster manifest is empty.')
    seen = set()
    for row in rows:
        name = row['relative_path']
        if not name or name.casefold() in seen:
            raise ValueError('Empty or duplicate path in raster manifest.')
        seen.add(name.casefold())
        if not re.fullmatch('[0-9a-f]{64}', row['sha256'] or '') or not (row['size_bytes'] or '').isdigit():
            raise ValueError(f'Invalid SHA-256 or size for {name}.')
    return rows


def verify_files(root, rows, reject_extra=True):
    """Shared verification routine, also tested with deliberately damaged fixtures."""
    if not root.is_dir():
        raise ValueError(f'Dataset directory not found: {root}')
    expected = set()
    failures = []
    for i, row in enumerate(rows, 1):
        name = row['relative_path']
        path = safe_path(root, name)
        expected.add(name)
        if not path.is_file():
            failures.append(f'MISSING: {name}')
        elif path.stat().st_size != int(row['size_bytes']):
            failures.append(f'SIZE MISMATCH: {name}')
        elif sha256(path) != row['sha256']:
            failures.append(f'SHA256 MISMATCH: {name}')
        if i % 500 == 0:
            print(f'Checked {i}/{len(rows)} files', flush=True)
    if reject_extra:
        actual = set()
        for path in root.rglob('*'):
            if path.is_symlink() or path.is_junction():
                raise ValueError('Unexpected filesystem link in dataset directory.')
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
        failures.extend(f'UNEXPECTED FILE: {name}' for name in sorted(actual - expected))
    if failures:
        raise ValueError('\n'.join(failures[:20]) + f'\nFailed checks: {len(failures)}. Files were not changed.')
    print(f'PASS: {len(rows)} files match the manifest by path, size and SHA-256.', flush=True)


def verify_archives(folder):
    """Checksum paths are resolved relative to SHA256SUMS.txt, not the shell cwd."""
    folder = folder.resolve()
    sums = folder / 'SHA256SUMS.txt'
    rows = []
    for line in sums.read_text(encoding='utf-8-sig').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError('Invalid SHA256SUMS.txt; expected SHA-256, two spaces, filename.')
        checksum, name = match.groups()
        safe_path(folder, name)
        rows.append((checksum, name))
    if len(rows) != 3 or {name for _, name in rows} != ARCHIVES:
        raise ValueError('SHA256SUMS.txt must list exactly the three v1.0.0 archives once each.')
    for checksum, name in rows:
        path = folder / name
        print(f'Checking {name} ...', flush=True)
        if not path.is_file() or sha256(path) != checksum:
            raise ValueError(f'Archive missing or SHA-256 mismatch: {name}. Download it again.')
    print('PASS: all three archive SHA-256 checks match.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, help='Folder containing three ZIP files and SHA256SUMS.txt; checks archives only')
    parser.add_argument('--data-root', type=Path, help='Extracted SAMPLE_DATASET_256x256 directory')
    parser.add_argument('--manifest', type=Path, help='Path to extracted metadata/raster_manifest.csv')
    args = parser.parse_args()
    if args.archives:
        if args.data_root or args.manifest:
            raise ValueError('Run archive and extracted-data verification as two separate commands.')
        verify_archives(args.archives)
        return
    release = Path(__file__).resolve().parent.parent
    data_root = args.data_root or release / 'data/SAMPLE_DATASET_256x256'
    manifest = args.manifest or release / 'metadata/raster_manifest.csv'
    rows = load_manifest(manifest)
    if len(rows) != 9000:
        raise ValueError(f'v1.0.0 requires 9000 raster entries; found {len(rows)}.')
    verify_files(data_root, rows)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('Verification cancelled; no data files were modified.', file=sys.stderr)
        sys.exit(130)
