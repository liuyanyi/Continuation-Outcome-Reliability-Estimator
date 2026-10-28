"""Build separate public code/data ZIPs without local GPQA data or run artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

EXCLUDED_DIRS = {'.venv', '.git', '__pycache__', '.cache', '.pytest_cache', '.ruff_cache', 'outputs', 'logs', 'build', 'dist', 'downloads'}
TOKEN_MAPS = {'track_token_ids.json', 'track_token_id2text.json'}


def archive_group(relative: Path) -> str | None:
    parts = relative.parts
    if any(part in EXCLUDED_DIRS or part.endswith('.egg-info') for part in parts):
        return None
    if relative.suffix in {'.pyc', '.pyo', '.log', '.zip', '.sha256'}:
        return None
    if parts[:2] in {('data', 'private'), ('data', 'gpqa')}:
        return None
    if parts[0] == 'data' and 'licenses' not in parts:
        if any('gpqa' in part.lower() for part in parts):
            return None
    return 'data' if parts[0] == 'data' or relative.name in TOKEN_MAPS else 'code'


def package(root: Path, output_dir: Path, only: str = "all") -> list[Path]:
    if only not in {"all", "code", "data"}:
        raise ValueError("only must be all, code or data")
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(root):
        raise ValueError('Output directory must be outside the release directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = {'code': [], 'data': []}
    for path in sorted(root.rglob('*')):
        group = archive_group(path.relative_to(root))
        if group is None:
            continue
        if path.is_symlink():
            raise ValueError(f'Unexpected symlink: {path}')
        if path.is_file():
            groups[group].append(path)
    outputs = []
    for group, paths in groups.items():
        if only != "all" and group != only:
            continue
        output = output_dir / f'confidence_aware_{group}.zip'
        partial = output.with_suffix('.zip.partial')
        print(f'Packing {group}: {len(paths)} files', flush=True)
        try:
            with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                for path in paths:
                    if path.stat().st_size > 100 * 1024**2:
                        print(f'Compressing {path.relative_to(root)}', flush=True)
                    archive.write(path, arcname=str(Path('release') / path.relative_to(root)))
            print(f'Verifying {output.name} ({partial.stat().st_size:,} bytes)', flush=True)
            with zipfile.ZipFile(partial) as archive:
                expected = {str(Path('release') / path.relative_to(root)) for path in paths}
                if set(archive.namelist()) != expected:
                    raise ValueError('Archive membership mismatch')
                bad = archive.testzip()
                if bad:
                    raise ValueError(f'ZIP integrity check failed: {bad}')
            partial.replace(output)
            with output.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
            print(f'OK: {output.name}, {output.stat().st_size:,} bytes, SHA256 {digest}', flush=True)
            outputs.append(output)
        finally:
            partial.unlink(missing_ok=True)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory outside release/; existing archives with these names are replaced')
    parser.add_argument("--only", choices=["all", "code", "data"], default="all", help="Choose which archive to rebuild")
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], args.output_dir, only=args.only)
