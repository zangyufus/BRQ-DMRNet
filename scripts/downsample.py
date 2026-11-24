"""Uniformly downsample every processed UrbanBIS PTH without replacement."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from preprocess_superpoints import atomic_save, load_scene


def sample_indices(n, ratio, seed, relative_path):
    ratio = Fraction(str(ratio))
    if n < 1 or not 0 < ratio <= 1:
        raise ValueError("Input must be nonempty and 0 < ratio <= 1")
    keep = max(1, n * ratio.numerator // ratio.denominator)
    key = f"{seed}:{Path(relative_path).as_posix()}".encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    rng = np.random.default_rng(local_seed)
    return np.sort(rng.choice(n, size=keep, replace=False)).astype(np.int64)


def atomic_save_pth(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(values, handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_file(job):
    source_root, output_root, relative, ratio, seed, overwrite = job
    torch.set_num_threads(1)
    started = time.perf_counter()
    source, destination = source_root / relative, output_root / relative
    before = source.stat()
    arrays = load_scene(source)
    n = len(arrays[0])
    selected = sample_indices(n, ratio, seed, relative)
    index_path = destination.with_suffix(".source_indices.npy")
    if not overwrite and (destination.exists() or index_path.exists()):
        if not destination.exists() or not index_path.exists():
            raise FileExistsError(f"Incomplete output at {destination}; use --overwrite to regenerate")
        np.testing.assert_array_equal(np.load(index_path, allow_pickle=False), selected)
        status = "existing"
    else:
        values = tuple(np.ascontiguousarray(value[selected]) for value in arrays)
        atomic_save_pth(destination, values)
        atomic_save(index_path, selected)
        status = "prepared"
    # Check the serialized output, including all field values and dtypes.
    restored = load_scene(destination)
    saved_indices = np.load(index_path, allow_pickle=False)
    np.testing.assert_array_equal(saved_indices, selected)
    for original, saved in zip(arrays, restored):
        if original.dtype != saved.dtype:
            raise ValueError(f"Field dtype changed: {destination}")
        np.testing.assert_array_equal(saved, original[saved_indices])
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source file changed during sampling: {source}")
    return dict(
        file=relative.as_posix(), source=str(source), destination=str(destination),
        source_indices=index_path.relative_to(output_root).as_posix(),
        input_points=n, points=len(selected), status=status,
        source_bytes=before.st_size, output_bytes=destination.stat().st_size,
        source_mtime_ns=before.st_mtime_ns, verified_all_five_fields=True,
        seconds=time.perf_counter() - started,
    )


def summarize(records):
    result = {}
    for split in ["all", *sorted({Path(r["file"]).parts[1] for r in records})]:
        rows = records if split == "all" else [r for r in records if Path(r["file"]).parts[1] == split]
        sizes = [r["points"] for r in rows]
        total = sum(sizes)
        result[split] = dict(files=len(rows), input_points=sum(r["input_points"] for r in rows),
                             points=total, min_points=min(sizes), max_points=max(sizes),
                             mean_points=total / len(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--ratio", default="0.1", help="Fraction to retain per file; counts round down")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        ratio = Fraction(args.ratio)
    except (ValueError, ZeroDivisionError):
        parser.error("ratio must be a finite number in (0, 1]")
    if not 0 < ratio <= 1 or args.num_workers < 1:
        parser.error("Require 0 < ratio <= 1 and num_workers >= 1")
    source, output = args.dataset_root.resolve(), args.output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate, non-nested directories")
    paths = sorted(source.glob("*/*/*.pth"))
    if not paths:
        parser.error("No files matched dataset_root/<city>/<split>/*.pth")
    policy = dict(
        source_root=str(source), output_root=str(output), ratio=float(ratio),
        ratio_exact=str(ratio), seed=args.seed,
        selection="uniform random without replacement; independent of labels",
        count="max(1, floor(input_points * ratio))",
        point_order="selected source indices sorted in ascending order",
        seed_per_file="first 8 bytes of SHA256(seed:relative_path), little endian",
        fields=["coords", "colors", "semantic_labels", "instance_labels", "building_type"],
        labels_and_coordinate_values_preserved=True,
        source_files=[p.relative_to(source).as_posix() for p in paths],
    )
    policy_path = output / "downsampling_policy.json"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        if not policy_path.is_file() or json.loads(policy_path.read_text()) != policy:
            parser.error("Output is nonempty and its sampling policy differs; choose another output_root")
    atomic_save(policy_path, policy, json_file=True)
    jobs = [(source, output, p.relative_to(source), str(ratio), args.seed, args.overwrite) for p in paths]
    started, records = time.perf_counter(), []
    if args.num_workers == 1:
        for record in map(process_file, jobs):
            records.append(record)
            print(json.dumps({k: record[k] for k in ("file", "input_points", "points", "status")}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context("spawn")) as pool:
            for record in pool.map(process_file, jobs):
                records.append(record)
                print(json.dumps({k: record[k] for k in ("file", "input_points", "points", "status")}), flush=True)
    summary = summarize(records)
    manifest = dict(policy=policy, summary=summary, records=records,
                    seconds=time.perf_counter() - started, verified_all_five_fields=True)
    atomic_save(output / "preparation_manifest.json", manifest, json_file=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
