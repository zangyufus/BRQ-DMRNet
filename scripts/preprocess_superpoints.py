"""Offline fixed-count Hilbert superpoints for five-field UrbanBIS PTH files.

By default only coordinates enter grouping. The explicit semantic separation
option uses ground truth and creates oracle assets. Save one int32 ID per point
in a <split>/<scene>/superpoint.npy layout. No downsampling is applied.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch


def _load_hilbert():
    # Import the Hilbert encoder directly instead of going through the model
    # registry, which pulls in the training networks and their CUDA deps.
    path = Path(__file__).resolve().parents[3] / "models/utils/serialization/hilbert.py"
    spec = importlib.util.spec_from_file_location("_urbanbis_hilbert", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode


hilbert_encode = _load_hilbert()
SEMANTIC_PARTITION = "urbanbis_building_background_ignore"
SEMANTIC_ORACLE_METHOD = "ptv3_hilbert_binary_semantic_fixed_count_oracle_v1"


def load_scene(path):
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, (tuple, list)) or len(values) != 5:
        raise ValueError(f"{path}: expected (coords, colors, semantic, instance, building_type)")
    arrays = tuple(
        value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        for value in values
    )
    n = len(arrays[0])
    if arrays[0].shape != (n, 3) or arrays[1].shape != (n, 3):
        raise ValueError(f"{path}: coordinate and color shapes must be (N, 3)")
    if any(value.shape != (n,) for value in arrays[2:]):
        raise ValueError(f"{path}: each label field must have shape (N,)")
    return arrays


def coordinate_digest(coords):
    array = np.ascontiguousarray(coords, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def binary_semantic_labels(semantic, instance):
    """Match the training interface: background=0, building=1, ignored=-1.

    Instance identity is not used for grouping; only annotation validity is used
    to keep unannotated building points ignored, as in the training converter.
    """
    semantic, instance = np.asarray(semantic), np.asarray(instance)
    if semantic.ndim != 1 or instance.shape != semantic.shape:
        raise ValueError("Semantic and instance labels must be aligned 1D arrays")
    if not np.isin(semantic, [-100, -1, 0, 1, 2, 3, 4, 5, 6]).all():
        raise ValueError("Unexpected UrbanBIS semantic labels")
    segment = np.full(semantic.shape, -1, dtype=np.int32)
    segment[(semantic >= 0) & (semantic < 6)] = 0
    segment[(semantic == 6) & (instance >= 0)] = 1
    return segment


def semantic_digest(segment):
    return hashlib.sha256(np.ascontiguousarray(segment, dtype=np.int32).tobytes()).hexdigest()


@torch.inference_mode()
def generate_superpoints(coords, points_per_superpoint=16, grid_size=0.2, order="hilbert"):
    """Return IDs in input point order and serialization metadata.

    All groups contain exactly K points except the last group, which may be
    smaller. Identical positions remain separate input points. Tied Hilbert
    codes retain input order, giving deterministic output for a fixed input.
    """
    if isinstance(points_per_superpoint, bool) or not isinstance(points_per_superpoint, int) or points_per_superpoint < 1:
        raise ValueError("points_per_superpoint must be a positive integer")
    if not np.isfinite(grid_size) or grid_size <= 0:
        raise ValueError("grid_size must be finite and positive")
    if order not in ("hilbert", "hilbert-trans"):
        raise ValueError("order must be hilbert or hilbert-trans")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or not np.isfinite(coords).all():
        raise ValueError("coords must be a finite, nonempty (N, 3) array")
    origin = coords.min(axis=0)
    scaled = np.floor((coords - origin) / grid_size)
    if scaled.max() >= 2**16:
        raise ValueError("Serialization depth exceeds PTv3's 16-bit limit; increase grid_size")
    grid = scaled.astype(np.int64)
    depth = max(1, int(grid.max()).bit_length())
    if order == "hilbert-trans":
        grid = np.ascontiguousarray(grid[:, [1, 0, 2]])
    code = hilbert_encode(torch.from_numpy(grid), num_dims=3, num_bits=depth).reshape(-1).numpy()
    permutation = np.argsort(code, kind="stable")
    group_count = (len(coords) + points_per_superpoint - 1) // points_per_superpoint
    if group_count > np.iinfo(np.int32).max:
        raise ValueError("Too many superpoints for int32 IDs")
    ids = np.empty(len(coords), dtype=np.int32)
    ids[permutation] = np.arange(len(coords), dtype=np.int64) // points_per_superpoint
    return ids, dict(serialization_depth=depth, origin_m=origin.tolist())


def validate_ids(ids, n, k):
    if ids.dtype != np.int32 or ids.shape != (n,) or np.any(ids < 0):
        raise ValueError("superpoint.npy must contain N nonnegative int32 IDs")
    groups = (n + k - 1) // k
    if int(ids.max()) != groups - 1:
        raise ValueError("Superpoint ID range does not match the requested point budget")
    counts = np.bincount(ids, minlength=groups)
    if not np.all(counts[:-1] == k) or counts[-1] != (n - (groups - 1) * k):
        raise ValueError("Superpoint group sizes do not match the requested point budget")
    return counts


def validate_class_separated_ids(ids, segment, k):
    """Check disjoint semantic partitions and one possible tail per class."""
    segment = np.asarray(segment)
    if segment.ndim != 1 or not len(segment) or not np.isin(segment, [-1, 0, 1]).all():
        raise ValueError("Expected nonempty labels in {-1, 0, 1}")
    if ids.dtype != np.int32 or ids.shape != segment.shape or np.any(ids < 0):
        raise ValueError("superpoint.npy must contain N nonnegative int32 IDs")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("points_per_superpoint must be a positive integer")
    offset, counts = 0, []
    for label in np.unique(segment):
        subset = ids[segment == label] - offset
        part_counts = validate_ids(subset, len(subset), k)
        counts.append(part_counts)
        offset += len(part_counts)
    return np.concatenate(counts)


def generate_class_separated_superpoints(coords, segment, points_per_superpoint=16,
                                        grid_size=0.2, order="hilbert"):
    """Filter the same scene-wide Hilbert sequence by class, then cut every K.

    Different buildings may share a superpoint. The only grouping labels are
    background/building/ignored; this is deliberately a ground-truth oracle.
    """
    segment = np.asarray(segment)
    if segment.shape != (len(coords),) or not np.isin(segment, [-1, 0, 1]).all():
        raise ValueError("Expected one binary/ignored semantic label per point")
    if isinstance(points_per_superpoint, bool) or not isinstance(points_per_superpoint, int) or points_per_superpoint < 1:
        raise ValueError("points_per_superpoint must be a positive integer")
    # K=1 IDs are the exact rank, including the stable tie-breaking rule.
    rank, details = generate_superpoints(coords, 1, grid_size, order)
    ids = np.empty(len(coords), dtype=np.int32)
    offset, partitions = 0, []
    names = {-1: "ignored", 0: "background", 1: "building"}
    for label in np.unique(segment):
        members = np.flatnonzero(segment == label)
        members = members[np.argsort(rank[members], kind="stable")]
        ids[members] = offset + np.arange(len(members)) // points_per_superpoint
        groups = (len(members) + points_per_superpoint - 1) // points_per_superpoint
        partitions.append(dict(label=int(label), name=names[int(label)], points=len(members),
                               superpoints=groups, first_superpoint=offset))
        offset += groups
    validate_class_separated_ids(ids, segment, points_per_superpoint)
    return ids, dict(details, semantic_partitions=partitions)


def atomic_save(path, value, *, json_file=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            if json_file:
                handle.write((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
            else:
                np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_scene(job):
    source, dataset_root, output_root, settings, overwrite = job
    torch.set_num_threads(1)
    started = time.perf_counter()
    coords, _, semantic, instance, _ = load_scene(source)
    separated = settings.get("semantic_partition") == SEMANTIC_PARTITION
    if "semantic_partition" in settings and not separated:
        raise ValueError("Unknown semantic partition")
    method = SEMANTIC_ORACLE_METHOD if separated else "ptv3_hilbert_fixed_count_v1"
    segment = binary_semantic_labels(semantic, instance) if separated else None
    partition_digest = semantic_digest(segment) if separated else None
    city, split, filename = source.relative_to(dataset_root).parts
    output = output_root / split / f"{city}_{Path(filename).stem}"
    array_path, meta_path = output / "superpoint.npy", output / "superpoint.json"
    digest = coordinate_digest(coords)
    if not overwrite and (array_path.exists() or meta_path.exists()):
        if not array_path.exists() or not meta_path.exists():
            raise FileExistsError(f"Incomplete output at {output}; rerun with --overwrite")
        metadata = json.loads(meta_path.read_text())
        if (metadata.get("settings") != settings or metadata.get("coordinate_sha256") != digest
                or metadata.get("method") != method
                or (separated and metadata.get("semantic_partition_sha256") != partition_digest)):
            raise FileExistsError(f"Different input/settings at {output}; choose another output_root or --overwrite")
        ids = np.load(array_path, allow_pickle=False)
        status = "existing"
    else:
        options = {key: settings[key] for key in ("points_per_superpoint", "grid_size", "order")}
        if separated:
            ids, details = generate_class_separated_superpoints(coords, segment, **options)
        else:
            ids, details = generate_superpoints(coords, **options)
        metadata = dict(
            method=method, source=str(source),
            source_relative=str(source.relative_to(dataset_root)),
            settings=settings, points=len(coords), superpoints=int(ids.max()) + 1,
            coordinate_sha256=digest, point_order="same as source PTH",
            gt_used_for_generation=separated, downsampled=False, **details,
        )
        if separated:
            metadata.update(oracle_only=True, semantic_partition_sha256=partition_digest,
                            generation_inputs=["coords", "semantic_labels", "instance_annotation_validity"],
                            instance_identity_used_for_grouping=False)
        atomic_save(array_path, ids)
        atomic_save(meta_path, metadata, json_file=True)
        status = "prepared"
    counts = (validate_class_separated_ids(ids, segment, settings["points_per_superpoint"])
              if separated else validate_ids(ids, len(coords), settings["points_per_superpoint"]))
    return dict(source=str(source.relative_to(dataset_root)), output=str(array_path),
                status=status, points=len(coords), superpoints=len(counts),
                min_group_points=int(counts.min()), max_group_points=int(counts.max()),
                last_group_points=int(counts[-1]), seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--points_per_superpoint", default=16, type=int)
    parser.add_argument("--grid_size", default=0.2, type=float, help="Hilbert coordinate quantization in metres; no downsampling")
    parser.add_argument("--order", default="hilbert", choices=("hilbert", "hilbert-trans"))
    parser.add_argument("--separate_semantic_classes", action="store_true",
                        help="GT oracle: partition background/building/ignored separately before fixed-K grouping")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--cities", nargs="+")
    parser.add_argument("--limit", type=int, help="Process only the first N selected files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1 or args.points_per_superpoint < 1 or (args.limit is not None and args.limit < 1):
        parser.error("num_workers, points_per_superpoint and limit must be positive")
    if not np.isfinite(args.grid_size) or args.grid_size <= 0:
        parser.error("grid_size must be finite and positive")
    root, output = args.dataset_root.resolve(), args.output_root.resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")
    cities = args.cities or sorted(p.name for p in root.iterdir() if p.is_dir())
    sources = sorted({p for city in cities for split in args.splits for p in (root / city / split).glob("*.pth")})
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        parser.error("No PTH files matched dataset_root/<city>/<split>/*.pth")
    settings = dict(points_per_superpoint=args.points_per_superpoint, grid_size=args.grid_size, order=args.order)
    if args.separate_semantic_classes:
        settings["semantic_partition"] = SEMANTIC_PARTITION
    jobs = [(source, root, output, settings, args.overwrite) for source in sources]
    if args.num_workers == 1:
        records = []
        for job in jobs:
            record = process_scene(job)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context("spawn")) as pool:
            records = []
            for record in pool.map(process_scene, jobs):
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
    inputs = (["coords", "semantic_labels", "instance_annotation_validity"]
              if args.separate_semantic_classes else ["coords"])
    manifest = dict(dataset_root=str(root), settings=settings, generation_inputs=inputs,
                    gt_used_for_generation=args.separate_semantic_classes, files=len(records),
                    points=sum(r["points"] for r in records),
                    superpoints=sum(r["superpoints"] for r in records), records=records)
    if args.separate_semantic_classes:
        manifest.update(oracle_only=True, instance_identity_used_for_grouping=False)
    atomic_save(output / "preparation_manifest.json", manifest, json_file=True)


if __name__ == "__main__":
    main()
