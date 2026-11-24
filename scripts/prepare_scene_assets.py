"""Convert sampled UrbanBIS PTHs and fixed superpoints to per-scene assets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from preprocess_superpoints import (
    SEMANTIC_ORACLE_METHOD, atomic_save, binary_semantic_labels, coordinate_digest,
    load_scene, semantic_digest, validate_class_separated_ids, validate_ids,
)


def convert_labels(semantic, instance):
    """Map UrbanBIS building 6 to class 1; preserve instance zero as valid."""
    semantic, instance = np.asarray(semantic), np.asarray(instance)
    segment = binary_semantic_labels(semantic, instance)
    valid = segment == 1
    # Building points without an instance annotation remain ignored. This keeps
    # -1 out of process_instance's foreground reindexing in the existing model.
    raw_ids, inverse = np.unique(instance[valid], return_inverse=True)
    mapped = np.full(instance.shape, -1, dtype=np.int32)
    mapped[valid] = inverse.astype(np.int32)
    return segment, mapped, raw_ids


def convert_scene(job):
    source, source_root, superpoint_root, output_root, overwrite = job
    coord, color, semantic, instance, building_type = load_scene(source)
    if not np.isfinite(coord).all() or not np.isfinite(color).all():
        raise ValueError(f"Nonfinite coordinates/colors in {source}")
    if color.min() < -1.001 or color.max() > 1.001:
        raise ValueError(f"Expected input RGB in [-1, 1]: {source}")
    relative = source.relative_to(source_root)
    city, split, _ = relative.parts
    scene_name = f"{city}_{source.stem}"
    sp_dir = superpoint_root / split / scene_name
    sp_meta = json.loads((sp_dir / "superpoint.json").read_text())
    digest = coordinate_digest(coord)
    if sp_meta["coordinate_sha256"] != digest or Path(sp_meta["source"]).resolve() != source.resolve():
        raise ValueError(f"Superpoints do not match source coordinates/order: {source}")
    superpoint = np.load(sp_dir / "superpoint.npy", allow_pickle=False)
    segment, mapped_instance, raw_ids = convert_labels(semantic, instance)
    oracle = sp_meta.get("method") == SEMANTIC_ORACLE_METHOD
    if oracle:
        if not sp_meta.get("gt_used_for_generation") or sp_meta.get("semantic_partition_sha256") != semantic_digest(segment):
            raise ValueError(f"Semantic oracle metadata/labels do not match source: {source}")
        validate_class_separated_ids(superpoint, segment, sp_meta["settings"]["points_per_superpoint"])
    else:
        validate_ids(superpoint, len(coord), sp_meta["settings"]["points_per_superpoint"])
    assets = dict(
        coord=np.ascontiguousarray(coord, dtype=np.float32),
        # The color normalizer divides by 255. Keep float32 here to avoid an
        # unnecessary uint8 quantization of the original RGB values.
        color=np.ascontiguousarray((color + 1.0) * 127.5, dtype=np.float32),
        segment=segment, instance=mapped_instance, superpoint=superpoint,
        building_type=building_type.astype(np.int32),
    )
    index_path = source.with_suffix(".source_indices.npy")
    if index_path.is_file():
        assets["source_index"] = np.load(index_path, allow_pickle=False)
        if assets["source_index"].shape != (len(coord),):
            raise ValueError(f"Source indices do not align: {source}")
    destination = output_root / split / scene_name
    metadata = dict(
        format="pointcept_urbanbis_building_v1", source=str(source),
        superpoint_source=str(sp_dir / "superpoint.npy"), coordinate_sha256=digest,
        points=len(coord), superpoints=int(superpoint.max()) + 1,
        raw_instance_ids=raw_ids.tolist(), semantic_map={"background": 0, "building": 1, "ignore": -1},
        instance_ignore_index=-1, color_range=[0, 255],
        point_order="same as sampled source PTH", superpoint_settings=sp_meta["settings"],
    )
    if oracle:
        metadata.update(superpoint_method=SEMANTIC_ORACLE_METHOD, gt_used_for_superpoints=True,
                        oracle_only=True, semantic_partition_sha256=semantic_digest(segment))
    meta_path = destination / "metadata.json"
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        if not meta_path.is_file() or json.loads(meta_path.read_text()) != metadata:
            raise FileExistsError(f"Different/incomplete output at {destination}; use --overwrite")
        status = "existing"
    else:
        for name, array in assets.items():
            atomic_save(destination / f"{name}.npy", array)
        atomic_save(meta_path, metadata, json_file=True)
        status = "prepared"
    for name, expected in assets.items():
        saved = np.load(destination / f"{name}.npy", allow_pickle=False)
        if saved.dtype != expected.dtype:
            raise ValueError(f"Incorrect dtype for {destination}/{name}.npy")
        np.testing.assert_array_equal(saved, expected)
    return dict(scene=f"{split}/{scene_name}", source=relative.as_posix(),
                points=len(coord), superpoints=metadata["superpoints"],
                building_instances=len(raw_ids), building_points=int((segment == 1).sum()),
                ignored_points=int((segment == -1).sum()), status=status,
                gt_used_for_superpoints=oracle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--superpoint_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root, sp_root, output = (p.resolve() for p in (args.dataset_root, args.superpoint_root, args.output_root))
    if args.num_workers < 1:
        parser.error("num_workers must be positive")
    if output == root or output == sp_root or root in output.parents or sp_root in output.parents:
        parser.error("Use a separate output directory")
    paths = sorted(root.glob("*/*/*.pth"))
    if not paths:
        parser.error("No UrbanBIS PTH files found")
    jobs = [(p, root, sp_root, output, args.overwrite) for p in paths]
    records = []
    if args.num_workers == 1:
        iterator = map(convert_scene, jobs)
        for record in iterator:
            records.append(record)
            print(json.dumps(record), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp.get_context("spawn")) as pool:
            for record in pool.map(convert_scene, jobs):
                records.append(record)
                print(json.dumps(record), flush=True)
    summary = {}
    for split in sorted({Path(r["scene"]).parts[0] for r in records}):
        selected = [r for r in records if Path(r["scene"]).parts[0] == split]
        summary[split] = dict(scenes=len(selected), points=sum(r["points"] for r in selected),
                              building_instances=sum(r["building_instances"] for r in selected))
    manifest = dict(dataset_root=str(root), superpoint_root=str(sp_root), output_root=str(output),
                    semantic_names=["background", "building"], summary=summary, records=records)
    if any(r["gt_used_for_superpoints"] for r in records):
        manifest.update(gt_used_for_superpoints=True, oracle_only=True)
    atomic_save(output / "preparation_manifest.json", manifest, json_file=True)
    # The dataset loader accepts JSON scene lists; these are useful for a
    # bounded training/validation smoke test without changing any trainer code.
    for split in ("train", "val"):
        selected = [r for r in records if Path(r["scene"]).parts[0] == split]
        if selected:
            smallest = min(selected, key=lambda r: r["points"])
            atomic_save(output / f"smoke_{split}.json", [smallest["scene"]], json_file=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
