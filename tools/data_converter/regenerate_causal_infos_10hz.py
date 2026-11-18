"""Regenerate ETRI train/test info PKLs and the configured train/val split,
10Hz ego-motion variant.

Sibling of regenerate_causal_infos.py, differing only in which converter
modules it delegates to: etri_vad_converter_10hz.py / etri_test_converter_10hz.py,
which sample ego pose for velocity/acceleration at every raw 0.1s frame
instead of every TRAJ_STEP(0.5s) frame; this does not require a matching
fed image. Pass a distinct
--prefix (e.g. "vad_etri_10hz") so the output pkls don't collide with the
non-10hz ones.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path
from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


def install_mmcv_fallback() -> None:
    """Provide only the file/progress helpers used by the converters."""
    try:
        import mmcv  # noqa: F401
        return
    except ImportError:
        pass

    fallback = types.ModuleType("mmcv")

    def track_iter_progress(items: Iterable[T]) -> Iterator[T]:
        sequence = list(items)
        total = len(sequence)
        for index, item in enumerate(sequence, 1):
            if index == 1 or index % 10 == 0 or index == total:
                print(f"  progress: {index}/{total}", flush=True)
            yield item

    def mkdir_or_exist(path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def dump(data: object, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            pickle.dump(data, stream)

    fallback.track_iter_progress = track_iter_progress
    fallback.mkdir_or_exist = mkdir_or_exist
    fallback.dump = dump
    sys.modules["mmcv"] = fallback


install_mmcv_fallback()

import etri_test_converter_10hz as test_converter
import etri_vad_converter_10hz as train_converter


def split_train_infos(
    full_path: Path,
    val_scenes_path: Path,
    output_dir: Path,
    prefix: str,
) -> None:
    with full_path.open("rb") as stream:
        data = pickle.load(stream)

    val_scenes = {
        line.strip()
        for line in val_scenes_path.read_text().splitlines()
        if line.strip()
    }
    if not val_scenes:
        raise ValueError(f"No validation scenes in {val_scenes_path}")

    infos = data["infos"]
    known_scenes = {info["scene_token"] for info in infos}
    missing = val_scenes - known_scenes
    if missing:
        raise ValueError(f"Unknown validation scenes: {sorted(missing)}")

    train_infos = [
        info for info in infos if info["scene_token"] not in val_scenes
    ]
    val_infos = [
        info for info in infos if info["scene_token"] in val_scenes
    ]

    outputs = {
        output_dir / f"{prefix}_infos_temporal_train_split.pkl": train_infos,
        output_dir / f"{prefix}_infos_temporal_val_split.pkl": val_infos,
    }
    for path, split_infos in outputs.items():
        with path.open("wb") as stream:
            pickle.dump(
                {"infos": split_infos, "metadata": data["metadata"]},
                stream,
            )
        print(f"wrote {path} ({len(split_infos)} infos)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-scenes", type=Path, required=True)
    parser.add_argument("--prefix", default="vad_etri")
    parser.add_argument("--ego-length", type=float, default=4.635)
    parser.add_argument("--ego-width", type=float, default=1.890)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating full train infos", flush=True)
    train_converter.create_etri_infos(
        str(args.train_root),
        str(args.output_dir),
        args.prefix,
        args.ego_length,
        args.ego_width,
    )

    full_path = (
        args.output_dir / f"{args.prefix}_infos_temporal_train.pkl"
    )
    print("Generating train/validation split", flush=True)
    split_train_infos(
        full_path,
        args.val_scenes,
        args.output_dir,
        args.prefix,
    )

    print("Generating test infos", flush=True)
    test_converter.create_test_infos(
        str(args.test_root),
        str(args.output_dir),
        args.prefix,
    )


if __name__ == "__main__":
    main()
