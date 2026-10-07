"""Zip a finished submission and refuse to ship a malformed one.

The inner filename is fixed to submission.json because that is what the
accepted 2026-09-18 submission carried, and the scorer reads the archive by
that name rather than by whatever the file was called on disk.

The assertions here exist because a submission was rejected on 2026-09-23 with
CUTOFF_RESULT_MISSING / "__flops__ field is required": the trajectories alone
are not a submission. The scorer derives its cutoff boolean from __flops__, so
a file without it fails after upload rather than here, which costs one of the
few attempts. Checking before packing is the whole point.
"""
import json
import os
import zipfile

out = os.environ["OUT"]
tag = os.environ["TAG"]
zip_path = f"submission_{tag}.zip"

d = json.load(open(out))
scenes = [k for k in d if not k.startswith("__")]

assert "__flops__" in d, (
    "__flops__ is missing -- run tools/measure_flops.py --out on this file "
    "before packing, or the scorer rejects it with CUTOFF_RESULT_MISSING"
)
assert len(scenes) == 1125, f"expected 1125 scenes, found {len(scenes)}"
assert {len(d[k]) for k in scenes} == {6}, "some clip does not have 6 waypoints"
assert {len(p) for k in scenes for p in d[k]} == {2}, "some waypoint is not 2-D"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out, arcname="submission.json")

print(f"  전체 키   : {len(d)}")
print(f"  __flops__ : {d['__flops__']}")
print(f"  scene     : {len(scenes)}")
print(f"  zip 안    : submission.json  ({zip_path})")
