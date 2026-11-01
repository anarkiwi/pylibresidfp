"""Realtime factor of SID.run, serial against pipelined, for each model and method."""

import argparse
import time

import numpy as np
from tqdm import tqdm

from pylibresidfp import SID

TONE = np.array(
    [
        (0, 24, 0x1F),
        (0, 23, 0xF7),
        (0, 1, 0x10),
        (0, 5, 0x09),
        (0, 6, 0xF0),
        (0, 4, 0x21),
        (0, 8, 0x18),
        (0, 13, 0xF0),
        (0, 11, 0x41),
        (0, 15, 0x0C),
        (0, 20, 0xF0),
        (0, 18, 0x11),
    ],
    dtype=np.int64,
)


def realtime(sid, seconds, chunks):
    edges = np.linspace(0, round(sid.clock_hz * seconds), chunks + 1).astype(np.int64)
    sid.run(TONE, 0)
    start = time.perf_counter()
    for until in edges[1:]:
        sid.run(np.zeros((0, 3), np.int64), int(until))
    return seconds / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--sampling-hz", type=float, default=48000.0)
    args = parser.parse_args()
    cases = [
        (model, method, pipeline)
        for model in ("6581", "8580")
        for method in ("resample", "decimate")
        for pipeline in (False, True)
    ]
    rows = []
    for model, method, pipeline in tqdm(cases, unit="case"):
        sid = SID(model, sampling_hz=args.sampling_hz, method=method, pipeline=pipeline)
        rows.append((model, method, pipeline, realtime(sid, args.seconds, args.chunks)))
    for model, method, pipeline, factor in rows:
        mode = "pipeline" if pipeline else "serial"
        print(f"{model} {method:8} {mode:8} {factor:6.1f}x realtime")


if __name__ == "__main__":
    main()
