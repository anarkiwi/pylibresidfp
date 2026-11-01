import hashlib
import pickle

import numpy as np
import pytest

from conftest import CASES
from pylibresidfp import SID

BLOCK = 2048

RELEASED = {
    (
        "6581",
        "decimate",
    ): "469482d379ef94ef01727919faafd951f33ba1d9f368fabee3a4b56c5f3d123d",
    (
        "6581",
        "resample",
    ): "c246c8ecb06ed8cf6dc5cf87dc091fde9e9914aaba38acec935eac2ffe64a933",
    (
        "8580",
        "decimate",
    ): "6d87b1799b224c5a903525d0b42a689e1a1fdea7d5215141d20304cf7f4848ed",
    (
        "8580",
        "resample",
    ): "3e8bf203845282cd1888b224dcd1bfd40f4b4883f5e4d3a93289076b2d577d70",
}


def program(cycles, n, seed, start=0):
    rng = np.random.default_rng(seed)
    w = np.stack(
        [
            np.sort(rng.integers(start, cycles, n)),
            rng.integers(0, 25, n),
            rng.integers(0, 256, n),
        ],
        1,
    )
    head = np.array([[start, 24, 0x1F], [start, 23, 0xF7]])
    return np.concatenate([head, w]).astype(np.int64)


def session(sid, w, cuts):
    out, start = [], sid.cycle
    for end in cuts:
        out.append(sid.run(w[(w[:, 0] >= start) & (w[:, 0] < end)], end))
        start = end
    out.append(sid.clock(BLOCK + 7))
    return np.concatenate(out), sid.save_state(), sid.read(27), sid.read(28)


@pytest.mark.parametrize("case", CASES, ids="-".join)
def test_serial_matches_released_output(case):
    sid = SID(case[0], method=case[1])
    audio = sid.run(program(300000, 1500, 7), 300000)
    digest = hashlib.sha256(audio.tobytes()).hexdigest()
    assert digest == RELEASED[case]


@pytest.mark.parametrize("filt", [True, False])
@pytest.mark.parametrize("case", CASES, ids="-".join)
def test_pipeline_matches_serial(case, filt):
    w = program(400000, 3000, 1)
    cuts = (BLOCK, 3 * BLOCK + 1, 150000, 400000)
    serial = session(SID(case[0], method=case[1], filter=filt), w, cuts)
    piped = session(SID(case[0], method=case[1], filter=filt, pipeline=True), w, cuts)
    assert np.array_equal(serial[0], piped[0]) and np.abs(serial[0]).max() > 0
    assert serial[1:] == piped[1:]


@pytest.mark.parametrize("case", CASES, ids="-".join)
def test_pipeline_filter_writes_on_block_edges(case):
    edges = np.arange(1, 9) * BLOCK
    regs = np.resize([21, 22, 23, 24, 4], edges.size)
    w = np.concatenate(
        [
            np.array([[0, 24, 0x1F], [0, 1, 0x20], [0, 6, 0xF0], [0, 4, 0x21]]),
            np.stack([edges, regs, (edges // BLOCK * 37) & 0xFF], 1),
        ]
    ).astype(np.int64)
    end = int(edges[-1])
    serial, piped = SID(case[0], method=case[1]), SID(
        case[0], method=case[1], pipeline=True
    )
    assert np.array_equal(serial.run(w, end), piped.run(w, end))
    assert serial.save_state() == piped.save_state()
    assert np.array_equal(serial.clock(BLOCK), piped.clock(BLOCK))


def test_pipeline_resumes_restored_state():
    w = program(200000, 800, 3)
    head, tail = w[w[:, 0] < 90000], w[w[:, 0] >= 90000]
    src = SID("6581", method="resample")
    src.run(head, 90000)
    serial = SID("6581", method="resample")
    piped = SID("6581", method="resample", pipeline=True)
    for sid in (serial, piped):
        sid.restore_state(src.save_state())
    assert piped.pipeline and piped.cycle == 90000
    assert np.array_equal(serial.run(tail, 200000), piped.run(tail, 200000))
    assert serial.save_state() == piped.save_state()


def test_pipeline_writes_without_cycles():
    w = np.array([[0, 24, 0x1F], [0, 23, 0xF1], [0, 4, 0x21]], np.int64)
    serial, piped = SID(), SID(pipeline=True)
    assert serial.run(w, 0).size == piped.run(w, 0).size == 0
    assert piped.clock(0).size == 0
    assert serial.save_state() == piped.save_state()


def test_copy_and_pickle_keep_pipeline():
    sid = SID("8580", method="resample", pipeline=True)
    sid.run(program(50000, 200, 5), 50000)
    for other in (sid.copy(), pickle.loads(pickle.dumps(sid))):
        assert other.pipeline and other.params() == sid.params()
        assert np.array_equal(other.clock(30000), sid.copy().clock(30000))
