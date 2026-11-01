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
    ): "20d5f61f483b3359fce0c0c4f6ccefd462e45c496e013705d8e25594e37e553f",
    (
        "6581",
        "resample",
    ): "b8e76b2842bec32b04b9a4e103c846f21e6beed18fbd35ddafa916f56e6c018d",
    (
        "8580",
        "decimate",
    ): "731242ce0c046fbe55c9696a517f7ab76d0aff74726a003d74931939dd9bbc6b",
    (
        "8580",
        "resample",
    ): "3099487a16ec9e6d9dbf13ef150112278628cb11b51e0cf41929941be0813e0d",
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
    digest = hashlib.sha256(audio.tobytes() + sid.save_state()).hexdigest()
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
