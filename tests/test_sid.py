# pylint: disable=redefined-outer-name
import pickle
import threading
import time

import numpy as np
import pytest

from conftest import TONE
from pylibresidfp import SID

CYCLES = 20000


def test_deterministic_across_instances(case):
    a, b = SID(case[0], method=case[1]), SID(case[0], method=case[1])
    first = a.run(TONE, CYCLES)
    assert first.dtype == np.int16 and np.abs(first).max() > 0
    assert np.array_equal(first, b.run(TONE, CYCLES))
    assert np.array_equal(first, SID(case[0], method=case[1]).run(TONE, CYCLES))


def test_deterministic_without_filter():
    runs = [SID(filter=False).run(TONE, CYCLES) for _ in range(3)]
    assert all(np.array_equal(runs[0], r) for r in runs)


def test_save_restore_tail(case):
    ref = SID(case[0], method=case[1])
    head = TONE[TONE[:, 0] < 6000]
    tail = TONE[TONE[:, 0] >= 6000]
    ref.run(head, 6000)
    mid = ref.copy()
    assert mid.cycle == 6000
    expect = ref.run(tail, CYCLES)
    assert np.array_equal(expect, mid.run(tail, CYCLES))


def test_pickle_roundtrip(case):
    a = SID(case[0], 1022730.0, 48000.0, case[1], filter=False)
    a.run(TONE[TONE[:, 0] < 6000], 6000)
    b = pickle.loads(pickle.dumps(a))
    assert b.params() == a.params() and b.cycle == a.cycle
    assert np.array_equal(a.clock(8000), b.clock(8000))


def test_copy_keeps_sampling_params():
    a = SID("8580", 1022730.0, 22050.0, "resample")
    b = a.copy()
    assert b.params() == a.params()
    assert len(b.clock(10000)) == len(a.clock(10000))


def test_restore_rejects_bad_state(sid):
    with pytest.raises(ValueError):
        sid.restore_state(b"short")


def test_run_equals_write_and_clock(sid):
    other = SID(sid.model, method=sid.method)
    out = [other.clock(0)]
    for cycle, reg, value in TONE:
        out.append(other.clock(cycle - other.cycle))
        other.write(reg, value)
    out.append(other.clock(CYCLES - other.cycle))
    assert np.array_equal(sid.run(TONE, CYCLES), np.concatenate(out))


def test_run_accepts_sequences_and_empty(sid):
    pairs = [tuple(int(x) for x in row) for row in TONE]
    other = SID(sid.model, method=sid.method)
    assert np.array_equal(sid.run(pairs, CYCLES), other.run(TONE, CYCLES))
    assert len(sid.run([], CYCLES + 100)) > 0
    assert sid.cycle == CYCLES + 100


def test_run_is_chunk_invariant(sid):
    other = SID(sid.model, method=sid.method)
    whole = sid.run(TONE, CYCLES)
    parts = np.concatenate(
        [
            other.run(TONE[TONE[:, 0] < 3000], 3000),
            other.run(TONE[TONE[:, 0] >= 3000], CYCLES),
        ]
    )
    assert np.array_equal(whole, parts)


def test_long_render_spanning_internal_chunks():
    span = 3 * (1 << 24) + 7
    whole = SID().run(TONE, span)
    other = SID()
    parts = np.concatenate([other.run(TONE, 1 << 24), other.clock(span - (1 << 24))])
    assert np.array_equal(whole, parts)


@pytest.mark.parametrize(
    "writes,until",
    [
        ([(5, 0, 1), (4, 0, 1)], 100),
        ([(101, 0, 1)], 100),
        ([(-1, 0, 1)], 100),
        ([(1, 32, 1)], 100),
        ([(1, 0, 256)], 100),
        ([(1, 0, 1)], -5),
        (np.zeros((2, 4), dtype=np.int64), 100),
        (np.zeros((2, 3), dtype=float), 100),
    ],
)
def test_run_validates(sid, writes, until):
    sid.clock(10)
    before = sid.cycle
    with pytest.raises(ValueError):
        sid.run(writes, until)
    assert sid.cycle == before


def test_run_rejects_cycles_before_now(sid):
    sid.clock(1000)
    with pytest.raises(ValueError):
        sid.run([(999, 0, 1)], 2000)
    with pytest.raises(ValueError):
        sid.run([], 999)
    with pytest.raises(ValueError):
        sid.clock(-1)


def test_write_at_until_is_allowed(sid):
    sid.run([(500, 0, 0x33)], 500)
    assert sid.cycle == 500


def test_serve_matches_run_then_read(sid):
    other = SID(sid.model, method=sid.method)
    other.run(TONE, 12000)
    assert sid.serve(TONE, 12000, 0x1B) == other.read(0x1B)
    assert sid.cycle == 12000
    assert np.array_equal(sid.clock(500), other.clock(500))


def test_register_basics(sid):
    sid.write(24, 15)
    sid.write(14, 0x40)
    sid.write(15, 0x20)
    sid.write(19, 0x00)
    sid.write(20, 0xF0)
    sid.write(18, 0x21)
    sid.clock(30000)
    assert sid.peek(0x1B) != 0 and sid.peek(0x1C) != 0
    assert sid.read(0x1B) == sid.peek(0x1B)
    sid.reset()
    sid.clock(8000)
    assert sid.peek(0x1B) == 0 and sid.peek(0x1C) == 0


def test_models_differ():
    a = SID("6581").run(TONE, CYCLES)
    b = SID("8580").run(TONE, CYCLES)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kwargs", [{"model": "6582"}, {"method": "linear"}])
def test_bad_strings(kwargs):
    with pytest.raises(ValueError):
        SID(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_hz": 0.0},
        {"sampling_hz": -1.0},
        {"clock_hz": 0.0},
        {"clock_hz": float("nan")},
        {"method": "resample", "sampling_hz": 4000.0},
    ],
)
def test_bad_sampling_parameters(kwargs):
    with pytest.raises(ValueError):
        SID(**kwargs)


def test_gate_toggle_within_a_frame_changes_output(case):
    setup = [(0, 24, 15), (0, 5, 0x00), (0, 6, 0xF0), (0, 1, 0x20), (0, 4, 0x41)]
    plain = SID(case[0], method=case[1]).run(setup, 19656)
    toggled = SID(case[0], method=case[1]).run(
        setup + [(4000, 4, 0x40), (4104, 4, 0x41)], 19656
    )
    assert not np.array_equal(plain, toggled)


def test_threads_match_serial(case):
    serial = SID(case[0], method=case[1]).run(TONE, 200000)
    results = [None, None]

    def work(i):
        results[i] = SID(case[0], method=case[1]).run(TONE, 200000)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(np.array_equal(serial, r) for r in results)


def test_concurrent_use_of_one_instance_raises():
    sid = SID()
    worker = threading.Thread(target=sid.clock, args=(100_000_000,))
    worker.start()
    try:
        deadline = time.time() + 30
        with pytest.raises(RuntimeError):
            while time.time() < deadline:
                sid.peek(0x1B)
    finally:
        worker.join()
    sid.peek(0x1B)


def test_voices_reports_written_state(sid):
    sid.run([(0, 5, 0x9A), (1, 6, 0x7C), (2, 4, 0x41), (3, 0, 0x34), (4, 1, 0x12)], 10)
    v = sid.voices()
    assert (v["attack"][0], v["decay"][0], v["release"][0]) == (9, 10, 12)
    assert v["sustain"][0] == 0x77
    assert (v["gate"][0], v["waveform"][0], v["freq"][0]) == (1, 4, 0x1234)
    assert not v["gate"][1:].any()


def test_trace_matches_per_cycle_reads():
    writes = [(0, 24, 15), (10, 5, 0x00), (20, 6, 0xF0), (30, 4, 0x41), (900, 4, 0x40)]
    traced = SID().trace(writes, 2000)
    other, seen = SID(), []
    pending = list(writes)
    for cycle in range(2001):
        while pending and pending[0][0] == cycle:
            other.write(*pending.pop(0)[1:])
        v = other.voices()
        seen += [(cycle, i, v["envelope"][i], v["state"][i]) for i in range(3)]
        other.clock(1)
    last, expect = {}, []
    for event in seen:
        if last.get(event[1]) != event[2:]:
            expect.append(event)
            last[event[1]] = event[2:]
    assert [tuple(e) for e in traced] == expect


def _lfsr_steps(lfsr: int, target: int) -> int:
    steps = 0
    while lfsr != target:
        lfsr = (lfsr >> 1) | (((lfsr << 14) ^ (lfsr << 13)) & 0x4000)
        steps += 1
    return steps


def _attack(prelude_release: int) -> tuple:
    """``(cycles from gate-on to the envelope rising, LFSR steps to the attack-0 rate)``."""
    sid = SID()
    sid.run([(0, 5, 0x00), (1, 6, prelude_release)], 20000)
    lfsr = int(sid.voices()["lfsr"][0])
    t = sid.trace([(20000, 4, 0x01)], 60000)
    start = t.envelope[t.voice == 0][0]
    rising = t[(t.voice == 0) & (t.state == 0) & (t.envelope > start)]
    return int(rising.cycle[0]) - 20000, _lfsr_steps(lfsr, 0x007F)


def test_adsr_delay_bug_is_modelled():
    prompt, _ = _attack(0x00)
    stalled, steps = _attack(0x0F)
    assert prompt < 64
    assert steps > 1000 and abs(stalled - steps) < 64
