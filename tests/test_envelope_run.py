import time

import numpy as np
import pytest

from pylibresidfp import SID, envelope_run, envelope_runs

POOLS = (range(0, 9), range(9, 13), range(13, 16))


def _gap(rng):
    lo, hi = [(0, 2), (1, 9), (9, 200), (200, 20000), (20000, 90000)][
        int(rng.choice(5, p=[0.2, 0.25, 0.2, 0.25, 0.1]))
    ]
    return int(rng.integers(lo, hi))


def _chip(rng):
    pool = POOLS[int(rng.integers(3))]
    sid = SID()
    sid.write(5, int(rng.choice(pool)) << 4 | int(rng.choice(pool)))
    sid.write(6, int(rng.integers(0, 16)) << 4 | int(rng.choice(pool)))
    for _ in range(int(rng.integers(1, 4))):
        sid.write(4, 0x40 | int(rng.integers(0, 2)))
        sid.clock(_gap(rng) + 3)
    return sid


def _writes(rng, start):
    pool = POOLS[int(rng.integers(3))]
    writes, now = [], start
    for _ in range(int(rng.integers(0, 14))):
        now += _gap(rng)
        reg = int(rng.choice([4, 4, 5, 6, 6, 0, 11, 12]))
        value = int(rng.integers(0, 256))
        if reg == 4:
            value = 0x40 | int(rng.integers(0, 2)) | value & 0x0E
        elif reg == 5:
            value = int(rng.choice(pool)) << 4 | int(rng.choice(pool))
        writes.append((now, reg, value))
    return writes, now + _gap(rng) + 1


@pytest.mark.parametrize("seed", range(60))
def test_matches_envelope_trace_and_chip(seed):
    rng = np.random.default_rng(seed)
    sid = _chip(rng)
    start = sid.cycle
    state = sid.envelope_state(0)
    writes, until = _writes(rng, start)
    trace, end = envelope_run(state, writes, until, start)
    want = sid.envelope_trace(0, writes, until)
    assert np.array_equal(trace.cycle, want.cycle)
    assert np.array_equal(trace.envelope, want.envelope)
    assert np.array_equal(trace.state, want.state)
    sid.run([w for w in writes if w[0] < until], until)
    assert end == sid.envelope_state(0)


@pytest.mark.parametrize("seed", range(20))
def test_continues_mid_stream_as_the_chip_does(seed):
    rng = np.random.default_rng(1000 + seed)
    sid = _chip(rng)
    writes, until = _writes(rng, sid.cycle)
    split = sid.cycle + (until - sid.cycle) // 2
    first = [w for w in writes if w[0] < split]
    rest = [w for w in writes if w[0] >= split]
    state = sid.envelope_state(0)
    start = sid.cycle
    _, mid = envelope_run(state, first, split, start)
    sid.run(first, split)
    assert mid == sid.envelope_state(0)
    trace, end = envelope_run(mid, rest, until, split)
    want = sid.envelope_trace(0, rest, until)
    assert np.array_equal(trace.cycle, want.cycle)
    assert np.array_equal(trace.envelope, want.envelope)
    sid.run(rest, until)
    assert end == sid.envelope_state(0)


def test_batch_equals_singles():
    rng = np.random.default_rng(7)
    sid = _chip(rng)
    state, start = sid.envelope_state(1), sid.cycle
    cases = [_writes(rng, start) for _ in range(5)]
    until = max(u for _, u in cases)
    batch = envelope_runs(state, [w for w, _ in cases], until, start)
    for (writes, _), (trace, end) in zip(cases, batch):
        one, one_end = envelope_run(state, writes, until, start)
        assert np.array_equal(trace, one) and end == one_end
    assert envelope_runs(state, [], until, start) == []


def test_state_bytes_deterministic():
    states = {_chip(np.random.default_rng(3)).envelope_state(2) for _ in range(4)}
    assert len(states) == 1
    sid = SID()
    assert sid.envelope_state(0) == sid.copy().envelope_state(0)
    assert sid.envelope_state(0) == SID("8580").envelope_state(0)
    sid.write(4, 1)
    sid.clock(50)
    assert sid.envelope_state(0) != SID().envelope_state(0)


def test_empty_and_idle_runs():
    sid = SID()
    state = sid.envelope_state(0)
    trace, end = envelope_run(state, np.empty((0, 3), np.int64), 0, 0)
    assert len(trace) == 1 and end == state
    trace, end = envelope_run(state, [], 10**7, 0)
    sid.clock(10**7)
    assert end == sid.envelope_state(0)


def test_rejects_bad_input():
    state = SID().envelope_state(0)
    with pytest.raises(ValueError, match="size"):
        envelope_run(state[:-1], [], 10, 0)
    with pytest.raises(ValueError, match="invalid"):
        envelope_run(bytes(len(state)), [], 10, 0)
    with pytest.raises(ValueError, match="until"):
        envelope_run(state, [], 5, 10)
    with pytest.raises(ValueError, match="precede"):
        envelope_run(state, [(5, 4, 1)], 20, 10)
    with pytest.raises(ValueError, match="after until"):
        envelope_run(state, [(30, 4, 1)], 20, 10)
    with pytest.raises(ValueError, match="register"):
        envelope_run(state, [(15, 40, 1)], 20, 10)
    with pytest.raises(ValueError, match="value"):
        envelope_run(state, [(15, 4, 300)], 20, 10)
    with pytest.raises(ValueError, match="shape"):
        envelope_run(state, np.zeros((2, 2), np.int64), 20, 10)
    with pytest.raises(ValueError, match="voice"):
        SID().envelope_state(3)


def test_much_faster_than_envelope_trace():
    sid = SID()
    sid.write(5, 0xEE)
    sid.write(6, 0xDE)
    sid.clock(10)
    state, start = sid.envelope_state(0), sid.cycle
    frame = 19656
    writes = [
        (start + k * frame + off, 4, v)
        for k in range(300)
        for off, v in ((0, 0x41), (frame // 2, 0x40))
    ]
    until = start + 300 * frame

    def best(call, reps=5):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            result = call()
            times.append(time.perf_counter() - t0)
        return min(times), result

    fast, (trace, _) = best(lambda: envelope_run(state, writes, until, start))
    slow, want = best(lambda: sid.envelope_trace(0, writes, until))
    assert np.array_equal(trace.cycle, want.cycle)
    assert slow > 20 * fast
