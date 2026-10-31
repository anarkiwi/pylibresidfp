import time

import numpy as np
import pytest

from pylibresidfp import SID, Phases

FRAME = 19656
POOLS = {
    "fast": range(0, 9),
    "mid": range(9, 13),
    "slow": range(13, 16),
}
RATE_PERIODS = {0: 9, 1: 32, 8: 392, 15: 31251}


def _gap(rng):
    return int(
        rng.choice(
            [
                0,
                rng.integers(1, 9),
                rng.integers(9, 200),
                rng.integers(200, 20000),
            ],
            p=[0.2, 0.3, 0.25, 0.25],
        )
    )


def _control(rng):
    return int(rng.integers(0, 256) & rng.choice([0xFF, 0xF1, 0x09, 0x01, 0x00]))


def _chip(rng, voice, pool, release):
    sid = SID()
    base = 7 * voice
    sid.write(base + 5, int(rng.choice(pool)) << 4 | int(rng.choice(pool)))
    sr = int(rng.integers(0, 16)) << 4 | int(rng.choice(pool))
    gates = [int(rng.integers(0, 2)), int(rng.integers(0, 2))]
    if release is not None:
        sr, gates = sr & 0xF0 | release, [1, 0]
    sid.write(base + 6, sr)
    for gate in gates:
        sid.write(base + 4, 0x40 | gate)
        sid.clock(int(rng.integers(3, 5000)))
    return sid


def _writes(rng, voice, start, window):
    base = 7 * voice
    writes, now = [], start
    for _ in range(int(rng.integers(2, 12))):
        now += _gap(rng)
        if now - start > window:
            break
        reg = int(rng.choice([4, 4, 5, 6, 6, -1]))
        if reg < 0:
            other = int(rng.integers(0, 4)) + 7 * ((voice + 1) % 3)
            writes.append((now, other, 7))
        else:
            value = _control(rng) if reg == 4 else int(rng.integers(0, 256))
            writes.append((now, base + reg, value))
    until = start + int(rng.integers(1, window + 1))
    return writes, (max(now, until) if rng.random() < 0.8 else now)


def _case(seed, nibbles, window, release=None):
    """A chip whose voice sits in a random envelope state at a rate drawn from `nibbles`, and a
    random write sequence for that voice (with writes to other registers mixed in)."""
    rng = np.random.default_rng(seed)
    voice = int(rng.integers(0, 3))
    sid = _chip(rng, voice, list(nibbles), release)
    return (sid, voice, *_writes(rng, voice, sid.cycle, window))


def _check(sid, voice, writes, until):
    phases = sid.envelope_phases(voice, writes, until)
    for k in range(phases.count):
        got = phases.trajectory(k)
        want = sid.envelope_trace(voice, writes, until, int(phases.states[k]))
        assert got.dtype == want.dtype, k
        for name in want.dtype.names:
            assert np.array_equal(got[name], want[name]), (k, name)
    return phases


@pytest.mark.parametrize("seed", range(40))
def test_matches_envelope_trace_for_every_phase_small_period(seed):
    sid, voice, writes, until = _case(seed, POOLS["fast"], 3 * FRAME)
    assert _check(sid, voice, writes, until).count <= 392


@pytest.mark.parametrize("seed", range(100, 108))
def test_matches_envelope_trace_for_every_phase_mid_period(seed):
    sid, voice, writes, until = _case(seed, POOLS["mid"], 3 * FRAME)
    assert 392 < _check(sid, voice, writes, until).count <= 3907


@pytest.mark.parametrize("nibble", sorted(RATE_PERIODS))
@pytest.mark.parametrize("seed", range(200, 203))
def test_matches_envelope_trace_at_each_release_rate(nibble, seed):
    window = FRAME if nibble == 15 else 3 * FRAME
    sid, voice, writes, until = _case(seed, [nibble], window, release=nibble)
    assert _check(sid, voice, writes, until).count == RATE_PERIODS[nibble]


def test_matches_envelope_trace_full_period_over_three_frames():
    sid = SID()
    sid.run([(0, 5, 0xFF), (1, 6, 0xFF), (2, 4, 0x01)], 4000)
    start = sid.cycle
    writes = [
        (start + 3, 4, 0x00),
        (start + 9000, 6, 0xF0),
        (start + 9001, 4, 0x01),
        (start + 30000, 4, 0x08),
        (start + 30002, 5, 0x00),
        (start + 41000, 4, 0x01),
    ]
    phases = _check(sid, 0, writes, start + 3 * FRAME)
    assert phases.count == 31251


def test_no_writes_and_empty_window():
    sid = SID()
    sid.run([(0, 5, 0x00), (1, 6, 0x00), (2, 4, 0x01)], 50)
    for until in (sid.cycle, sid.cycle + 1, sid.cycle + 5000):
        phases = _check(sid, 0, [], until)
        assert phases.trajectory(0).cycle[0] == sid.cycle


def test_chip_is_unchanged():
    sid = SID()
    sid.run([(0, 5, 0x0A), (1, 6, 0x88), (2, 4, 0x01)], 500)
    before = (
        sid.cycle,
        sid.save_state(),
        {k: v.copy() for k, v in sid.voices().items()},
    )
    sid.envelope_phases(0, [(sid.cycle + 4, 4, 0x00)], sid.cycle + 3000)
    assert sid.cycle == before[0]
    assert sid.save_state() == before[1]


def test_validates():
    sid = SID()
    with pytest.raises(ValueError):
        sid.envelope_phases(3, [], 10)
    with pytest.raises(ValueError):
        sid.envelope_phases(0, [(5, 4, 1)], 4)
    sid.clock(10)
    with pytest.raises(ValueError):
        sid.envelope_phases(0, [], 5)
    phases = sid.envelope_phases(0, [], 20)
    with pytest.raises(IndexError):
        phases.trajectory(phases.count)
    with pytest.raises(IndexError):
        phases.trajectory(-1)


def test_phase_states_follow_the_rate_cycle():
    sid = SID()
    sid.run([(0, 5, 0x00), (1, 6, 0x00), (2, 4, 0x40)], 2 * FRAME)
    phases = sid.envelope_phases(0, [], sid.cycle + 10)
    assert isinstance(phases, Phases)
    assert phases.states.tolist() == [
        0x7FFF,
        0x3FFF,
        0x1FFF,
        0xFFF,
        0x7FF,
        0x3FF,
        0x1FF,
        0xFF,
        0x7F,
    ]


@pytest.mark.parametrize(
    "ad,prompt,stalls,odd",
    [(0x0A, [12, 11, 10, 9, 8, 7], [32773, 32772], [13]), (0x50, None, None, None)],
)
def test_hard_restart_pieces(ad, prompt, stalls, odd):
    sid = SID()
    sid.run([(0, 5, 0x00), (1, 6, 0x00), (2, 4, 0x40)], 2 * FRAME)
    g = sid.cycle
    writes = [(g, 5, ad), (g + 1, 6, 0x88), (g + 2, 4, 0x01)]
    phases = sid.envelope_phases(0, writes, g + 3 * FRAME)
    onsets = []
    for k in range(phases.count):
        t = phases.trajectory(k)
        onsets.append(
            int(t.cycle[(t.state == 0) & (t.envelope > t.envelope[0])][0] - g)
        )
    if prompt is None:
        assert onsets == [223, 222, 221, 220, 219, 218, 6, 216, 224]
        return
    assert onsets == prompt + stalls + odd
    assert phases.pieces[:, :2].tolist() == [[0, 6], [6, 8], [8, 9]]
    first = phases.events[phases.pieces[0, 2]]
    assert first[1] == 0
    assert set(phases.events[:, 1].tolist()) <= {0, -1}


def test_much_faster_than_per_phase_traces():
    sid = SID()
    sid.run([(0, 5, 0xEE), (1, 6, 0xEE), (2, 4, 0x01)], 3000)
    g = sid.cycle
    writes = [
        (g + 5, 4, 0x00),
        (g + 9000, 4, 0x01),
        (g + 40000, 4, 0x00),
    ]
    until = g + 3 * FRAME
    fast = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        phases = sid.envelope_phases(0, writes, until)
        fast = min(fast, time.perf_counter() - t0)
    assert phases.count == 19532
    sample = range(0, phases.count, phases.count // 200)
    t0 = time.perf_counter()
    for k in sample:
        sid.envelope_trace(0, writes, until, int(phases.states[k]))
    brute = (time.perf_counter() - t0) / len(sample) * phases.count
    assert brute > 100 * fast
    for k in sample:
        want = sid.envelope_trace(0, writes, until, int(phases.states[k]))
        assert np.array_equal(phases.trajectory(k).cycle, want.cycle)


def _settled(ad, sr, control, gate=0):
    sid = SID()
    sid.run([(0, 5, ad), (1, 6, sr), (2, 4, control | gate)], 2 * FRAME)
    return sid


def _slow_release(release, attack):
    sid = _settled(0x00, 0xF0 | release, 0x40)
    g = sid.cycle
    writes = [(g + 5, 5, attack << 4), (g + 6, 6, 0x00), (g + 2 * FRAME, 4, 0x41)]
    return sid, writes, g + 3 * FRAME


def _hold():
    sid = _settled(0x08, 0xF0, 0x40, 1)
    g = sid.cycle
    writes = [
        (g + 5, 5, 0x00),
        (g + 2 * FRAME + 100, 4, 0x40),
        (g + 2 * FRAME + 7000, 4, 0x41),
    ]
    return sid, writes, g + 3 * FRAME


def _disabled(release):
    sid = _settled(0x00, 0xF0, 0x40)
    sid.run([(sid.cycle + 1, 6, 0xF0 | release)], sid.cycle + 50)
    g = sid.cycle
    writes = [(g + 2 * FRAME, 4, 0x41), (g + 2 * FRAME + 9000, 4, 0x40)]
    return sid, writes, g + 3 * FRAME


def _brute_ratio(sid, writes, until):
    fast = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        phases = sid.envelope_phases(0, writes, until)
        fast = min(fast, time.perf_counter() - t0)
    sample = range(0, phases.count, max(1, phases.count // 40))
    t0 = time.perf_counter()
    for k in sample:
        sid.envelope_trace(0, writes, until, int(phases.states[k]))
    brute = (time.perf_counter() - t0) / len(sample) * phases.count
    return brute / fast


@pytest.mark.parametrize("release,attack", [(8, 0), (8, 1), (9, 0), (9, 1)])
def test_slow_release_to_fast_attack_is_exact_and_faster(release, attack):
    sid, writes, until = _slow_release(release, attack)
    phases = _check(sid, 0, writes, until)
    assert phases.count == {8: 392, 9: 977}[release]
    assert len(phases.events) < 25 * len(phases.pieces)
    assert _brute_ratio(sid, writes, until) > 10


def test_hard_restart_is_exact_and_faster():
    sid = _settled(0x00, 0x00, 0x40)
    g = sid.cycle
    writes = [(g + 5, 5, 0x00), (g + 6, 6, 0x00), (g + 2 * FRAME, 4, 0x01)]
    _check(sid, 0, writes, g + 3 * FRAME)
    assert _brute_ratio(sid, writes, g + 3 * FRAME) > 5


def test_sustain_hold_then_release_is_exact():
    sid, writes, until = _hold()
    assert _check(sid, 0, writes, until).count == 392
    assert _brute_ratio(sid, writes, until) > 10


def test_parked_sustain_has_no_events_until_the_next_write():
    sid = _settled(0x08, 0xF0, 0x40, 1)
    g = sid.cycle
    writes = [(g + 5, 5, 0x00)]
    phases = _check(sid, 0, writes, g + 3 * FRAME)
    assert len(phases.events) == 1 and phases.events[0, 6] == 1
    assert _brute_ratio(sid, writes, g + 3 * FRAME) > 50


@pytest.mark.parametrize("release", [8, 9])
def test_disabled_counter_at_zero_is_exact(release):
    sid, writes, until = _disabled(release)
    assert sid.voices()["counter_enabled"][0] == 0
    _check(sid, 0, writes, until)


@pytest.mark.parametrize("seed", range(300, 306))
def test_matches_envelope_trace_across_long_quiet_windows(seed):
    rng = np.random.default_rng(seed)
    sid = _settled(int(rng.integers(0, 256)), int(rng.integers(0, 256)), 0x40, seed & 1)
    g = sid.cycle
    gate = 0x40 | int(rng.integers(0, 2))
    writes = [
        (g + 5, 5, int(rng.integers(0, 256))),
        (g + 6, 6, int(rng.integers(0, 256))),
        (g + 2 * FRAME + int(rng.integers(0, 3000)), 4, gate),
    ]
    _check(sid, 0, writes, g + 3 * FRAME)


def test_events_are_ramps_that_expand_to_the_trace():
    sid, writes, until = _slow_release(8, 0)
    phases = sid.envelope_phases(0, writes, until)
    assert phases.events.shape[1] == 7
    assert phases.events[:, 6].max() > 100
    assert (phases.events[:, 6] >= 1).all()
    assert set(phases.events[phases.events[:, 6] > 1, 4].tolist()) <= {-1, 1}


@pytest.mark.parametrize("case", ["slow", "hold", "plain"])
def test_expand_matches_trajectories(case):
    if case == "slow":
        sid, writes, until = _slow_release(8, 1)
    elif case == "hold":
        sid, writes, until = _hold()
    else:
        sid = _settled(0x0A, 0x88, 0x40)
        writes, until = [(sid.cycle + 4, 4, 0x01)], sid.cycle + FRAME
    phases = sid.envelope_phases(0, writes, until)
    table = {}
    ids, times = phases.expand(table)
    assert ids.shape == (phases.count,) and times.dtype == np.int64
    assert len(table) == len(set(ids.tolist())) == ids.max() + 1
    seen = {}
    for k in range(phases.count):
        t = phases.trajectory(k)
        assert times[k, : len(t)].tolist() == t.cycle.tolist()
        assert (times[k, len(t) :] == -1).all()
        key = (t.envelope.tobytes(), t.state.tobytes())
        assert seen.setdefault(int(ids[k]), key) == key
    assert len(set(seen.values())) == len(seen)
    again, _ = phases.expand(table)
    assert np.array_equal(again, ids)
    assert len(table) == ids.max() + 1


def test_expand_numbers_sequences_consistently_across_calls():
    sid, writes, until = _slow_release(8, 0)
    table = {}
    first, _ = sid.envelope_phases(0, writes, until).expand(table)
    second, _ = sid.envelope_phases(0, writes, until).expand(table)
    assert np.array_equal(first, second)
    assert np.array_equal(first, sid.envelope_phases(0, writes, until).expand()[0])
