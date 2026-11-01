# pylibresidfp

pybind11 binding of [libresidfp](https://github.com/libsidplayfp/libresidfp) (the reSIDfp SID
emulator) with chip state save/restore, a deterministic output and a GIL-free render loop.

```python
from pylibresidfp import SID

sid = SID(model="6581", clock_hz=985248.0, sampling_hz=44100.0, method="decimate", filter=True,
          pipeline=False)  # pipeline=True: voices, filter and output on three threads, same samples
sid.write(24, 15); sid.read(27); sid.peek(27); sid.reset()
sid.cycle                      # absolute cycle count, starts at 0
sid.clock(cycles)              # np.int16 samples for those cycles
sid.run(writes, until)         # writes: (N, 3) ints or [(cycle, reg, value), ...], absolute cycles
sid.serve(writes, cycle, reg)  # run to `cycle`, then read(reg): the body of a NativeSid on_read
sid.voices()                   # per-voice envelope (LFSR, rate, counter, state, ...) and oscillator state
sid.trace(writes, until)       # run without audio: (cycle, voice, envelope, state) at each envelope change
sid.envelope_trace(v, writes, until, lfsr=None)  # one voice's envelope alone from a copy (its
                               # rate counter's LFSR set to lfsr if given): the chip is untouched
sid.envelope_phases(v, writes, until)  # envelope_trace for every rate-counter phase at once: Phases
                               # (pieces, events, count, states) with .trajectory(k), exact for each k;
                               # events are ramps of envelope steps; .expand(table) -> (ids, times)
state = sid.envelope_state(v)  # that voice's envelope generator as bytes: equal states, equal bytes
trace, end = envelope_run(state, writes, until, start)   # from pylibresidfp import envelope_run:
                               # envelope_trace from `state` (taken at cycle start) with no chip,
                               # event-driven; regs 4/5/6 are control/AD/SR, others ignored
runs = envelope_runs(state, [writes_a, writes_b], until, start)  # several candidates, one call
grid = envelope_grid(states, programs, cycles)  # (N, P, 2) digests of every program from every state
                               # (an ENVELOPE_STATE array): equal digests, same envelope and future
state = sid.save_state(); sid.restore_state(state)
twin = sid.copy()              # same parameters and state; SID is also picklable
```

`run` clocks to each write's cycle, writes it, then clocks to `until`. Write cycles must not
precede `sid.cycle` or each other and must not exceed `until`; violations raise `ValueError`.
`serve` discards the samples it produces (use `run` to keep audio).

`trace` clocks the oscillators and envelopes only (libresidfp's `clockDigital`), so a chip used for
tracing no longer renders the same audio as one that was not. `voices` and `trace` read libresidfp's
internals through `patches/0002-probe-access.patch`, which befriends a `reSIDfp::Probe` class.
`patches/0003-zero-state-padding.patch` zeroes the saved state's struct padding, so `save_state` is
reproducible byte for byte; `patches/0004-initialise-params.patch` gives the filter parameters
libresidfp keeps for it their defaults rather than uninitialised memory. `patches/0006-state-fc-width.patch`
widens the saved filter cutoff to its full 11 bits, so a restored chip keeps the cutoff's high bits
through later writes to register 0x15.

`clock`, `run`, `serve`, `trace`, `envelope_trace`, `envelope_phases`, `envelope_run`, `envelope_runs`, `envelope_grid`, `save_state` and `restore_state` release the GIL. Each instance has a busy
flag: use from a second thread while a call is in progress raises `RuntimeError`. Distinct
instances run in parallel.

`pipeline=True` splits `clock`, `run` and `serve` across three threads with identical samples
and state; see [docs/pipeline.md](docs/pipeline.md).

Fed directly from `vagrant_moses`: `sid.run(native_sid.drain_writes(), until)`.

## Determinism

Two instances with the same parameters and writes produce identical samples, for both models and
both methods. Upstream libresidfp dithers the filter's float-to-int conversion from a table indexed by
a counter held in the process-wide filter-model singleton; `patches/0001-deterministic-rounding.patch`
(applied at build time) replaces the dither with plain rounding. Other process-wide state is
immutable after first use (filter and waveform tables). The static setters
`Filter6581::setFilterRange`/`setCaps` are not exposed. The pinned libresidfp has no `noiseLevel`, so there is no
`noise_level` option.

State is not portable across builds. `sid.cycle` is part of the saved state.

## Build

`git clone --recurse-submodules`, then `pip install .`. Needs CMake >= 3.20, `patch` and a C++20
compiler. libresidfp is the submodule `extern/libresidfp`, pinned to a release tag; its version is
read from the submodule's `configure.ac`, and `.github/workflows/libresidfp.yml` opens a PR when a
newer release appears. The build copies its sources and applies `patches/` to the copy. Tests: `pip install -e .[dev]`, then
`pytest -n auto --cov=pylibresidfp`. `docker build .` (build arg `PYTHON_VERSION`) runs the full
lint and test suite.

GPL-2.0-or-later (see COPYING).
