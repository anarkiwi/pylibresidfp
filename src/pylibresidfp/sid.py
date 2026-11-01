from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import (  # pylint: disable=no-name-in-module,import-error
    Chip,
    envelope_run as _envelope_run,
    envelope_runs as _envelope_runs,
    envelope_grid as _envelope_grid,
)

_MODELS = {"6581": 0, "8580": 1}
_METHODS = {"decimate": 0, "resample": 1}
_MAX_RATIO = 16384.0


def _as_writes(writes: ArrayLike) -> NDArray[np.int64]:
    arr = np.asarray(writes)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if arr.dtype.kind not in "iu":
        raise ValueError("writes must be an integer array")
    return arr.astype(np.int64, copy=False)


@dataclass(frozen=True)
class Phases:
    """Every rate-counter phase of :meth:`SID.envelope_phases`, compressed.

    ``states[k]`` is the LFSR state of phase ``k`` (``count`` phases). ``pieces`` rows are
    ``(k0, k1, first, runs)``: phases ``k0 <= k < k1`` share the ``runs`` rows of ``events`` from
    ``first``, whose columns are ``(base, slope, envelope, state, step, spacing, count)``. A row
    is a ramp of ``count`` consecutive envelope changes: change ``j`` of phase ``k`` falls at
    cycle ``base + slope * (k - k0) + spacing * j`` and sets envelope ``envelope + step * j`` in
    ``state``."""

    pieces: NDArray[np.int64]
    events: NDArray[np.int64]
    count: int
    states: NDArray[np.uint16]

    def _piece(self, row: NDArray[np.int64]) -> tuple[NDArray[np.int64], ...]:
        """The piece's per-change ``(base, slope, envelope, state)`` with ramps expanded."""
        _, _, first, runs = row
        base, slope, level, state, step, spacing, count = self.events[
            first : first + runs
        ].T
        j = np.arange(count.sum()) - np.repeat(count.cumsum() - count, count)
        return (
            np.repeat(base, count) + np.repeat(spacing, count) * j,
            np.repeat(slope, count),
            np.repeat(level, count) + np.repeat(step, count) * j,
            np.repeat(state, count),
        )

    def trajectory(self, k: int) -> np.recarray:
        """Phase ``k``'s ``(cycle, envelope, state)``, as ``envelope_trace`` gives it."""
        if not 0 <= k < self.count:
            raise IndexError(f"phase {k} out of range for {self.count} phases")
        row = self.pieces[np.searchsorted(self.pieces[:, 0], k, side="right") - 1]
        base, slope, level, state = self._piece(row)
        return np.rec.fromarrays(
            (
                base + slope * (k - row[0]),
                level.astype(np.uint8),
                state.astype(np.uint8),
            ),
            names=("cycle", "envelope", "state"),
        )

    def expand(
        self, table: dict[bytes, int] | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(ids, times)`` over all phases: ``ids[k]`` numbers phase ``k``'s sequence of
        ``(envelope, state)`` changes in ``table`` (a new one if not given; pass the same one to
        number sequences consistently across calls), and ``times[k]`` its change cycles, padded
        with -1 to the longest sequence."""
        table = {} if table is None else table
        pieces = [self._piece(row) for row in self.pieces]
        ids = np.empty(self.count, np.int64)
        times = np.full((self.count, max(len(p[0]) for p in pieces)), -1, np.int64)
        for (k0, k1, _, _), (base, slope, level, state) in zip(
            self.pieces.tolist(), pieces
        ):
            key = np.stack((level, state)).astype(np.uint8).tobytes()
            ids[k0:k1] = table.setdefault(key, len(table))
            times[k0:k1, : len(base)] = base + slope * np.arange(k1 - k0)[:, None]
        return ids, times


class SID:
    """MOS 6581 / CSG 8580 emulation backed by libresidfp.

    ``pipeline=True`` renders audio on three threads, one each for the voices,
    the filter and the external filter with the resampler, handing blocks of
    cycles from stage to stage. Each stage clocks its part of the chip in cycle
    order, so output and state are identical to the single-threaded path."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model: str = "6581",
        clock_hz: float = 985248.0,
        sampling_hz: float = 44100.0,
        method: str = "decimate",
        filter: bool = True,  # pylint: disable=redefined-builtin
        pipeline: bool = False,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"model must be one of {sorted(_MODELS)}: {model!r}")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}: {method!r}")
        if not (0.0 < clock_hz < math.inf and 0.0 < sampling_hz < math.inf):
            raise ValueError("clock_hz and sampling_hz must be positive and finite")
        if method == "resample" and 125.0 * clock_hz / sampling_hz >= _MAX_RATIO:
            raise ValueError("clock_hz / sampling_hz is too large for resampling")
        self.model = model
        self.clock_hz = float(clock_hz)
        self.sampling_hz = float(sampling_hz)
        self.method = method
        self.filter = bool(filter)
        self.pipeline = bool(pipeline)
        self._chip = Chip(
            _MODELS[model],
            self.clock_hz,
            self.sampling_hz,
            _METHODS[method],
            self.filter,
            self.pipeline,
        )

    def write(self, reg: int, value: int) -> None:
        self._chip.write(int(reg), int(value))

    def read(self, reg: int) -> int:
        return self._chip.read(int(reg))

    def peek(self, reg: int) -> int:
        return self._chip.peek(int(reg))

    def reset(self) -> None:
        self._chip.reset()

    def clock(self, cycles: int) -> NDArray[np.int16]:
        return self._chip.clock(int(cycles))

    @property
    def cycle(self) -> int:
        return self._chip.cycle

    def run(self, writes: ArrayLike, until: int) -> NDArray[np.int16]:
        return self._chip.run(_as_writes(writes), int(until))

    def serve(self, writes: ArrayLike, cycle: int, reg: int) -> int:
        return self._chip.serve(_as_writes(writes), int(cycle), int(reg))

    def voices(self) -> dict[str, NDArray[np.int64]]:
        """Each voice's envelope and oscillator state, one ``(3,)`` array per field."""
        return self._chip.voices()

    def trace(self, writes: ArrayLike, until: int) -> np.recarray:
        """:meth:`run` without audio, clocking the oscillators and envelopes only: each voice's
        envelope ``(cycle, voice, envelope, state)`` at the start and whenever it changes.
        """
        return np.rec.fromarrays(
            self._chip.trace(_as_writes(writes), int(until)),
            names=("cycle", "voice", "envelope", "state"),
        )

    def envelope_trace(
        self, voice: int, writes: ArrayLike, until: int, lfsr: int | None = None
    ) -> np.recarray:
        """``voice``'s envelope ``(cycle, envelope, state)`` at the start and at each change as
        :meth:`trace` gives it, stepping a copy of that envelope generator alone (its rate
        counter's LFSR set to ``lfsr`` if given): the chip is left as it is, and writes to other
        registers are ignored."""
        return np.rec.fromarrays(
            self._chip.envelope_trace(
                int(voice),
                _as_writes(writes),
                int(until),
                -1 if lfsr is None else int(lfsr),
            ),
            names=("cycle", "envelope", "state"),
        )

    def envelope_state(self, voice: int) -> bytes:
        """``voice``'s envelope generator, serialised field by field: equal generator states give
        equal bytes. The input of :func:`envelope_run`, at :attr:`cycle`."""
        return self._chip.envelope_state(int(voice))

    def envelope_phases(self, voice: int, writes: ArrayLike, until: int) -> Phases:
        """:meth:`envelope_trace` for every phase of ``voice``'s rate counter at once: phase ``k``
        starts the LFSR at the ``k``-th state, from 0x7FFF, of the cycle that ends at the voice's
        current rate. Exact for every phase, at a cost that follows the events and the distinct
        behaviours rather than the number of phases or cycles."""
        return Phases(
            *self._chip.envelope_phases(int(voice), _as_writes(writes), int(until))
        )

    def save_state(self) -> bytes:
        return self._chip.save_state()

    def restore_state(self, state: bytes) -> None:
        self._chip.restore_state(bytes(state))

    def params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "clock_hz": self.clock_hz,
            "sampling_hz": self.sampling_hz,
            "method": self.method,
            "filter": self.filter,
            "pipeline": self.pipeline,
        }

    def copy(self) -> SID:
        other = SID(**self.params())
        other.restore_state(self.save_state())
        return other

    def __getstate__(self) -> dict[str, Any]:
        return {**self.params(), "state": self.save_state()}

    def __setstate__(self, data: dict[str, Any]) -> None:
        data = dict(data)
        state = data.pop("state")
        self.__init__(**data)  # type: ignore[misc]
        self.restore_state(state)


def _envelope_result(raw: tuple[Any, ...]) -> tuple[np.recarray, bytes]:
    return (
        np.rec.fromarrays(raw[:3], names=("cycle", "envelope", "state")),
        raw[3],
    )


def envelope_run(
    state: bytes, writes: ArrayLike, until: int, start: int
) -> tuple[np.recarray, bytes]:
    """Step one envelope generator from ``state`` (:meth:`SID.envelope_state` at cycle ``start``)
    to ``until`` under ``writes`` ``(cycle, reg, value)``, registers 4, 5 and 6 meaning control,
    attack/decay and sustain/release (others are ignored; a write at ``until`` is left for the
    next run). Returns what :meth:`SID.envelope_trace` gives and the end state, without a chip,
    at a cost that follows the envelope's events rather than the cycles."""
    return _envelope_result(
        _envelope_run(bytes(state), _as_writes(writes), int(until), int(start))
    )


def envelope_runs(
    state: bytes, writes: Sequence[ArrayLike], until: int, start: int
) -> list[tuple[np.recarray, bytes]]:
    """:func:`envelope_run` of each of ``writes`` from the same ``state``, in one call."""
    return [
        _envelope_result(raw)
        for raw in _envelope_runs(
            bytes(state), [_as_writes(w) for w in writes], int(until), int(start)
        )
    ]


ENVELOPE_STATE = np.dtype(
    [
        ("lfsr", "<u2"),
        ("rate", "<u2"),
        ("exponential_counter", "<u4"),
        ("exponential_counter_period", "<u4"),
        ("new_exponential_counter_period", "<u4"),
        ("state_pipeline", "<u4"),
        ("envelope_pipeline", "<u4"),
        ("exponential_pipeline", "<u4"),
        ("state", "<i4"),
        ("next_state", "<i4"),
        ("counter_enabled", "u1"),
        ("gate", "u1"),
        ("reset_lfsr", "u1"),
        ("envelope_counter", "u1"),
        ("attack", "u1"),
        ("decay", "u1"),
        ("sustain", "u1"),
        ("release", "u1"),
        ("env3", "u1"),
    ]
)
"""The fields of :meth:`SID.envelope_state`'s bytes, for reading or building states."""


def envelope_grid(
    states: ArrayLike, programs: ArrayLike, cycles: int, levels: bool = False
) -> NDArray[np.uint64]:
    """Each of ``programs`` run from each of ``states`` for ``cycles`` cycles, as a 128-bit digest
    of the :func:`envelope_run` trace and end state: equal digests mean the same envelope and the
    same future. ``states`` is an :data:`ENVELOPE_STATE` array (or rows of state bytes) and
    ``programs`` a ``(P, K, 3)`` array of writes ``(cycle from 0, reg, value)``, padded with
    writes to other registers. With ``levels`` the digest covers only the envelope's level changes
    and the end state, not when the generator changed state. Returns ``(len(states), P, 2)``; the
    GIL is released."""
    raw = np.ascontiguousarray(states)
    if raw.dtype == ENVELOPE_STATE:
        raw = raw.view(np.uint8).reshape(-1, ENVELOPE_STATE.itemsize)
    progs = np.ascontiguousarray(programs, dtype=np.int64)
    return _envelope_grid(
        raw.astype(np.uint8, copy=False), progs, int(cycles), bool(levels)
    )
