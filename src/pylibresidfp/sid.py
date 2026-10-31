from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import Chip  # pylint: disable=no-name-in-module,import-error

_MODELS = {"6581": 0, "8580": 1}
_METHODS = {"decimate": 0, "resample": 1}
_MAX_RATIO = 16384.0


@dataclass(frozen=True)
class Phases:
    """Every rate-counter phase of :meth:`SID.envelope_phases`, compressed.

    ``states[k]`` is the LFSR state of phase ``k`` (``count`` phases). ``pieces`` rows are
    ``(k0, k1, first, events)``: phases ``k0 <= k < k1`` share the ``events`` rows of ``events``
    from ``first``, whose columns are ``(base, slope, envelope, state)``; event ``i`` of phase ``k``
    falls at cycle ``base + slope * (k - k0)``."""

    pieces: NDArray[np.int64]
    events: NDArray[np.int64]
    count: int
    states: NDArray[np.uint16]

    def trajectory(self, k: int) -> np.recarray:
        """Phase ``k``'s ``(cycle, envelope, state)``, as ``envelope_trace`` gives it."""
        if not 0 <= k < self.count:
            raise IndexError(f"phase {k} out of range for {self.count} phases")
        k0, _, first, events = self.pieces[
            np.searchsorted(self.pieces[:, 0], k, side="right") - 1
        ]
        base, slope, level, state = self.events[first : first + events].T
        return np.rec.fromarrays(
            (
                base + slope * (k - k0),
                level.astype(np.uint8),
                state.astype(np.uint8),
            ),
            names=("cycle", "envelope", "state"),
        )


class SID:
    """MOS 6581 / CSG 8580 emulation backed by libresidfp."""

    def __init__(
        self,
        model: str = "6581",
        clock_hz: float = 985248.0,
        sampling_hz: float = 44100.0,
        method: str = "decimate",
        filter: bool = True,  # pylint: disable=redefined-builtin
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
        self._chip = Chip(
            _MODELS[model],
            self.clock_hz,
            self.sampling_hz,
            _METHODS[method],
            self.filter,
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

    @staticmethod
    def _writes(writes: ArrayLike) -> NDArray[np.int64]:
        arr = np.asarray(writes)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        if arr.dtype.kind not in "iu":
            raise ValueError("writes must be an integer array")
        return arr.astype(np.int64, copy=False)

    def run(self, writes: ArrayLike, until: int) -> NDArray[np.int16]:
        return self._chip.run(self._writes(writes), int(until))

    def serve(self, writes: ArrayLike, cycle: int, reg: int) -> int:
        return self._chip.serve(self._writes(writes), int(cycle), int(reg))

    def voices(self) -> dict[str, NDArray[np.int64]]:
        """Each voice's envelope and oscillator state, one ``(3,)`` array per field."""
        return self._chip.voices()

    def trace(self, writes: ArrayLike, until: int) -> np.recarray:
        """:meth:`run` without audio, clocking the oscillators and envelopes only: each voice's
        envelope ``(cycle, voice, envelope, state)`` at the start and whenever it changes.
        """
        return np.rec.fromarrays(
            self._chip.trace(self._writes(writes), int(until)),
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
                self._writes(writes),
                int(until),
                -1 if lfsr is None else int(lfsr),
            ),
            names=("cycle", "envelope", "state"),
        )

    def envelope_phases(self, voice: int, writes: ArrayLike, until: int) -> Phases:
        """:meth:`envelope_trace` for every phase of ``voice``'s rate counter at once: phase ``k``
        starts the LFSR at the ``k``-th state, from 0x7FFF, of the cycle that ends at the voice's
        current rate. Exact for every phase, at a cost that follows the events and the distinct
        behaviours rather than the number of phases or cycles."""
        return Phases(
            *self._chip.envelope_phases(int(voice), self._writes(writes), int(until))
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
