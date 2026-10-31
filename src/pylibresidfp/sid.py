from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import Chip  # pylint: disable=no-name-in-module,import-error

_MODELS = {"6581": 0, "8580": 1}
_METHODS = {"decimate": 0, "resample": 1}
_MAX_RATIO = 16384.0


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
