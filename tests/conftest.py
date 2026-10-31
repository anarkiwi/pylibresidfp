# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from pylibresidfp import SID

TONE = np.array(
    [
        (0, 24, 15),
        (5, 0, 0x10),
        (10, 1, 0x20),
        (15, 5, 0x09),
        (20, 6, 0xF0),
        (25, 4, 0x21),
        (4000, 21, 0x40),
        (4010, 23, 0xF1),
        (9000, 4, 0x20),
    ],
    dtype=np.int64,
)
CASES = [(m, me) for m in ("6581", "8580") for me in ("decimate", "resample")]


@pytest.fixture(params=CASES, ids="-".join)
def case(request):
    return request.param


@pytest.fixture
def sid(case):
    return SID(case[0], method=case[1])
