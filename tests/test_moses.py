import numpy as np
import pytest

from pylibresidfp import SID

vm = pytest.importorskip("vagrant_moses.machine")

ORG = 0xC000
PROGRAM = bytes.fromhex(
    "a90f8d18d4"  # LDA #$0F ; STA $D418
    "a9108d00d4"  # LDA #$10 ; STA $D400
    "a9208d01d4"  # LDA #$20 ; STA $D401
    "a9098d05d4"  # LDA #$09 ; STA $D405
    "a9f08d06d4"  # LDA #$F0 ; STA $D406
    "a9218d04d4"  # LDA #$21 ; STA $D404
    "ad1bd4"  # LDA $D41B
    "4c1dc0"  # JMP *
)


def machine_with(on_read):
    sid = vm.NativeSid("6581", on_read)
    machine = vm.Machine(sids={0xD400: sid})
    machine.ram[ORG : ORG + len(PROGRAM)] = PROGRAM
    return machine, sid


def test_drained_writes_render_directly():
    machine, sid = machine_with(lambda cycle, reg: 0)
    end = machine.reset(ORG) + 400
    machine.run(end)
    writes = sid.drain_writes()
    assert len(writes) == 6
    chip = SID()
    samples = chip.run(writes, end)
    assert chip.cycle == end and len(samples) > 0 and np.abs(samples).max() > 0
    assert np.array_equal(samples, SID().run(np.array(writes), end))


def test_serve_is_the_on_read_body():
    chip = SID()
    log = []

    def on_read(cycle, reg):
        value = chip.serve(machine_sid.drain_writes(), cycle, reg)
        log.append((cycle, reg, value))
        return value

    machine, machine_sid = machine_with(on_read)
    machine.run(machine.reset(ORG) + 400)
    assert log and log[0][1] == 0x1B
    assert chip.cycle == log[0][0]
