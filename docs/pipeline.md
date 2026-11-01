# Pipelined rendering

`SID(..., pipeline=True)` renders `clock`, `run` and `serve` on three threads:

| stage | thread | owns |
|---|---|---|
| voices | caller | waveform and envelope generators, voice sync, bus value |
| filter | worker | the 6581 and 8580 filters |
| output | worker | external filter, resampler |

Stage 1 clocks the oscillators and envelopes one cycle at a time and stores each voice's analog
output (`Voice::output()`) and envelope level, in voice order, into a block of cycles. Writes to
registers 0x15-0x18 update the bus value and voice sync on stage 1 and are queued with their offset
in the block. Stage 2 applies each queued write before the cycle it precedes and clocks the filter
from the stored values through `Filter::clock(wav1, wav2, wav3, env1, env2, env3)`, added by
`patches/0005-filter-clock-values.patch`; the original `Filter::clock(Voice&, ...)` reads the same
values in the same order and calls it. Stage 3 runs the external filter and the resampler over the
filter output. Blocks pass between stages through a ring of four, signalled by atomic counters;
every call joins both workers before returning, so the chip state seen by `save_state`, `read` or a
later serial call is the same as after the single-threaded loop.

Every component sees the same inputs in the same order as in `SID::clock`, so samples and state
are identical with and without the pipeline (`tests/test_pipeline.py` also pins the serial audio
to digests from the 0.5.0 release).

## Throughput

`python scripts/bench.py` on a Xeon E5-2697 v4 (2.3 GHz, shared, load average ~7), 48 kHz,
three gated voices, 10 s in 20 `run` calls:

| model | method | serial | pipeline |
|---|---|---|---|
| 6581 | resample | 9.2x | 16.2x |
| 6581 | decimate | 11.4x | 16.2x |
| 8580 | resample | 12.5x | 27.0x |
| 8580 | decimate | 15.7x | 27.5x |

Wall time equals the slowest stage. On the 8580 the stages are close to balanced. On the 6581 the
filter stage takes about twice as long as the others: two `Integrator6581::solve` calls per cycle,
each depending on the last. Moving the per-voice normalisation into stage 1 shortens the filter
stage little and makes stage 1 the slowest on the 8580, so the split stays at the voice outputs.

The pipeline uses three cores for one chip. To render many independent streams, run one serial
`SID` per core instead.
