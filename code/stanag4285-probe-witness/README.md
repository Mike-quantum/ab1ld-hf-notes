# Probe-witness test for STANAG 4285: code, symbol template and IQ excerpt

**Mike Burns, AB1LD (with Claude, Anthropic AI), 2026-10-07.** Written for Nils Schiffhauer, DK8OK, who asked for it
to add Section 5 of our [STANAG 4285 report](../../reports/2026-10-dk8ok-stanag4285/AB1LD_DK8OK_STANAG4285_2026-10.pdf)
("Are the bumps real? Sync and probes as independent witnesses") to his app; shared here for anyone who wants to use it.
It contains the Python validation code, the exact scrambled reference-symbol template, an IQ excerpt, and how the
unknown neighbouring symbols were handled.

## Files
| File | What it is |
|---|---|
| `s4285_probe_witness.py` | The validation code, standalone: reads a plain IQ WAV, fits a straight-line frame clock (DK8OK's method), estimates the received response by least squares from the sync and, separately, from the probes, and scores held-out prediction. Needs only numpy, scipy, matplotlib and the next file. |
| `s4285_known_symbols.py`, `s4285_known_symbols.csv` | The 256-symbol frame: 80 sync + 48 probe symbols with their exact (scrambled) values; data symbols marked unknown. |
| `test_s4285_probe_witness.py` | Known-answer tests: synthetic 4285 signals with paths we chose, run through the script (see Tests below). |
| [`../stanag4285-dk8ok-port/`](../stanag4285-dk8ok-port/) | DK8OK's measurement core in Python, which reads the same excerpt. |
| `results/` | Our result on the excerpt (`witness_FUF8552_105500Z.png`, `_summary.txt`, `_profiles.csv`), to check yours against. |
| [`../../data/2026-10-01_FUF_8552.6kHz/`](../../data/2026-10-01_FUF_8552.6kHz/) | The IQ excerpt: 10 minutes of FUF, 8552.6 kHz USB, 2026-10-01 **10:55:00-11:05:00 UTC**, 2-channel int16 IQ at exactly 12000 S/s. IQ centre 8554.4 kHz, so the 4285 signal centre (carrier + 1800 Hz) is at **0 Hz**. 28.8 MB. A 2-minute piece in the original KiwiSDR format is there too; the tools that made both are in [`../kiwi-tools/`](../kiwi-tools/). |

Run, from this folder: `python3 s4285_probe_witness.py ../../data/2026-10-01_FUF_8552.6kHz/FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav 0 out` (about 25 s).
Tests: `python3 test_s4285_probe_witness.py` (about 20 s; writes test files into the current folder).

## The excerpt
- KiwiSDR (GPS), loop-on-ground antenna, Arlington MA, about 3200 km from Fort-de-France. The interval spans the
  appearance of the +2.3 ms component (absent before about 11:00 UTC, strong afterwards).
- The Kiwi's true rate is about 11998.96 S/s, not 12000. The excerpt was resampled (64-tap windowed sinc) onto an exact
  12000-S/s grid: **sample n is at 10:55:00 UTC + n/12000 s** (GPS time from a straight line fitted to the Kiwi's
  per-block stamps over the 10 minutes; the stamps lie within 1 µs of that line). So `audioread` gives correct timing directly.
- Level as recorded (peak about −6 dBFS); nothing else done to it.

## The reference symbols (exact, scrambled)
- **Sync, symbols 0-79:** the 31-bit PN, `0101100111110001101110101000010` repeated (31 + 31 + 18), bit 0 → +1, bit 1 → −1.
  This is DK8OK's PN31 exactly.
- **Probes, symbols 112-127, 160-175, 208-223:** STANAG 4285 scrambles the 176 symbols after the sync with a 9-stage
  shift register, x⁹ + x⁴ + 1, loaded with all ones at the start of every frame's data part and stepped three times per
  symbol; the three lowest register bits give an 8PSK index k (0-7) added mod 8. Probe symbols are index 0 before
  scrambling, so on air they are the scrambler output itself, value exp(jπk/4), and **identical in every frame**.
- Independent check: the 48 probe values we had measured from received frames last month agree with this scrambler
  output at all 48 positions. Our measured template differed only by one constant 180° rotation, the same overall
  sign we had already found against DK8OK's PN31. The CSV uses the standard's sign (sync +1 = index 0).

## Unknown neighbouring symbols: how they were handled
They were **excluded, not estimated**. The model per frame is

  r[k] = Σₘ h[m] · a[(k − m)/4],  taps m = −10 … +29 (−1.04 … +3.0 ms at 9600 S/s, 4 samples/symbol),

with r the received samples (frame clock from the straight-line fit, main-path Doppler removed within the frame), a
the transmitted symbols and h the whole received response (transmit pulse, receiver filter and propagation together;
no RRC or other pulse shape assumed).

- A received sample k is used **only if every symbol under the whole tap span is known**. If any of those symbols is
  a data symbol, the sample is dropped. No data symbol is decided, guessed or treated as noise.
- Each usable sample is assigned to one witness: **sync** if all symbols in its span are sync symbols, **probe** if all
  are probe symbols. Samples whose span mixes sync and probe are dropped, so the two witnesses share no samples.
- With this span, a frame gives **284 sync rows and 84 probe rows**. A 16-symbol probe block is short compared with the
  10-symbol span, so each block gives only 7 symbol positions × 4 samples = 28 rows. A shorter tap span gives more
  probe rows, at the cost of the late part of the response.
- Least squares then gives h from the sync rows alone and, separately, from the probe rows alone. Each h predicts the
  **other** set's rows (held out). Response models compared: narrow ±0.4 ms, core ±1.0 ms, core + late (adds +1.5 …
  +3.0 ms), full −1.04 … +3.0 ms. Score: normalised prediction error per frame, median over frames, and the paired
  change against "core".

## Result on the excerpt (5624 frames)
| fit sync → predict probe | median error | vs core | better than core in |
|---|---|---|---|
| narrow ±0.4 ms | −3.6 dB | +5.3 dB | 1 % of frames |
| core ±1.0 ms | −9.4 dB | 0 | — |
| core + late | −11.6 dB | −1.0 dB | 92 % |
| full | −12.4 dB | −2.0 dB | 98 % |

The reverse direction (fit probes, predict sync) points the same way but less strongly (core + late better than core
in 61 % of frames, by 0.4 dB). The probe fit rests on only 84 rows per frame.

**Read "core + late", not "full".** The synthetic tests below show that "full" beats "core" even for a single path,
because the transmit pulse (RRC 0.2) has tails beyond ±1 ms. Only "core + late", which adds +1.5 … +3.0 ms and
nothing else, tests for a late path: a single path gains 0.16 dB from it, while a real +2.3 ms path at −9 dB gains 9 dB.
On FUF it gains 1.0 dB in 92 % of frames.

- **Both witnesses give the same response.** The shoulder at −0.5 ms and the component at +2.3 ms are within 0.4 dB in
  the two independent estimates. They are in the received signal.
- **+1.3 … +1.5 ms** (about −17 dB in plain correlation): 3-4 dB weaker in the sync least-squares estimate, 1-3 dB
  weaker in the probe estimate. The probe estimate is the noisier one, and its mean power reads high at weak taps
  (see Tests). These ten minutes do not settle this bump. (Over the full 30 minutes in our report it was about 4 dB
  weaker in both, i.e. partly a correlation sidelobe.)
- **+0.6 ms:** −11 dB here, but −8 dB in our earlier run of the same method that used a different receive filter.
  That supports our earlier reading that it is mostly filter and pulse ripple, not a path.
- Cross-check: this standalone script reproduces our original pipeline on the same 10 minutes to within about 0.5 dB
  (original: core + late −0.96 dB / 92 %, full −2.48 dB / 98 %).

## Tests (known answer)
`test_s4285_probe_witness.py` builds synthetic 4285 signals and runs them through the script. Each signal has 2 minutes
of 256-symbol frames (sync, scrambled probes, random 8PSK data), an RRC 0.2 pulse, a transmitter clock 0.69 ppm fast,
paths delayed and Doppler-shifted as chosen, and 20 dB SNR. All checks pass:

| | put in | recovered |
|---|---|---|
| A: three paths (0 ms; −0.5 ms at −12 dB; +2.3 ms at −9 dB) | response = pulse at each path | sync estimate within 0.3 dB on every tap within 25 dB of the peak; probe estimate within 0.7 dB on every tap within 20 dB |
| B: as A, signal moved +1000 Hz in the IQ | same | same as A (tests the centre-offset argument) |
| C: main path only | pulse alone | within 0.2 dB (sync) / 0.4 dB (probe); nothing at +2.3 ms |
| main-path Doppler | −1.100 Hz | −1.0997 to −1.1000 Hz |
| frame period | −0.693 ppm | −0.728 ppm (a consistent 0.035 ppm bias in the simple peak clock; harmless here, since each frame's fit absorbs it) |
| core + late vs core | A/B: real late path; C: none | A/B −9.1 to −9.4 dB, 100 % of frames; C −0.16 dB (and +0.66 dB in reverse) |
| full vs core | — | C still −2.9 dB: the pulse tails, not a path |

Estimation-noise floor in the single-path case at 20 dB SNR: −44 dB below the peak from the sync, −35 dB from the
probes. Mean |h|² includes this noise, so weak taps read high, more so from the probes. On weaker signals both floors
rise, so judge only features well above them.

## Two cautions
- The frame clock here is the simple version: integer peak of the sync-correlation power, parabolic refinement, one
  robust straight line. Its rms residual is about 1.4 samples, but over 10 minutes around sunrise that is mostly real
  group-delay change, not clock noise. The least-squares fit is per frame, so a shift of a sample or two is absorbed
  by the taps.
- Main-path Doppler is removed within each frame from the phase of the main correlation, smoothed over 33 frames
  (3.5 s). On a weak or deeply fading signal this needs care.

## Credits
Test idea: Codex (OpenAI), 2026-10-06. Straight-line frame clock: DK8OK's ESSENTIAL app (which credits ZL2AFP and G3PLX).
KiwiSDR recording: kiwiclient by John Seamons. Code and text: Claude (Anthropic) with Mike Burns, AB1LD. Code: MIT licence; text and figures: CC BY 4.0
(see [LICENSE.md](../../LICENSE.md)).
