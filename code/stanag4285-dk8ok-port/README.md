# DK8OK's STANAG 4285 sounder in Python (port of ESSENTIAL V1.1)

A function-by-function Python translation of the measurement core of **Nils Schiffhauer, DK8OK**'s MATLAB app
*DK8OK_STANAG4285_ESSENTIAL* V1.1 (2026), described in his *STANAG 4285: Probing the Ionosphere*. It is the port
compared with our own sounder in the
[2026-10 report](../../reports/2026-10-dk8ok-stanag4285/AB1LD_DK8OK_STANAG4285_2026-10.pdf).

**Method and original code:** DK8OK, who credits its origin to Con Wassilieff, ZL2AFP (STANAG 4285 Receiver), and
suggestions to Peter Martinez, G3PLX. **Published with DK8OK's permission** (Dopplergram group, 2026-10-07): "Please
feel free to use, improve, change and share my software". The terms are his; please keep these credits.
**Port:** Claude (Anthropic AI) for Mike Burns, AB1LD, 2026-10-06; audited by Codex (OpenAI) the same day; plain-WAV
reader added 2026-10-07.

## What it does
The STANAG 4285 USB channel (150-3600 Hz above the suppressed carrier) is filtered out, centred and resampled to
9600 S/s. The 80-symbol sync (PN31, repeated), shaped with a root-raised-cosine filter, is the reference. After
acquisition, one straight-line frame clock is fitted to all reliable correlation peaks, and every frame's complex
correlation is read at that clock at delays −2 … +4 ms. Fourier transforms across frames (64-s Hann windows) give each
delay cell's Doppler spectrum. DK8OK's constants, thresholds and order of operations are unchanged; the header of
`s4285_dk8ok.py` lists exactly which MATLAB functions were ported and which were not (the UI, the wideband feed, the
empirical-PSF residual fit and the burst correlation were not).

## Run it on the sample data
Needs Python 3 with numpy and scipy.
```
python3 s4285_dk8ok.py ../../data/2026-10-01_FUF_8552.6kHz/FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav 8554.4 8552.6 start all fuf
```
Arguments: IQ file, IQ centre (kHz), USB carrier (kHz), start (`start`, a UTC time `YYYY-mm-ddTHH:MM:SS`, or a Unix
time), seconds (a number or `all`), output prefix. It takes a few seconds and writes `fuf.npz` with the complex delay
response of every frame (`rho_iq`, `delay_ms`, `t_frame`), the clock (`global_ppm`, `clock_obs_residual_samples`), the
Doppler image (`DG`, `fd`, `tc`) and DK8OK's noise and competition diagnostics.

**Input:** any plain 2-channel (I, Q) WAV file, int16, int32 or float. Frame times are only as good as the file's
own sample rate, which is taken from its header. To select by time, the file name must carry its start as
`YYYYMMDDTHHMMSSZ`, as the sample does.

**KiwiSDR recordings** (kiwirecorder IQ files, with a GPS stamp on every block) are recognised automatically and read
with [`kiwiwav.py`](../kiwi-tools/) from `../kiwi-tools`. The sample rate is then measured from the GPS stamps over the
interval, so frame times are GPS-accurate, provided the Kiwi had a current GPS solution: the port checks each
block's `last_gps_solution` byte and warns if not. Recordings from public Kiwis work too, including the 20.25-kS/s wide
mode; see [kiwi-tools](../kiwi-tools/) for what to watch (gaps, GPS). Give START as a time and SECONDS as a number:
```
python3 s4285_dk8ok.py ../../data/2026-10-01_FUF_8552.6kHz/20261001T105800Z_8554400_kiwi_iq.wav 8554.4 8552.6 2026-10-01T10:58:00 119 kiwi
```

## Checks
| Test | Result |
|---|---|
| Sample excerpt vs the original GPS-stamped KiwiSDR recording, same 10 minutes | total main-path Doppler −1.119 Hz in both; mean delay profiles within 1 dB on every cell within 25 dB of the peak; frame times within 13 ± 10 µs; frame clock −0.110 vs −0.165 ppm |
| 2-minute Kiwi-format sample vs the plain excerpt, same interval (10:58:00 + 119 s) | total Doppler −1.1680 Hz in both; frame times within 11 ± 12 µs; delay profiles within 1 dB on 57 of 58 cells within 25 dB of the peak (one cell on a steep flank, +0.42 ms, 1.9 dB); frame clock +0.958 vs +0.617 ppm over 2 minutes (see the limitation below) |
| Synthetic signal: paths at 0, −0.5 ms (−12 dB) and +2.3 ms (−9 dB), Doppler −1.100 Hz, relative +0.16 and +0.40 Hz | Doppler −1.1000 Hz; relative Dopplers +0.160 and +0.400 Hz; path levels −10.1 and −8.9 dB (the −0.5 ms value includes the pulse's own tail) |

**Known limitation (V1.1):** the frame clock is fitted to whole-sample peak positions. On the 2-minute synthetic
signals it was off by 0.4-0.5 ppm (about 50 µs, half a sample); on real data the two runs above differ by 0.055 ppm.
DK8OK's V1.2 adds fractional peak refinement for exactly this; it is not in this port.
The acquisition searches tuning in 2-Hz steps, so `residual_hz` can differ by one step between runs. The Doppler in
`fd` is relative to that tuning: add `residual_hz` for the total.

## Changes to the port since the report
- 2026-10-07: reads plain IQ WAV files; for Kiwi files, passes the WAV header rate to the reader (so Kiwis in other IQ
  modes work), looks for it in `../kiwi-tools`, and checks the Kiwi's GPS lock (`kiwi_gps_fresh_fraction` in the
  output). Also fixed: the Kiwi reader's straight-line fit to the GPS stamps is now made
  on times relative to the first sample. Fitted to absolute Unix times (about 1.8 × 10⁹ s) over millions of samples,
  numpy's fit lost precision: up to 0.4 ms of false departures and up to 1 ppm error in the measured sample rate. The
  clock figure for the port in the report (−0.038 ppm over 30 minutes) went through that fit; re-fitting that interval's
  stamps the correct way moves the measured sample rate by 0.08 ppm, so that figure is off by about that much; the Doppler and delay results do not depend on it.
