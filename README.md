# AB1LD HF notes

Reports, figures and selected code from passive HF propagation measurements at amateur station **AB1LD**
(Arlington, Massachusetts, USA): oblique chirp sounders, STANAG 4285 transmissions, CODAR ocean radars and
beacons, received on a GPS-timed KiwiSDR and an ELAD FDM-S3.

These are working notes shared with the [Dopplergram group](https://groups.io/g/dopplergram) and correspondents. They
are longer than a group post, and they change as the work develops; every version stays in this repository's history.

## Using the code
Everything here is plain Python 3. If you have not used Python before:

1. **Install Python 3.10 or newer.** From [python.org](https://www.python.org/downloads/) (Windows, macOS), with your
   Linux distribution's package manager, or with [Miniforge](https://github.com/conda-forge/miniforge) if you prefer
   conda. The code was tested with Python 3.10 and 3.13 (numpy 2.2 and 2.5, scipy 1.15 and 1.18).
2. **Get the files.** On this page, *Code → Download ZIP* and unpack it, or `git clone
   https://github.com/Mike-quantum/ab1ld-hf-notes`. The sample data (about 35 MB) is included either way.
3. **Install the three libraries** (in a terminal, in the unpacked folder):
   ```
   python3 -m pip install -r requirements.txt
   ```
   On Windows type `py` instead of `python3`. With conda: `conda install numpy scipy matplotlib`.
4. **Check that it works**, with the sample data (each takes under a minute):
   ```
   cd code/stanag4285-probe-witness
   python3 test_s4285_probe_witness.py
   python3 s4285_probe_witness.py ../../data/2026-10-01_FUF_8552.6kHz/FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav 0 out
   cd ../stanag4285-dk8ok-port
   python3 s4285_dk8ok.py ../../data/2026-10-01_FUF_8552.6kHz/FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav 8554.4 8552.6 start all fuf
   ```
   The test should end with `ALL PASSED`; the other two write their results next to the scripts (`out.png`,
   `out_summary.txt`, `fuf.npz`). Compare `out_summary.txt` with `results/witness_FUF8552_105500Z_summary.txt`.

Each folder's README explains its program, its inputs and how it was checked. **To use your own recordings:** any
2-channel (I, Q) WAV file works, or a KiwiSDR recording made with `kiwirecorder.py ... -m iq -w` (see
[kiwi-tools](code/kiwi-tools/)). You need the frequency at the centre of the IQ spectrum and the STANAG 4285 carrier.

## Reports
| Date | Report | For |
|---|---|---|
| 2026-10 | [STANAG 4285: DK8OK's method in Python, compared with ours (PDF)](reports/2026-10-dk8ok-stanag4285/AB1LD_DK8OK_STANAG4285_2026-10.pdf) | Nils Schiffhauer, DK8OK |

## Code
| Date | Code | What it does |
|---|---|---|
| 2026-10 | [kiwi-tools](code/kiwi-tools/) | Reading GPS-stamped KiwiSDR (kiwirecorder) IQ files with the GPS time of every sample; cutting and converting them. |
| 2026-10 | [stanag4285-dk8ok-port](code/stanag4285-dk8ok-port/) | DK8OK's STANAG 4285 sounder (ESSENTIAL V1.1 measurement core) in Python, published with his permission: frame clock, complex delay response per frame, Doppler image. Reads plain IQ WAV files and GPS-stamped KiwiSDR recordings, including from public Kiwis. |
| 2026-10 | [stanag4285-probe-witness](code/stanag4285-probe-witness/) | Tests whether features in a STANAG 4285 delay profile are real: least-squares response from the sync and, independently, from the probe symbols, scored by held-out prediction. Standalone Python, with the exact sync/probe symbol template and known-answer tests. |

## Data
| Date | Data | Notes |
|---|---|---|
| 2026-10-01 | [FUF 8552.6 kHz IQ samples](data/2026-10-01_FUF_8552.6kHz/) | STANAG 4285, Martinique to Arlington MA across sunrise: 10 min as plain IQ WAV (12000 S/s, GPS-timed) and 2 min in KiwiSDR format |

## Comments and questions
Please open an [Issue](https://github.com/Mike-quantum/ab1ld-hf-notes/issues) on this repository, or write in the Dopplergram group.

## How the work is done
The analysis code and much of the text are written by Claude (Anthropic's AI assistant) working with me: I choose
the questions, run the station, check the results and decide what is published. Outside code, methods, data and ideas
are credited in each report and collected in [ATTRIBUTION.md](ATTRIBUTION.md).

## Licence
See [LICENSE.md](LICENSE.md).
