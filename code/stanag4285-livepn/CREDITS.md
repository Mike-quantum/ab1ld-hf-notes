# Credits: where DK8OK's program went

The source is `DK8OK_STANAG4285_LIVE_PN_V1_4.m`, © 2026 Nils Schiffhauer, DK8OK. Where each credit comes from:

- **The program:** posted by DK8OK to the Dopplergram group (groups.io) on 11 October 2026 as "STANAG4285: P/N
  Sequence and Equalizer". In that post he says he wrote it with the help of ChatGPT.
- **The file's own header:** cites STANAG 4285 Annex A (Fig. A-4) and MathWorks' audiorecorder/audiodevinfo.
- **The lineage of his STANAG 4285 work:** Con Wassilieff, ZL2AFP ("STANAG 4285 Receiver"), with suggestions from
  Peter Martinez, G3PLX. This is as he credits it in the documentation of his earlier ESSENTIAL app
  (S4285_Essential_EN.pdf); the V1.4 file itself does not name them.
- **Permission:** his statement of 7 October 2026 about his STANAG 4285 software, "Please feel free to use, improve,
  change and share my software", and his post of 9 October inviting a version more widely used by the community.
  The Python version is offered back to him.

| File (in `livepn/`) | Taken from DK8OK | His functions |
|---|---|---|
| `s4285_reference.py` | Constants (19200 S/s, 8 samples/symbol, RRC 0.2 / span 10, centre 1800 Hz); his PN31 generator (register 11010, taps 5 XOR 3) and PN80 | first lines of the program |
| `s4285_livepn.py` | His signal path with defaults unchanged: <ul><li>Hilbert transform with an absolute-index mixer, resampling to 19.2 kS/s, RRC filter</li><li>±450 Hz / 15 Hz × 8-phase PN80 search with the one-frame-later check, lock gate 0.48</li><li>per-frame carrier offset and phase fit, PN BER, EVM</li><li>ridge FIR equalizer trained on the previous frame (test, then retrain)</li><li>scheduling: buffer, minimum interval, window</li><li>1-s archive and his column names</li></ul> | tick, paintAudio, analyse, collectArchive, archiveValues |
| `s4285_testsignals.py` | `demo_audio` / `demo_truth` are his demoAudio and its known answer. `stanag4285_audio` is ours. | demoAudio |
| `livepn_offline.py` | His CSV layouts and default file names, his 0.2-s cadence, static versions of his panels | saveCSV, saveArchive, paint, paintArchive |
| `livepn_live.py` | The layout of his Live PN / Equalizer tab and status line | paint, paintAudio |

## Ours, not DK8OK's

**Whole files:**

- `s4285_sources.py`: file, I/Q and KiwiSDR sources, and USB demodulation.
- `compare_with_matlab.py`
- `iq_to_usb_audio.py`
- `code/kiwi-tools/kiwiwav.py` (elsewhere in this repository), which reads KiwiSDR recordings.
- `test_livepn.py`: uses his demo known answer.
- `try_demo.py`

**Options in his files** (all off by default):

- `--complete`
- `--cfo-guard`, which is `cfo_method='guarded'` together with its `ml_offset` and `pn_correlation` helpers.

**External, not included:** kiwiclient / kiwirecorder.py by John Seamons (jks-prv),
https://github.com/jks-prv/kiwiclient.
