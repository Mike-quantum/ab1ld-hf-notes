# FUF, 8552.6 kHz USB, STANAG 4285: IQ samples, 2026-10-01

Two files from the same GPS-stamped KiwiSDR recording:

| File | Format | Interval (UTC) | Size |
|---|---|---|---|
| `FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav` | plain IQ WAV, exactly 12000 S/s | 10:55:00-11:05:00 | 28.8 MB |
| `20261001T105800Z_8554400_kiwi_iq.wav` | KiwiSDR kiwirecorder format (GPS stamp on every 512-sample block), byte-for-byte from the original | 10:57:59.97-10:59:59.99 | 5.8 MB |

The Kiwi-format file lets you test a Kiwi reader (ours is in [code/kiwi-tools](../../code/kiwi-tools/)) and check it
against the plain file over the same two minutes.

## The plain 10-minute excerpt

| | |
|---|---|
| File | `FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav` (28.8 MB) |
| Format | WAV, 2 channels (I, Q), int16, **exactly 12000 S/s**, 7 200 000 samples |
| Time | sample n is at **2026-10-01 10:55:00 UTC + n/12000 s** (GPS) |
| Tuning | IQ centre 8554.4 kHz. The USB carrier is 8552.6 kHz, so the STANAG 4285 signal centre (carrier + 1800 Hz) is at **0 Hz** |
| Transmitter | French Navy, Fort-de-France, Martinique (FUF), about 3200 km |
| Receiver | KiwiSDR (GPS-disciplined), loop-on-ground antenna, AB1LD, Arlington MA, USA |
| Level | as recorded, peak about −6 dBFS |

The interval runs across local sunrise and includes the appearance of a component about +2.3 ms behind the main path
(absent before about 11:00 UTC, strong afterwards).

## How it was made
The KiwiSDR records about 11998.96 S/s (its header says 11999) with a GPS time stamp on every 512-sample block.
[`make_excerpt.py`](../../code/kiwi-tools/make_excerpt.py) fitted a straight line to those stamps over the excerpt (they lie within 1 µs of it; the fit is
made on times relative to the first sample, because a fit to absolute Unix times loses precision) and resampled the IQ onto an exact 12000-S/s grid starting on the whole second (64-tap Blackman-windowed sinc).
Nothing else was done to the signal.

## The 2-minute Kiwi-format sample
Cut with [`cut_kiwi.py`](../../code/kiwi-tools/cut_kiwi.py): the original header and 2813 original records (1 440 256
samples, about 11998.96 S/s), only the RIFF size field updated. Keep the time stamp at the start of its name: the GPS
week is taken from it.

## Use
CC BY 4.0: free to use with credit to "Mike Burns, AB1LD". Used in
[code/stanag4285-probe-witness](../../code/stanag4285-probe-witness/) and
[code/stanag4285-dk8ok-port](../../code/stanag4285-dk8ok-port/).
