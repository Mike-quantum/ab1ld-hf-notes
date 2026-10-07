# KiwiSDR IQ file tools

Reading the GPS-stamped IQ recordings that **kiwirecorder** writes (`kiwirecorder.py -w`, from
[kiwiclient](https://github.com/jks-prv/kiwiclient) by John Seamons and contributors). Each 512-sample block of such a
file carries its own GPS time stamp, so every sample has a GPS time; this is what makes absolute timing possible.

| File | What it is |
|---|---|
| `kiwiwav.py` | The reader (AB1LD, 2026-10-01, revised after an independent audit by Codex). `KiwiWav(path).read(t0, seconds)` returns full-scale complex samples, the GPS (Unix) time of every sample, and a flag that is False if the interval touches a missing, non-monotonic or gapped stamp. Used by the [DK8OK port](../stanag4285-dk8ok-port/) for Kiwi files. |
| `cut_kiwi.py` | Cuts a piece out of a Kiwi file without changing its format (original header and stamped blocks, byte for byte). Made the 2-minute Kiwi-format sample in [data/](../../data/2026-10-01_FUF_8552.6kHz/). |
| `make_excerpt.py` | Converts an interval of a Kiwi file into a plain IQ WAV on an exact 12000-S/s grid timed from the GPS stamps. Made the 10-minute plain sample in [data/](../../data/2026-10-01_FUF_8552.6kHz/). |

Needs Python 3 with numpy (and scipy for `make_excerpt.py`).

## Recordings from public KiwiSDRs
The same tools read kiwirecorder files from the public network (e.g. `kiwirecorder.py -s HOST -p PORT -f FREQ_kHz
-m iq -w`). Tested on recordings from two public Kiwis (KM3T, about 12 kS/s; KA1GXR, 20.25-kS/s wide mode): same
record layout, stamps on a straight line to within 1 µs. Three things to watch:
- **Sample rate:** public Kiwis may run the 20.25-kS/s wide mode. Pass the header rate, `KiwiWav(path, fs=rate)`.
- **Gaps:** a stream over the internet can drop data. One of the test recordings lost 109 s in 7 gaps. `read()` flags
  any interval that touches a gap (`ok` False), and the port refuses it; choose a clean stretch.
- **GPS:** not every public Kiwi has a current GPS solution. One of the two test Kiwis reported none for its whole
  recording (`last_gps_solution` 252 throughout). Doppler and relative delays are unaffected; absolute times, and
  comparisons between Kiwis, are not GPS-true. See the GPS-lock note below.

## Things to know about the reader
- **File layout:** 36-byte RIFF + fmt header, then 2074-byte records: a `kiwi` chunk (GPS seconds of week and
  nanoseconds) followed by a `data` chunk of 512 complex int16 samples.
- **GPS week:** the stamps give only seconds within the GPS week, so the week is taken from the recording start in the
  **file name** (`YYYYMMDDTHHMMSSZ`, as kiwirecorder names files). Keep that time stamp when renaming; the reader
  refuses a file whose stamps disagree with its name by more than 120 s. GPS − UTC = 18 s is assumed (valid since 2017).
- **Sample rate:** `KiwiWav(path)` defaults to 11998.9554 S/s, the measured rate of AB1LD's Kiwi in its 12-kS/s IQ
  mode. For another Kiwi or mode pass `fs=` (the WAV header rate is close enough; the DK8OK port does this, then
  measures the true rate from the stamps).
- **Index file:** on first use it writes a small seek index next to the recording (`<name>.idx256.npz`).
- **Straight-line fits to the stamps:** fit times relative to the first sample, not absolute Unix times. A
  least-squares line through absolute times (about 1.8 × 10⁹ s) over millions of samples loses precision in numpy: up
  to 0.4 ms of false departures and 1 ppm of rate error (found 2026-10-07; the tools here and the port do it the
  correct way). The reader's own fit, in `iter_blocks`, runs over short chunks and is accurate to about 2 µs.
- **GPS lock:** besides the time stamp, every block carries a byte that kiwiclient calls `last_gps_solution`. In the
  KiwiSDR server it is the number of seconds since the Kiwi's last GPS position solution, capped at 252, and 255 when
  the Kiwi has no GPS clock at all. A Kiwi without a recent solution keeps stamping blocks from its own clock: the
  stamps stay smooth, but they are not GPS time. `kiwiwav.py` does not look at this byte; the DK8OK port does, and
  warns.
- **Style:** the reader is older and more compact than the other code here; it is published as used in the
  measurements.
