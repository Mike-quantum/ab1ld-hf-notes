#!/usr/bin/env python3
"""
Cut a plain IQ WAV excerpt from a GPS-stamped KiwiSDR recording, resampled onto an exact 12000-Hz grid that starts on
a whole UTC second, so that any WAV reader (MATLAB audioread, Python, SDR software) gets correct timing.

Written by Claude (Anthropic AI) for Mike Burns, AB1LD, 2026-10-07. Reader: kiwiwav.py (ours, for kiwirecorder files
from kiwiclient by John Seamons, jks-prv). The Kiwi's true sample rate is about 11998.96 S/s (header says 11999);
every output sample n sits at UTC time START + n / 12000 s, found from a straight line fitted to the Kiwi's own
per-block GPS stamps over the excerpt (fitted on times relative to the first sample, for numerical precision), and windowed-sinc
interpolation (64 taps, Blackman window).
Usage: make_excerpt.py KIWI.wav START_UTC(YYYY-mm-ddTHH:MM:SS) SECONDS OUT.wav
"""
import os
import sys
import datetime as dt
import numpy as np
from scipy.io import wavfile

# kiwiwav.py sits next to this script in the ab1ld-hf-notes repository; in AB1LD's lab it is in ../../hf-antenna-census
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hf-antenna-census'))
import kiwiwav

OUTPUT_RATE = 12000
HALF_TAPS = 32
MARGIN_S = 2.0


def sinc_interpolate(x, positions):
    """Value of the band-limited signal x at fractional sample positions (windowed sinc, 64 taps)."""
    result = np.zeros(len(positions), complex)
    base = np.floor(positions).astype(int)
    fraction = positions - base
    for tap in range(-HALF_TAPS + 1, HALF_TAPS + 1):
        distance = tap - fraction
        window = 0.42 + 0.5 * np.cos(np.pi * distance / HALF_TAPS) + 0.08 * np.cos(2 * np.pi * distance / HALF_TAPS)
        result += x[base + tap] * np.sinc(distance) * window
    return result


def main():
    kiwi_path, start_text, seconds_text, out_path = sys.argv[1:5]
    start = dt.datetime.strptime(start_text, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
    seconds = float(seconds_text)
    recording = kiwiwav.KiwiWav(kiwi_path)
    x, t, ok = recording.read(start - MARGIN_S, seconds + 2 * MARGIN_S)
    if not ok:
        sys.exit('the requested interval touches a gap or a bad GPS stamp; choose another interval')
    sample_number = np.arange(len(t))
    # fit times relative to the first sample: np.polyfit on absolute Unix times (~1.8e9 s) over millions of samples
    # loses precision (false departures of hundreds of microseconds, rate errors of ~0.1-1 ppm)
    relative = t - t[0]
    slope, intercept_relative = np.polyfit(sample_number, relative, 1)
    intercept = t[0] + intercept_relative
    straightness_us = np.max(np.abs(relative - (intercept_relative + slope * sample_number))) * 1e6
    print(f'Kiwi rate over the interval {1 / slope:.4f} S/s; largest departure of the GPS stamps from a straight line '
          f'{straightness_us:.1f} us')
    output_times = start + np.arange(int(round(seconds * OUTPUT_RATE))) / OUTPUT_RATE
    # straight-line clock from the stamps
    positions = (output_times - intercept) / slope
    y = np.zeros(len(positions), complex)
    chunk = 1_000_000
    for first in range(0, len(positions), chunk):
        y[first:first + chunk] = sinc_interpolate(x, positions[first:first + chunk])
    stereo = np.empty((len(y), 2), np.int16)
    stereo[:, 0] = np.clip(np.round(y.real * 32768), -32768, 32767)
    stereo[:, 1] = np.clip(np.round(y.imag * 32768), -32768, 32767)
    wavfile.write(out_path, OUTPUT_RATE, stereo)
    peak = np.max(np.abs(stereo))
    print(f'wrote {out_path}: {len(y)} samples, {seconds:.0f} s, first sample at {start_text} UTC, peak {peak} of 32767')


if __name__ == '__main__':
    main()
