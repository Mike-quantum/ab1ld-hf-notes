#!/usr/bin/env python3
"""
STANAG 4285 PN80 channel sounder -- Python port of the measurement core of DK8OK's MATLAB app
"DK8OK_STANAG4285_ESSENTIAL" V1.1 (2026).

ATTRIBUTION (Mike Burns' rule: credit all external code, methods and ideas):
  Original MATLAB code and method: Nils Schiffhauer, DK8OK, (c) 2026, "STANAG 4285: Probing the Ionosphere"
  (S4285_Essential_EN.pdf) and DK8OK_STANAG4285_ESSENTIAL.m, received via the Dopplergram group 2026-10-06.
  DK8OK writes that the project "originated from STANAG 4285 Receiver by Con Wassilieff, ZL2AFP", thanks
  Peter Martinez, G3PLX, for suggestions, was created with the help of ChatGPT, and that "Everyone is encouraged
  to try out this app, improve it, and port it to other systems."
  Port: Claude (AI assistant) for Mike Burns, AB1LD, 2026-10-06. Ported: the functions v37_residual_core
  (channelizer, PN80/RRC reference, global PN clock, complex delay response rhoIQ), pnMatch, sampleCorrelation,
  acquirePnAcrossInterval, pnAcquisitionWindow, and the analysis part of prepareSpectrum and buildCube.
  NOT ported: the UI, the wideband feed (v29L_prepare_wide_iq; our Kiwi files are already narrow), the empirical-PSF
  residual fit (rhoHat/rhoRes, not used by the Essential displays) and the 8-segment burst correlation.
  Constants, thresholds and the order of operations are DK8OK's; where Python needs a different call, the MATLAB
  call is named in a comment. Our only additions are the Kiwi reader and the GPS time of the first sample.
  2026-10-06 (after the Codex audit, codex/stanag-audit-2026-10-06): reader fixed (P1/P2: GPS-measured rate, single
  scaling, gaps refused), DK8OK's input/clock guards restored (P3), his noise/competition outputs and NaN handling
  ported (P4), actual clock-fit residuals saved (M2). (The pre-audit version is kept in AB1LD's lab, not here.)
  2026-10-07: reads plain 2-channel IQ WAV files as well as KiwiSDR files (read_plain_wav_interval, format detected
  automatically), so that others can repeat the analysis on the published sample data; START may be a UTC time, a
  Unix time or 'start', SECONDS a number or 'all'. Measurement core unchanged. Also fixed: the Kiwi reader's
  straight-line fit to the GPS stamps is now made on times relative to the first sample (the fit to absolute Unix
  times lost precision: up to ~0.4 ms false departures and ~1 ppm error in the measured rate). (The version
  before this change is kept in AB1LD's lab, not here.)
  For Kiwi files (also 2026-10-07): the WAV header rate is passed to kiwiwav, so Kiwis in other IQ modes (e.g. the
  20.25-kS/s wide mode common on public Kiwis) work; and the per-block 'last GPS solution' byte is checked, with a
  warning when the Kiwi had no recent GPS solution (saved as kiwi_gps_fresh_fraction).
  2026-10-12: rewritten for readability (one step per line, descriptive names, comments on each step; residual_core
  split into named helpers). Function names, arguments and results unchanged; outputs verified bit-identical to the
  previous version on the published FUF excerpt, a Kiwi file and unit inputs. The previous version is in this
  repository's history.
Readers: plain IQ WAV via scipy; KiwiSDR files via kiwiwav.py (ours; kiwirecorder format from kiwiclient, jks-prv).

What it does (DK8OK's method): the selected USB channel (150-3600 Hz above the suppressed carrier) is filtered out,
moved so the 1800-Hz audio centre is at 0 Hz and resampled to 9600 Hz (4 samples per symbol). The 80-symbol sync
sequence (PN31, repeated) shaped with a root-raised-cosine filter is the reference. After acquisition (frame phase +
residual tuning), one straight-line clock (frame number -> sample position) is fitted to all reliable correlation
peaks; at that clock every frame's complex correlation is taken at relative delays -2 ... +4 ms (rhoIQ). Fourier
transforms across frames (64-s Hann windows) give each delay cell's Doppler spectrum; for each Doppler bin the
strongest delay cell in -0.25 ... +2.9 ms is kept (the "PN Channel Dynamic" image).

Usage: s4285_dk8ok.py IQ.wav IQ_CENTRE_KHZ USB_CARRIER_KHZ START SECONDS OUT_PREFIX
  IQ.wav            a plain 2-channel (I, Q) WAV file (int16, int32 or float), or a KiwiSDR kiwirecorder IQ file
  IQ_CENTRE_KHZ     the frequency at 0 Hz in the IQ spectrum
  USB_CARRIER_KHZ   the STANAG 4285 suppressed (USB) carrier; the signal occupies carrier + 150 ... 3600 Hz
  START             'start' (beginning of the file), a UTC time YYYY-mm-ddTHH:MM:SS, or a Unix time. Selecting by
                    time in a plain WAV needs the file's start time in its name as YYYYMMDDTHHMMSSZ.
  SECONDS           how much to analyse, or 'all'
Example, the sample data published with this code:
  s4285_dk8ok.py FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav 8554.4 8552.6 start all fuf
Timing: for a plain WAV the sample rate is the header rate, so frame times and the clock in ppm are only as good as
the file's own rate; the sample excerpt is resampled onto an exact, GPS-timed 12000-S/s grid. For a KiwiSDR file the
rate is measured from the GPS stamps over the interval.
"""
import os
import re
import struct
import sys
import time
import datetime as dt
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.io import wavfile

# ---------------------------------------------------------------------------------------------- constants
# Exactly as in DK8OK's v37_residual_core.
RS = 2400                     # STANAG 4285 symbol rate (symbols/s)
FSWORK = 9600                 # working sample rate after channelizing (S/s)
SPS = 4                       # samples per symbol at FSWORK
FRAME_SAMP = 256 * SPS        # one 256-symbol frame = 1024 samples at FSWORK
FRAME_SEC = 256 / RS          # one frame = 106.67 ms
RRC_BETA = 0.20               # root-raised-cosine roll-off
RRC_SPAN = 10                 # root-raised-cosine length in symbols
DELAY_MS = (-2, 4)            # relative delays at which every frame's correlation is sampled (ms)
ACQ_SECONDS = 8               # length of one acquisition window (s)
USB_AUDIO_HZ = 1800           # audio centre of the signal above the USB carrier (Hz)
TUNE_SEARCH_HZ = 100          # acquisition searches +/- this residual tuning (Hz) ...
TUNE_STEP_HZ = 2              # ... in steps of this (Hz)
PN31_BITS = '0101100111110001101110101000010'
# prepareSpectrum / Essential UI defaults (DK8OK).
APERTURE_S = 64               # length of each Doppler FFT window (s)
DELAY_GATE_MS = (-0.25, 2.9)  # delays searched for the strongest path in each Doppler bin (ms)

# DK8OK's acquisition and clock thresholds.
MIN_PEAK = 0.12               # minimum normalised correlation of the frame-folded peak
MIN_CONTRAST = 2.5            # minimum peak / median of the folded correlation
MIN_FRAMES = 30               # minimum number of frames / clock observations
MAX_CLOCK_PPM = 200           # largest believable frame-clock error (ppm)
EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------------------------- input files
def is_kiwi_wav(path):
    """[Ours, 2026-10-07] True for a kiwirecorder IQ file: right after the 'fmt ' chunk comes a 'kiwi' GPS-stamp chunk."""
    with open(path, 'rb') as f:
        f.seek(36)
        chunk_name = f.read(4)
    return chunk_name == b'kiwi'


def start_from_filename(path):
    """[Ours, 2026-10-07] Unix time of a YYYYMMDDTHHMMSSZ stamp in the file name, or None if there is none."""
    match = re.search(r'(\d{8}T\d{6})Z', os.path.basename(path))
    if not match:
        return None
    stamp = dt.datetime.strptime(match.group(1), '%Y%m%dT%H%M%S').replace(tzinfo=dt.timezone.utc)
    return stamp.timestamp()


def parse_start(text):
    """[Ours, 2026-10-07] 'start' -> None; 'YYYY-mm-ddTHH:MM:SS' (UTC) or a number (Unix time) -> Unix time."""
    if text == 'start':
        return None
    if 'T' in text:
        stamp = dt.datetime.strptime(text, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=dt.timezone.utc)
        return stamp.timestamp()
    return float(text)


def read_plain_wav_interval(path, start_unix, seconds):
    """[Ours, 2026-10-07] An interval of a plain 2-channel IQ WAV file as full-scale complex samples.
    Returns the same four things as read_kiwi_interval: samples, sample rate (the header rate), Unix time of the first
    sample (from the YYYYMMDDTHHMMSSZ stamp in the file name; NaN if there is none) and the header rate."""
    rate, data = wavfile.read(path, mmap=True)
    if data.ndim != 2 or data.shape[1] != 2:
        raise RuntimeError('expected a 2-channel (I, Q) WAV file')

    # Integer samples are scaled so that full scale is 1.
    if data.dtype == np.int16:
        scale = 32768.0
    elif data.dtype == np.int32:
        scale = 2147483648.0
    else:
        scale = 1.0

    # First sample of the interval.
    file_start = start_from_filename(path)
    if start_unix is None:
        first = 0
    elif file_start is None:
        raise RuntimeError('to select by time, the file name must carry its start time as YYYYMMDDTHHMMSSZ; '
                           'or give START as "start"')
    else:
        first = int(round((start_unix - file_start) * rate))
        if first < 0 or first >= len(data):
            raise RuntimeError('START lies outside the file')

    # Number of samples.
    if seconds is None:
        count = len(data) - first
    else:
        count = int(seconds * rate)
        if first + count > len(data):
            raise RuntimeError('recording is shorter than the requested interval')

    piece = np.asarray(data[first:first + count], dtype=float)
    in_phase = piece[:, 0]
    quadrature = piece[:, 1]
    z = (in_phase + 1j * quadrature) / scale

    if file_start is None:
        t0 = float('nan')
    else:
        t0 = file_start + first / rate
    return z, float(rate), t0, rate


# Layout of a kiwirecorder IQ file: a 36-byte header, then records of 2074 bytes holding 512 samples each.
KIWI_HEADER_BYTES = 36
KIWI_RECORD_BYTES = 2074
KIWI_SAMPLES_PER_RECORD = 512
GPS_SOLUTION_STALE = 252          # KiwiSDR caps "seconds since the last GPS solution" at 252; 255 = no GPS clock at all


def kiwi_gps_status(path, first_record, record_count):
    """[Ours, 2026-10-07] The 'last_gps_solution' byte of every record in the interval (the first byte after the 'kiwi'
    chunk header; kiwiclient unpacks it as last_gps_solution). In the KiwiSDR server (rx_sound.cpp) it is the number of
    seconds since the Kiwi's last GPS position solution, capped at 252, and 255 when the Kiwi has no GPS clock ticks.
    A Kiwi without a recent solution keeps stamping blocks from its own clock: the stamps stay smooth, but they are
    not GPS-true. Returns (fraction of records with a solution younger than 252 s, largest value seen)."""
    values = []
    with open(path, 'rb') as f:
        for record in range(first_record, first_record + record_count):
            byte_position = KIWI_HEADER_BYTES + record * KIWI_RECORD_BYTES + 8
            f.seek(byte_position)
            byte = f.read(1)
            if len(byte) == 1:
                values.append(byte[0])
    values = np.array(values)
    fresh_fraction = float(np.mean(values < GPS_SOLUTION_STALE))
    return fresh_fraction, int(values.max())


def read_kiwi_interval(path, start_unix, seconds):
    """[Ours; fixed 2026-10-06 after the Codex audit, findings P1/P2.] The interval through kiwiwav.read(): full-scale
    complex samples (kiwiwav already divides by 32768 -- the earlier extra division made amplitudes 90.3 dB low), the
    per-sample GPS times, the sample rate MEASURED from those stamps over this interval, and the WAV header rate.
    An interval with missing or gapped stamps is refused rather than silently spliced.
    Needs kiwiwav.py (ours) next to this script, in ../kiwi-tools or in ../hf-antenna-census."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(1, os.path.join(here, '..', 'kiwi-tools'))           # layout of the ab1ld-hf-notes repository
    sys.path.insert(2, os.path.join(here, '..', 'hf-antenna-census'))    # layout of AB1LD's lab
    try:
        import kiwiwav
    except ImportError:
        raise RuntimeError('KiwiSDR-format files need kiwiwav.py (AB1LD): next to this script or in ../kiwi-tools')
    if start_unix is None or seconds is None:
        raise RuntimeError('for a KiwiSDR file give START as a time and SECONDS as a number')

    # The rate written in the WAV header (bytes 24..27).
    with open(path, 'rb') as f:
        f.seek(24)
        header_rate = struct.unpack('<I', f.read(4))[0]

    # The header rate, not kiwiwav's default (AB1LD's Kiwi, ~11999 S/s), so that Kiwis in other IQ modes work too; the
    # true rate is measured from the GPS stamps below.
    wav = kiwiwav.KiwiWav(path, fs=float(header_rate))
    x, t, ok = wav.read(start_unix, seconds)

    # Check that the Kiwi had a recent GPS solution throughout the interval.
    first_record = wav.find_block(start_unix)
    record_count = int(np.ceil(seconds * header_rate / KIWI_SAMPLES_PER_RECORD))
    gps_fresh_fraction, gps_last_max = kiwi_gps_status(path, first_record, record_count)
    if gps_fresh_fraction < 0.99:
        print(f'WARNING: this Kiwi had no recent GPS solution for {100 * (1 - gps_fresh_fraction):.0f} % of the interval '
              f'(seconds since last solution up to {gps_last_max}; 252 = 252 s or more, 255 = no GPS clock). Its time '
              f'stamps then come from its own clock: Doppler and relative delays are unaffected, but frame times and '
              f'the clock in ppm are not GPS-true.')
    if not ok:
        raise RuntimeError('interval has missing or gapped GPS stamps; choose a clean interval')

    # Drop samples before the requested start.
    keep = t >= start_unix
    x = x[keep]
    t = t[keep]

    # Sample rate from a straight line through the GPS times.  The fit uses times RELATIVE to the first sample:
    # a straight-line fit to absolute Unix times (~1.8e9 s) over millions of samples loses precision in np.polyfit
    # (up to ~0.4 ms false departures and ~1 ppm rate error; found 2026-10-07).
    sample_number = np.arange(len(t))
    relative = t - t[0]
    line = np.polyfit(sample_number, relative, 1)
    seconds_per_sample = line[0]
    fs_measured = 1 / seconds_per_sample
    departure = np.abs(relative - np.polyval(line, sample_number))
    if np.max(departure) > 2 / fs_measured:
        raise RuntimeError('GPS stamps do not lie on one straight line within 2 samples')

    need = int(seconds * fs_measured)
    if len(x) < 0.99 * need:
        raise RuntimeError('recording is shorter than the requested interval')
    first_sample_time = t[0] + line[1]
    return x[:need].astype(complex), fs_measured, first_sample_time, header_rate, gps_fresh_fraction


# ---------------------------------------------------------------------------------------------- reference
def rcosdesign_sqrt(beta, span, sps):
    """MATLAB rcosdesign(beta, span, sps, 'sqrt'): root-raised-cosine taps, scaled to unit energy.

    Pulse, with t in symbol periods:
        p(t) = [sin(pi t (1 - beta)) + 4 beta t cos(pi t (1 + beta))] / [pi t (1 - (4 beta t)^2)]
    with the limits used at t = 0 and at t = +/- 1 / (4 beta), where the formula is 0/0.
    """
    tap_times = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.zeros(len(tap_times))
    for i, t in enumerate(tap_times):
        if abs(t) < 1e-12:
            taps[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * t) - 1) < 1e-12:
            sine_part = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            cosine_part = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            taps[i] = beta / np.sqrt(2) * (sine_part + cosine_part)
        else:
            numerator = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
            denominator = np.pi * t * (1 - (4 * beta * t) ** 2)
            taps[i] = numerator / denominator
    return taps / np.sqrt(np.sum(taps ** 2))


def pn80_reference():
    """DK8OK's reference: the 80 sync symbols, RRC-shaped at 4 samples per symbol, unit energy.

    MATLAB: b80 = [b31 b31 b31(1:18)], a80 = 1-2*b80 (bit 0 -> +1, bit 1 -> -1 in this program), upsampled x4,
    RRC-shaped, the 80*SPS samples after the filter delay kept.
    """
    b31 = np.array([int(character) for character in PN31_BITS])
    b80 = np.concatenate([b31, b31, b31[:18]])
    a80 = 1 - 2 * b80

    rrc = rcosdesign_sqrt(RRC_BETA, RRC_SPAN, SPS)
    impulses = np.zeros(len(a80) * SPS)
    impulses[::SPS] = a80
    shaped = np.convolve(impulses, rrc)

    # Skip the filter's group delay (half its length) so the reference starts at the first symbol.
    group_delay = RRC_SPAN * SPS // 2
    reference = shaped[group_delay:group_delay + 80 * SPS].astype(complex)
    return reference / np.sqrt(np.sum(np.abs(reference) ** 2))


# ---------------------------------------------------------------------------------------------- channelizer
def channelize(z, fs_in, delta_rf_hz):
    """DK8OK V2.9b channelizer: keep 150..3600 Hz above the suppressed carrier, put 1800 Hz at 0 Hz, resample to 9600.

    z: complex IQ samples at fs_in; delta_rf_hz: USB carrier minus IQ centre frequency (Hz).
    Steps: tune the carrier to 0 Hz; shift the middle of the 150..3600-Hz band to 0 Hz; low-pass (Kaiser FIR);
    shift back; then shift the 1800-Hz audio centre to 0 Hz and resample to FSWORK.
    """
    z = z - np.mean(z)                                     # remove any DC offset

    # The band kept, above the carrier: 150 Hz up to 3600 Hz (or 0.45 fs_in if that is lower).
    f_lo = 150.0
    f_hi = min(3600.0, 0.45 * fs_in)
    band_centre = (f_lo + f_hi) / 2
    band_half_width = (f_hi - f_lo) / 2

    # FIR order: 160 or more, at most 1600, even (so the delay is a whole number of samples).
    order = max(160, 2 * int(np.ceil(fs_in / 80)))
    order = min(order, 1600)
    if order % 2:
        order += 1

    # DK8OK's input guards (restored after audit P3).
    if f_hi <= f_lo + 1000:
        raise ValueError('Input sample rate is too low for the STANAG USB channelizer.')
    if delta_rf_hz + f_lo <= -fs_in / 2 or delta_rf_hz + f_hi >= fs_in / 2:
        raise ValueError('Selected STANAG USB information band lies outside the I/Q Nyquist range.')

    cutoff = min(0.98, band_half_width / (fs_in / 2))
    lowpass = signal.firwin(order + 1, cutoff, window=('kaiser', 8.0))   # MATLAB fir1(ord, cutoff, 'low', kaiser(ord+1, 8))

    sample_number = np.arange(len(z))
    tuned = z * np.exp(-2j * np.pi * delta_rf_hz * sample_number / fs_in)            # carrier -> 0 Hz
    band_at_zero = tuned * np.exp(-2j * np.pi * band_centre * sample_number / fs_in)  # band middle -> 0 Hz
    filtered = signal.oaconvolve(band_at_zero, lowpass)[:len(band_at_zero)]           # MATLAB fftfilt (causal FIR)

    # Compensate the FIR's group delay of order/2 samples (shift earlier, pad the end with zeros).
    group_delay = order // 2
    filtered = np.concatenate([filtered[group_delay:], np.zeros(group_delay, complex)])

    selected = filtered * np.exp(2j * np.pi * band_centre * sample_number / fs_in)    # band back in place
    audio_centred = selected * np.exp(-2j * np.pi * USB_AUDIO_HZ * sample_number / fs_in)   # 1800 Hz -> 0 Hz
    up, down = rational(FSWORK / fs_in)
    return signal.resample_poly(audio_centred, up, down)                              # MATLAB resample(z, p, q)


def rational(ratio):
    """p/q for the resampling ratio (MATLAB rat(ratio, 1e-12)); Kiwi: 9600/12000 = 4/5."""
    fraction = Fraction(ratio).limit_denominator(100000)
    return fraction.numerator, fraction.denominator


# ---------------------------------------------------------------------------------------------- correlation
def pn_match(z, ref):
    """DK8OK pnMatch: c[k] = ref' * z[k:k+L] for every k (one FFT convolution), e[k] = energy of z[k:k+L].

    c / sqrt(e) is the normalised correlation (1 = perfect match, since ref has unit energy).
    """
    length = len(ref)
    if len(z) < length:
        return np.zeros(0, complex), np.zeros(0)

    # Convolving with the reversed, conjugated reference = correlating with the reference.
    correlation = signal.fftconvolve(z, np.conj(ref[::-1]), mode='valid')

    # Energy in every window of `length` samples, from a running sum of |z|^2.
    power = np.abs(z) ** 2
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    energy = cumulative[length:] - cumulative[:-length]
    return correlation, energy[:len(correlation)]


def sample_correlation(matched, positions):
    """DK8OK sampleCorrelation: values of `matched` at fractional positions by linear interpolation (0-based here);
    NaN where a position falls outside."""
    positions = np.asarray(positions, dtype=float)
    below = np.floor(positions).astype(int)
    fraction = positions - below
    valid = (below >= 0) & (below + 1 < len(matched))
    safe_index = np.where(valid, below, 0)                 # any valid index where the position is outside

    values = (1 - fraction) * matched[safe_index] + fraction * matched[safe_index + 1]
    if np.iscomplexobj(matched):
        values = values.astype(complex)
    else:
        values = values.astype(float)
    values[~valid] = np.nan
    return values


# ---------------------------------------------------------------------------------------------- acquisition
def pn_acquisition_window(z, ref, fs, frame_n, grid):
    """DK8OK pnAcquisitionWindow: for each trial tuning in `grid` (Hz), fold the normalised correlation by frames
    and take the median; the best tuning has the highest folded peak that passes the peak and contrast thresholds.

    Returns (peak, frame phase in samples, tuning in Hz, contrast); peak is -inf if nothing passed.
    """
    sample_number = np.arange(len(z))
    best = (-np.inf, 0, 0.0, 0.0)
    for df in grid:
        tuned = z * np.exp(-2j * np.pi * df * sample_number / fs)
        cc, ee = pn_match(tuned, ref)
        metric = np.abs(cc) / np.sqrt(np.maximum(ee, EPS))

        # Fold: one row per frame, one column per sample position within a frame; median down the columns.
        n_frames = len(metric) // frame_n
        if n_frames < MIN_FRAMES:
            continue
        by_frame = metric[:n_frames * frame_n].reshape(n_frames, frame_n)
        fold = np.median(by_frame, axis=0)

        phase = int(np.argmax(fold))
        peak = fold[phase]
        contrast = peak / (np.median(fold) + EPS)
        if peak >= MIN_PEAK and contrast >= MIN_CONTRAST and peak > best[0]:
            best = (peak, phase, df, contrast)
    return best


def acquire_pn_across_interval(z, ref, fs, frame_n, window_sec, range_hz, step_hz):
    """DK8OK acquirePnAcrossInterval: try half-overlapping 8-s windows in turn until one acquires.

    The first window gets the full fine tuning grid; later windows a coarse 20-Hz grid, then a fine grid of
    +/-12 Hz around the coarse result.  Returns (start sample of the PN, tuning Hz, peak, window start in s).
    """
    window_n = round(window_sec * fs)
    hop_n = round(window_n / 2)
    min_n = MIN_FRAMES * frame_n + len(ref)

    last_start = max(0, len(z) - window_n)
    starts = list(range(0, last_start + 1, hop_n))
    if starts[-1] != last_start:
        starts.append(last_start)                          # make sure the end of the interval is covered

    for window_number, first in enumerate(starts):
        window = z[first:first + window_n]
        if len(window) < min_n:
            continue

        if window_number == 0:
            grid = np.arange(-range_hz, range_hz + step_hz / 2, step_hz)
        else:
            grid = np.arange(-range_hz, range_hz + 10, 20)
        peak, phase, df, contrast = pn_acquisition_window(window, ref, fs, frame_n, grid)

        if peak >= MIN_PEAK and contrast >= MIN_CONTRAST:
            if window_number > 0:
                fine_low = max(-range_hz, df - 12)
                fine_high = min(range_hz, df + 12)
                fine_grid = np.arange(fine_low, fine_high + step_hz / 2, step_hz)
                peak, phase, df, contrast = pn_acquisition_window(window, ref, fs, frame_n, fine_grid)
            if peak >= MIN_PEAK and contrast >= MIN_CONTRAST:
                print(f'PN80 acquired in window {first / fs:.1f} s | tune {df:+.1f} Hz | quality {peak:.3f} | contrast {contrast:.2f}')
                return first + phase, df, peak, first / fs
    raise RuntimeError('No reliable repeating PN80 found')


# ---------------------------------------------------------------------------------------------- PN clock
def robust_line(x, y, iterations=3):
    """DK8OK's clock fit: a straight line, then `iterations` refits on the points with
    |error - median error| <= max(2, 3 x median absolute deviation)."""
    line = np.polyfit(x, y, 1)
    for _ in range(iterations):
        error = y - np.polyval(line, x)
        deviation = np.abs(error - np.median(error))
        mad = np.median(deviation)
        good = deviation <= max(2, 3 * mad)
        if good.sum() < MIN_FRAMES:
            raise RuntimeError('PN clock estimation is unstable')
        line = np.polyfit(x[good], y[good], 1)
    return line


def _block_limits(first, total, block_n, guard):
    """Sample ranges for one processing block: [first, last] is the block itself, [lo, hi) the block with
    `guard` samples added on each side (clipped to the data), so correlations near the edges are complete."""
    last = min(total, first + block_n) - 1
    lo = max(0, first - guard)
    hi = min(total, last + guard + 1)
    return last, lo, hi


def _adaptive_clock_pass(z, ref, best_phase, best_score, n_frames, block_n, guard):
    """Pass 1 of DK8OK's clock: frame by frame, the correlation peak within +/-12 samples of the running clock
    prediction.  After each block the clock (origin, period) is refitted to the latest 1501 observations and
    moved by at most 1 sample.  Returns the frame numbers and peak positions that were accepted."""
    origin = float(best_phase)
    period = float(FRAME_SAMP)
    frame_ids = []
    peaks = []
    nominal = best_phase + np.arange(n_frames) * FRAME_SAMP        # frame starts if the clock were exact

    for first in range(0, len(z), block_n):
        last, lo, hi = _block_limits(first, len(z), block_n, guard)
        cc, ee = pn_match(z[lo:hi], ref)
        frames_here = np.flatnonzero((nominal >= first) & (nominal <= last))

        for k in frames_here:
            predicted = origin + k * period
            candidates = np.round(predicted).astype(int) + np.arange(-12, 13)
            local = candidates - lo                                 # position within this block's correlation
            inside = (local >= 0) & (local < len(cc))
            candidates = candidates[inside]
            local = local[inside]
            if len(local) == 0:
                continue
            metric = np.abs(cc[local]) / np.sqrt(np.maximum(ee[local], EPS))
            best = int(np.argmax(metric))
            # Accept a clear peak that is not at the edge of the search range.
            strong_enough = metric[best] > max(0.12, 0.35 * best_score)
            not_at_edge = 0 < best < len(local) - 1
            if strong_enough and not_at_edge:
                frame_ids.append(k)
                peaks.append(candidates[best])

        if len(frame_ids) >= MIN_FRAMES and len(frames_here) > 0:
            # Straight line through the latest observations, refitted once without outliers.
            use = slice(max(0, len(frame_ids) - 1501), len(frame_ids))
            fx = np.array(frame_ids[use], float)
            fy = np.array(peaks[use], float)
            line = np.polyfit(fx, fy, 1)
            error = fy - np.polyval(line, fx)
            good = np.abs(error - np.median(error)) <= max(2, 3 * np.median(np.abs(error - np.median(error))))
            if good.sum() >= 20:
                line = np.polyfit(fx[good], fy[good], 1)

            # Adopt the new period if it is believable; move the origin by at most 1 sample.
            if abs(line[0] / FRAME_SAMP - 1) < 200e-6:
                next_frame = frames_here.max() + 1                  # MATLAB: next = max(ix), 1-based = next frame id
                old_prediction = origin + next_frame * period
                new_prediction = np.polyval(line, next_frame)
                period = line[0]
                step = max(-1, min(1, new_prediction - old_prediction))
                origin = old_prediction + step - next_frame * period

    if len(frame_ids) < MIN_FRAMES:
        raise RuntimeError('Too few reliable PN clock observations')
    return frame_ids, peaks


def _reference_psf_width(ref, lag_samp, delay_ms):
    """-3 dB width (ms) of the reference's own autocorrelation (DK8OK psfWidth, used as an exclusion distance)."""
    psf = []
    for lag in lag_samp:
        if lag >= 0:
            a = ref[:len(ref) - lag]
            b = ref[lag:]
        else:
            a = ref[-lag:]
            b = ref[:len(ref) + lag]
        overlap = abs(np.vdot(a, b))
        norm = np.sqrt(np.sum(abs(a) ** 2) * np.sum(abs(b) ** 2))
        psf.append(overlap / norm)
    psf = np.array(psf) / max(psf)
    above = np.flatnonzero(20 * np.log10(np.maximum(psf, 1e-4)) >= -3)
    return delay_ms[above[-1]] - delay_ms[above[0]]


def residual_core(z_in, fs_in, iq_centre_hz, usb_rf_hz):
    """DK8OK v37_residual_core without the PSF residual fit. Returns rhoIQ (delays x frames) and the clock.

    Steps: channelize; acquire (frame phase, residual tuning); remove the residual tuning; pass 1 -- adaptive clock
    observations; pass 2 -- one robust straight-line clock for the whole interval; sample every frame's complex
    correlation at that clock for all delays -2..+4 ms.
    """
    z = channelize(z_in, fs_in, usb_rf_hz - iq_centre_hz)
    ref = pn80_reference()
    best_phase, residual_hz, best_score, acq_start = acquire_pn_across_interval(
        z, ref, FSWORK, FRAME_SAMP, ACQ_SECONDS, TUNE_SEARCH_HZ, TUNE_STEP_HZ)

    sample_number = np.arange(len(z))
    z = z * np.exp(-2j * np.pi * residual_hz * sample_number / FSWORK)

    # Relative delays sampled, in samples and in ms.
    lag_first = round(DELAY_MS[0] * 1e-3 * FSWORK)
    lag_last = round(DELAY_MS[1] * 1e-3 * FSWORK)
    lag_samp = np.arange(lag_first, lag_last + 1)
    delay_ms = 1e3 * lag_samp / FSWORK

    n_frames = (len(z) - best_phase - len(ref) - lag_samp.max()) // FRAME_SAMP + 1
    if n_frames < MIN_FRAMES:                                            # DK8OK guard (restored after audit P3)
        raise RuntimeError('At least 30 complete PN frames are required.')
    block_n = 60 * FSWORK                                                # process 60 s at a time
    guard = FSWORK                                                       # 1 s overlap on each side

    # Pass 1: frame-by-frame peaks near the running clock prediction (adaptive origin/period, as DK8OK).
    frame_ids, peaks = _adaptive_clock_pass(z, ref, best_phase, best_score, n_frames, block_n, guard)

    # Pass 2: one global straight-line clock for the whole interval.
    clock_ids = np.array(frame_ids, float)
    clock_peaks = np.array(peaks, float)
    clock_line = robust_line(np.array(frame_ids, float), np.array(peaks, float))
    global_period = clock_line[0]
    global_ppm = 1e6 * (global_period / FRAME_SAMP - 1)
    if abs(global_ppm) > MAX_CLOCK_PPM:                                  # DK8OK guard (restored after audit P3)
        raise RuntimeError('PN clock drift exceeds 200 ppm.')
    clock_residual = clock_peaks - np.polyval(clock_line, clock_ids)     # actual fit residuals (audit M2), samples
    timing = np.polyval(clock_line, np.arange(n_frames))                 # fractional start sample of every frame

    # Every frame's correlation at every delay, at the global clock (fractional positions, interpolated).
    rho_iq = np.full((len(lag_samp), n_frames), np.nan, complex)         # raw complex correlation
    rho = np.full((len(lag_samp), n_frames), np.nan, complex)            # normalised by the signal energy
    sync_rms = np.full(n_frames, np.nan)                                 # RMS amplitude over the sync sequence
    for first in range(0, len(z), block_n):
        last, lo, hi = _block_limits(first, len(z), block_n, guard)
        cc, ee = pn_match(z[lo:hi], ref)
        frames_here = np.flatnonzero((timing >= first) & (timing <= last))
        if len(frames_here) == 0:
            continue
        # positions[delay, frame] within this block
        positions = lag_samp[:, None] + timing[frames_here][None, :] - lo
        raw = sample_correlation(cc, positions)
        energy = sample_correlation(ee, positions)
        rho_iq[:, frames_here] = raw
        rho[:, frames_here] = raw / np.sqrt(np.maximum(energy, EPS))
        sync_rms[frames_here] = np.sqrt(sample_correlation(ee, timing[frames_here] - lo) / len(ref))

    t_frame = timing / FSWORK                    # seconds from the first sample (DK8OK: (timingSamp-1)/FSWORK, 1-based)
    keep = np.isfinite(sync_rms)                 # frames that lie fully inside the data

    psf_width = _reference_psf_width(ref, lag_samp, delay_ms)
    print(f'FFT PN80 global clock: {global_ppm:+.3f} ppm; residual tuning {residual_hz:+.1f} Hz; '
          f'{keep.sum()} frames; reference -3 dB width {psf_width:.3f} ms')
    return {'rho_iq': rho_iq[:, keep], 'rho': rho[:, keep], 'delay_ms': delay_ms, 't_frame': t_frame[keep],
            'sync_rms': sync_rms[keep], 'global_ppm': global_ppm, 'residual_hz': residual_hz, 'psf_width_ms': psf_width,
            'acquisition_start_s': acq_start, 'acquisition_score': best_score,
            'clock_obs_ids': clock_ids, 'clock_obs_residual_samples': clock_residual}


# ---------------------------------------------------------------------------------------------- Doppler image
def hann_periodic(n):
    """MATLAB hann(n, 'periodic')."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def prepare_spectrum(R, aperture_s=APERTURE_S):
    """DK8OK prepareSpectrum: time-Doppler image of the strongest delay cell per Doppler bin, 64-s Hann windows,
    with his noise-floor and competing-delay outputs (restored after audit P4). As in MATLAB, NaN cells stay NaN
    and are skipped by the maximum (MATLAB max omits NaN); nothing is zero-filled.

    Output arrays have one row per Doppler bin (sorted, centred on the strongest steady line) and one column per
    window:  DG  power of the strongest delay cell;  best_delay  its delay (ms);  noise  that cell's noise floor
    (median power / ln 2);  competition  strongest cell at least `exclusion` ms away, relative to the peak.
    """
    correlation = R['rho_iq']
    t = R['t_frame']
    delays = R['delay_ms']

    # Frames per second, window length (frames), FFT length, window function and window starts (quarter overlap).
    fs_slow = 1 / np.median(np.diff(t))
    nwin = min(len(t), max(8, round(aperture_s * fs_slow)))
    nfft = max(128, 1 << int(np.ceil(np.log2(4 * nwin))))           # >= 4 x zero padding, power of 2
    window = hann_periodic(nwin)
    starts = np.arange(0, len(t) - nwin + 1, max(1, round(nwin / 4)))

    raw_freq = np.arange(-nfft // 2, nfft // 2) * fs_slow / nfft      # Doppler axis after fftshift (Hz)
    raw_gate = np.abs(raw_freq) <= 0.48 * fs_slow
    delay_gate = (delays >= DELAY_GATE_MS[0]) & (delays <= DELAY_GATE_MS[1])
    gate_delay = delays[delay_gate]
    if np.isfinite(R['psf_width_ms']):
        exclusion = max(0.3, R['psf_width_ms'])
    else:
        exclusion = 0.3

    DG = np.full((nfft, len(starts)), np.nan)
    best_delay = np.full((nfft, len(starts)), np.nan)
    noise = np.full((nfft, len(starts)), np.nan)
    competition = np.full((nfft, len(starts)), np.nan)
    tc = np.zeros(len(starts))                                        # centre time of each window
    gated = correlation[delay_gate]
    columns = np.arange(nfft)

    for k, s in enumerate(starts):
        # Doppler spectrum of every delay cell in this window.
        spectrum = np.fft.fftshift(np.fft.fft(gated[:, s:s + nwin] * window, nfft, axis=1), axes=1)
        power = np.abs(spectrum) ** 2
        finite = np.isfinite(power)
        power_or_minus_inf = np.where(finite, power, -np.inf)          # NaN cells never win the maximum

        # Strongest delay cell in every Doppler bin.
        strongest = np.argmax(power_or_minus_inf, axis=0)
        peak = power_or_minus_inf[strongest, columns]
        has = np.isfinite(peak)
        DG[has, k] = peak[has]
        best_delay[has, k] = gate_delay[strongest[has]]

        # Noise floor of each delay cell: median power over the Doppler gate / ln 2.
        gated_power = np.where(finite[:, raw_gate], power[:, raw_gate], np.nan)
        noise_per_delay = np.nanmedian(gated_power, axis=1) / np.log(2)
        noise[has, k] = noise_per_delay[strongest[has]]

        # Competition: the strongest cell at least `exclusion` ms from the winning delay, relative to the peak.
        distance = np.abs(gate_delay[:, None] - gate_delay[strongest][None, :])
        eligible = distance >= exclusion
        alternative = np.where(eligible & finite, power, 0).max(axis=0)
        competition[has, k] = alternative[has] / np.maximum(peak[has], np.finfo(float).tiny)

        tc[k] = np.mean(t[s:s + nwin])

    # Put the steadiest strong line (highest median power over time) at 0 Hz and sort the Doppler axis.
    persistence = np.nanmedian(DG[raw_gate], axis=1)
    reference = raw_freq[raw_gate][np.nanargmax(persistence)]
    fd = np.mod(raw_freq - reference + fs_slow / 2, fs_slow) - fs_slow / 2
    order = np.argsort(fd)
    fd = fd[order]
    jf = np.abs(fd) <= 0.48 * fs_slow
    return {'DG': DG[order][jf], 'best_delay': best_delay[order][jf], 'noise': noise[order][jf],
            'competition': competition[order][jf], 'fd': fd[jf], 'tc': tc, 'reference_hz': reference,
            'nwin': nwin, 'nfft': nfft, 'fs_slow': fs_slow, 'starts': starts, 'order': order[jf]}


def cube_frame(R, K, start):
    """DK8OK buildCube, one frame: delay x Doppler power for the window starting at frame `start`."""
    window = hann_periodic(K['nwin'])
    correlation = R['rho_iq'][:, start:start + K['nwin']]                # NaN stays NaN (as MATLAB)
    spectrum = np.fft.fftshift(np.fft.fft(correlation * window, K['nfft'], axis=1), axes=1)
    return np.abs(spectrum[:, K['order']]) ** 2


# ---------------------------------------------------------------------------------------------- main
def main():
    path, centre_khz, usb_khz, start_text, seconds_text, out_prefix = sys.argv[1:7]
    started = time.time()
    start_unix = parse_start(start_text)
    if seconds_text == 'all':
        seconds = None
    else:
        seconds = float(seconds_text)

    if is_kiwi_wav(path):
        z, fs, t0, header_rate, gps_fresh_fraction = read_kiwi_interval(path, start_unix, seconds)
        print(f'KiwiSDR file: read {len(z) / fs:.0f} s; sample rate measured from GPS stamps {fs:.4f} S/s '
              f'(WAV header {header_rate}), first sample GPS {t0:.6f}')
    else:
        z, fs, t0, header_rate = read_plain_wav_interval(path, start_unix, seconds)
        gps_fresh_fraction = float('nan')
        print(f'plain IQ WAV: read {len(z) / fs:.0f} s at the header rate {fs:.0f} S/s, first sample '
              f'{t0:.6f} (from the file name; NaN if none)')

    R = residual_core(z, fs, float(centre_khz) * 1e3, float(usb_khz) * 1e3)
    K = prepare_spectrum(R)
    middle = K['starts'][len(K['starts']) // 2]                          # the delay x Doppler cube of the middle window
    cube = cube_frame(R, K, middle)

    np.savez_compressed(out_prefix + '.npz', t0_gps=t0, fs_measured=fs, fs_header=header_rate,
                        kiwi_gps_fresh_fraction=gps_fresh_fraction,
                        rho_iq=R['rho_iq'].astype(np.complex64), delay_ms=R['delay_ms'], t_frame=R['t_frame'],
                        sync_rms=R['sync_rms'], global_ppm=R['global_ppm'], residual_hz=R['residual_hz'],
                        psf_width_ms=R['psf_width_ms'], clock_obs_ids=R['clock_obs_ids'],
                        clock_obs_residual_samples=R['clock_obs_residual_samples'],
                        DG=K['DG'].astype(np.float32), best_delay=K['best_delay'].astype(np.float32),
                        noise=K['noise'].astype(np.float32), competition=K['competition'].astype(np.float32),
                        fd=K['fd'], tc=K['tc'], reference_hz=K['reference_hz'], cube=cube.astype(np.float32),
                        cube_tc=np.mean(R['t_frame'][middle:middle + K['nwin']]))
    print(f'done in {time.time() - started:.0f} s -> {out_prefix}.npz')


if __name__ == '__main__':
    main()
