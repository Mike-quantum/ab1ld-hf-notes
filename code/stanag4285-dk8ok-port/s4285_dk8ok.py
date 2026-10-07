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
  ported (P4), actual clock-fit residuals saved (M2). Pre-audit copy: s4285_dk8ok_pre_audit.py.
  2026-10-07: reads plain 2-channel IQ WAV files as well as KiwiSDR files (read_plain_wav_interval, format detected
  automatically), so that others can repeat the analysis on the published sample data; START may be a UTC time, a
  Unix time or 'start', SECONDS a number or 'all'. Measurement core unchanged. Also fixed: the Kiwi reader's
  straight-line fit to the GPS stamps is now made on times relative to the first sample (the fit to absolute Unix
  times lost precision: up to ~0.4 ms false departures and ~1 ppm error in the measured rate). Pre-change copy:
  s4285_dk8ok_pre_plainwav.py.
  For Kiwi files (also 2026-10-07): the WAV header rate is passed to kiwiwav, so Kiwis in other IQ modes (e.g. the
  20.25-kS/s wide mode common on public Kiwis) work; and the per-block 'last GPS solution' byte is checked, with a
  warning when the Kiwi had no recent GPS solution (saved as kiwi_gps_fresh_fraction).
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
import numpy as np
from math import gcd
from scipy import signal
from scipy.io import wavfile

# ---- constants exactly as in v37_residual_core (DK8OK) ----
RS = 2400
FSWORK = 9600
SPS = 4
FRAME_SAMP = 256 * SPS
FRAME_SEC = 256 / RS
RRC_BETA = 0.20
RRC_SPAN = 10
DELAY_MS = (-2, 4)
ACQ_SECONDS = 8
USB_AUDIO_HZ = 1800
TUNE_SEARCH_HZ = 100
TUNE_STEP_HZ = 2
PN31_BITS = '0101100111110001101110101000010'
# ---- prepareSpectrum / Essential UI defaults (DK8OK) ----
APERTURE_S = 64
DELAY_GATE_MS = (-0.25, 2.9)


def is_kiwi_wav(path):
    """[Ours, 2026-10-07] True for a kiwirecorder IQ file: right after the 'fmt ' chunk comes a 'kiwi' GPS-stamp chunk."""
    with open(path, 'rb') as f:
        f.seek(36)
        return f.read(4) == b'kiwi'


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
    if data.dtype == np.int16:
        scale = 32768.0
    elif data.dtype == np.int32:
        scale = 2147483648.0
    else:
        scale = 1.0
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
    if seconds is None:
        count = len(data) - first
    else:
        count = int(seconds * rate)
        if first + count > len(data):
            raise RuntimeError('recording is shorter than the requested interval')
    piece = np.asarray(data[first:first + count], dtype=float)
    z = (piece[:, 0] + 1j * piece[:, 1]) / scale
    if file_start is None:
        t0 = float('nan')
    else:
        t0 = file_start + first / rate
    return z, float(rate), t0, rate


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
            f.seek(KIWI_HEADER_BYTES + record * KIWI_RECORD_BYTES + 8)
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
    with open(path, 'rb') as f:
        f.seek(24)
        header_rate = struct.unpack('<I', f.read(4))[0]
    # the header rate, not kiwiwav's default (AB1LD's Kiwi, ~11999 S/s), so that Kiwis in other IQ modes work too; the
    # true rate is measured from the GPS stamps below
    wav = kiwiwav.KiwiWav(path, fs=float(header_rate))
    x, t, ok = wav.read(start_unix, seconds)
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
    keep = t >= start_unix
    x, t = x[keep], t[keep]
    n = np.arange(len(t))
    # fit times RELATIVE to the first sample: a straight-line fit to absolute Unix times (~1.8e9 s) over millions of
    # samples loses precision in np.polyfit (up to ~0.4 ms false departures and ~1 ppm rate error; found 2026-10-07)
    relative = t - t[0]
    line = np.polyfit(n, relative, 1)
    fs_measured = 1 / line[0]
    if np.max(np.abs(relative - np.polyval(line, n))) > 2 / fs_measured:
        raise RuntimeError('GPS stamps do not lie on one straight line within 2 samples')
    need = int(seconds * fs_measured)
    if len(x) < 0.99 * need:
        raise RuntimeError('recording is shorter than the requested interval')
    return x[:need].astype(complex), fs_measured, t[0] + line[1], header_rate, gps_fresh_fraction


def rcosdesign_sqrt(beta, span, sps):
    """MATLAB rcosdesign(beta, span, sps, 'sqrt'): root-raised-cosine, unit energy."""
    n = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.zeros(len(n))
    for i, t in enumerate(n):
        if abs(t) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * t) - 1) < 1e-12:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def pn80_reference():
    """DK8OK: b80 = [b31 b31 b31(1:18)], a80 = 1-2*b80, upsampled x4, RRC-shaped, centre 80*SPS samples, unit energy."""
    b31 = np.array([int(c) for c in PN31_BITS])
    b80 = np.concatenate([b31, b31, b31[:18]])
    a80 = 1 - 2 * b80
    rrc = rcosdesign_sqrt(RRC_BETA, RRC_SPAN, SPS)
    u = np.zeros(len(a80) * SPS)
    u[::SPS] = a80
    full = np.convolve(u, rrc)
    gd = RRC_SPAN * SPS // 2
    ref = full[gd:gd + 80 * SPS].astype(complex)
    return ref / np.sqrt(np.sum(np.abs(ref) ** 2))


def channelize(z, fs_in, delta_rf_hz):
    """DK8OK V2.9b channelizer: keep 150..3600 Hz above the suppressed carrier, put 1800 Hz at 0 Hz, resample to 9600."""
    z = z - np.mean(z)
    f_lo = 150.0
    f_hi = min(3600.0, 0.45 * fs_in)
    fc_ch = (f_lo + f_hi) / 2
    bw_ch = (f_hi - f_lo) / 2
    order = max(160, 2 * int(np.ceil(fs_in / 80)))
    order = min(order, 1600)
    if order % 2:
        order += 1
    if f_hi <= f_lo + 1000:                                              # DK8OK guard (restored after audit P3)
        raise ValueError('Input sample rate is too low for the STANAG USB channelizer.')
    if delta_rf_hz + f_lo <= -fs_in / 2 or delta_rf_hz + f_hi >= fs_in / 2:   # DK8OK guard (restored)
        raise ValueError('Selected STANAG USB information band lies outside the I/Q Nyquist range.')
    cutoff = min(0.98, bw_ch / (fs_in / 2))
    b = signal.firwin(order + 1, cutoff, window=('kaiser', 8.0))      # MATLAB fir1(ord, cutoff, 'low', kaiser(ord+1, 8))
    n0 = np.arange(len(z))
    z_tune = z * np.exp(-2j * np.pi * delta_rf_hz * n0 / fs_in)
    z_bb = z_tune * np.exp(-2j * np.pi * fc_ch * n0 / fs_in)
    z_bb = signal.oaconvolve(z_bb, b)[:len(z_bb)]                       # MATLAB fftfilt (causal FIR)
    gd = order // 2
    z_bb = np.concatenate([z_bb[gd:], np.zeros(gd, complex)])           # group-delay compensation
    z_sel = z_bb * np.exp(2j * np.pi * fc_ch * n0 / fs_in)
    z = z_sel * np.exp(-2j * np.pi * USB_AUDIO_HZ * n0 / fs_in)
    up, down = rational(FSWORK / fs_in)
    return signal.resample_poly(z, up, down)                            # MATLAB resample(z, p, q)


def rational(ratio):
    """p/q for the resampling ratio (MATLAB rat(ratio, 1e-12)); Kiwi: 9600/12000 = 4/5."""
    from fractions import Fraction
    f = Fraction(ratio).limit_denominator(100000)
    return f.numerator, f.denominator


def pn_match(z, ref):
    """DK8OK pnMatch: c[k] = ref' * z[k:k+L] for every k (one FFT convolution), e[k] = energy of z[k:k+L]."""
    L = len(ref)
    if len(z) < L:
        return np.zeros(0, complex), np.zeros(0)
    c = signal.fftconvolve(z, np.conj(ref[::-1]), mode='valid')
    power = np.abs(z) ** 2
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    e = cumulative[L:] - cumulative[:-L]
    return c, e[:len(c)]


def sample_correlation(matched, positions):
    """DK8OK sampleCorrelation: linear interpolation at fractional positions (0-based here), NaN outside."""
    positions = np.asarray(positions, dtype=float)
    i0 = np.floor(positions).astype(int)
    fraction = positions - i0
    valid = (i0 >= 0) & (i0 + 1 < len(matched))
    safe = np.where(valid, i0, 0)
    values = (1 - fraction) * matched[safe] + fraction * matched[safe + 1]
    values = values.astype(complex) if np.iscomplexobj(matched) else values.astype(float)
    values[~valid] = np.nan
    return values


def pn_acquisition_window(z, ref, fs, frame_n, grid):
    """DK8OK pnAcquisitionWindow: best tuning by the median-folded normalised correlation."""
    nt = np.arange(len(z))
    best = (-np.inf, 0, 0.0, 0.0)
    for df in grid:
        cc, ee = pn_match(z * np.exp(-2j * np.pi * df * nt / fs), ref)
        metric = np.abs(cc) / np.sqrt(np.maximum(ee, np.finfo(float).eps))
        nf = len(metric) // frame_n
        if nf < 30:
            continue
        fold = np.median(metric[:nf * frame_n].reshape(nf, frame_n), axis=0)
        ph = int(np.argmax(fold))
        pk = fold[ph]
        contrast = pk / (np.median(fold) + np.finfo(float).eps)
        if pk >= 0.12 and contrast >= 2.5 and pk > best[0]:
            best = (pk, ph, df, contrast)
    return best


def acquire_pn_across_interval(z, ref, fs, frame_n, window_sec, range_hz, step_hz):
    """DK8OK acquirePnAcrossInterval: overlapping 8-s windows; full fine grid in the first, coarse+fine later."""
    window_n = round(window_sec * fs)
    hop_n = round(window_n / 2)
    min_n = 30 * frame_n + len(ref)
    last_start = max(0, len(z) - window_n)
    starts = list(range(0, last_start + 1, hop_n))
    if starts[-1] != last_start:
        starts.append(last_start)
    for iw, first in enumerate(starts):
        zz = z[first:first + window_n]
        if len(zz) < min_n:
            continue
        if iw == 0:
            grid = np.arange(-range_hz, range_hz + step_hz / 2, step_hz)
        else:
            grid = np.arange(-range_hz, range_hz + 10, 20)
        pk, ph, df, contrast = pn_acquisition_window(zz, ref, fs, frame_n, grid)
        if pk >= 0.12 and contrast >= 2.5:
            if iw > 0:
                fine = np.arange(max(-range_hz, df - 12), min(range_hz, df + 12) + step_hz / 2, step_hz)
                pk, ph, df, contrast = pn_acquisition_window(zz, ref, fs, frame_n, fine)
            if pk >= 0.12 and contrast >= 2.5:
                print(f'PN80 acquired in window {first / fs:.1f} s | tune {df:+.1f} Hz | quality {pk:.3f} | contrast {contrast:.2f}')
                return first + ph, df, pk, first / fs
    raise RuntimeError('No reliable repeating PN80 found')


def robust_line(x, y, iterations=3):
    """DK8OK's clock fit: polyfit, then 3 passes keeping |err - median| <= max(2, 3 * MAD)."""
    p = np.polyfit(x, y, 1)
    for _ in range(iterations):
        err = y - np.polyval(p, x)
        mad = np.median(np.abs(err - np.median(err)))
        good = np.abs(err - np.median(err)) <= max(2, 3 * mad)
        if good.sum() < 30:
            raise RuntimeError('PN clock estimation is unstable')
        p = np.polyfit(x[good], y[good], 1)
    return p


def residual_core(z_in, fs_in, iq_centre_hz, usb_rf_hz):
    """DK8OK v37_residual_core without the PSF residual fit. Returns rhoIQ (delays x frames) and the clock."""
    z = channelize(z_in, fs_in, usb_rf_hz - iq_centre_hz)
    ref = pn80_reference()
    best_phase, residual_hz, best_score, acq_start = acquire_pn_across_interval(
        z, ref, FSWORK, FRAME_SAMP, ACQ_SECONDS, TUNE_SEARCH_HZ, TUNE_STEP_HZ)
    nn = np.arange(len(z))
    z = z * np.exp(-2j * np.pi * residual_hz * nn / FSWORK)
    lag_samp = np.arange(round(DELAY_MS[0] * 1e-3 * FSWORK), round(DELAY_MS[1] * 1e-3 * FSWORK) + 1)
    delay_ms = 1e3 * lag_samp / FSWORK
    n_frames = (len(z) - best_phase - len(ref) - lag_samp.max()) // FRAME_SAMP + 1
    if n_frames < 30:                                                    # DK8OK guard (restored after audit P3)
        raise RuntimeError('At least 30 complete PN frames are required.')
    block_n = 60 * FSWORK
    guard = FSWORK

    # pass 1: frame-by-frame peaks near the running clock prediction (adaptive origin/period, as DK8OK)
    origin = float(best_phase)
    period = float(FRAME_SAMP)
    frame_ids, peaks = [], []
    nominal = best_phase + np.arange(n_frames) * FRAME_SAMP
    for first in range(0, len(z), block_n):
        last = min(len(z), first + block_n) - 1
        lo = max(0, first - guard)
        hi = min(len(z), last + guard + 1)
        cc, ee = pn_match(z[lo:hi], ref)
        ix = np.flatnonzero((nominal >= first) & (nominal <= last))
        for k in ix:
            predicted = origin + k * period
            candidates = np.round(predicted).astype(int) + np.arange(-12, 13)
            local = candidates - lo
            ok = (local >= 0) & (local < len(cc))
            candidates, local = candidates[ok], local[ok]
            if len(local) == 0:
                continue
            metric = np.abs(cc[local]) / np.sqrt(np.maximum(ee[local], np.finfo(float).eps))
            im = int(np.argmax(metric))
            if metric[im] > max(0.12, 0.35 * best_score) and 0 < im < len(local) - 1:
                frame_ids.append(k)
                peaks.append(candidates[im])
        if len(frame_ids) >= 30 and len(ix) > 0:
            use = slice(max(0, len(frame_ids) - 1501), len(frame_ids))
            fx = np.array(frame_ids[use], float)
            fy = np.array(peaks[use], float)
            p = np.polyfit(fx, fy, 1)
            err = fy - np.polyval(p, fx)
            good = np.abs(err - np.median(err)) <= max(2, 3 * np.median(np.abs(err - np.median(err))))
            if good.sum() >= 20:
                p = np.polyfit(fx[good], fy[good], 1)
            if abs(p[0] / FRAME_SAMP - 1) < 200e-6:
                nxt = ix.max() + 1                       # MATLAB: next = max(ix), 1-based = next frame id
                old = origin + nxt * period
                target = np.polyval(p, nxt)
                period = p[0]
                origin = old + max(-1, min(1, target - old)) - nxt * period
    if len(frame_ids) < 30:
        raise RuntimeError('Too few reliable PN clock observations')

    # pass 2: one global straight-line clock for the whole interval
    p = robust_line(np.array(frame_ids, float), np.array(peaks, float))
    global_period = p[0]
    global_ppm = 1e6 * (global_period / FRAME_SAMP - 1)
    if abs(global_ppm) > 200:                                            # DK8OK guard (restored after audit P3)
        raise RuntimeError('PN clock drift exceeds 200 ppm.')
    clock_ids = np.array(frame_ids, float)
    clock_peaks = np.array(peaks, float)
    clock_residual = clock_peaks - np.polyval(p, clock_ids)              # actual fit residuals (audit M2), samples
    timing = np.polyval(p, np.arange(n_frames))
    rho_iq = np.full((len(lag_samp), n_frames), np.nan, complex)
    rho = np.full((len(lag_samp), n_frames), np.nan, complex)
    sync_rms = np.full(n_frames, np.nan)
    for first in range(0, len(z), block_n):
        last = min(len(z), first + block_n) - 1
        lo = max(0, first - guard)
        hi = min(len(z), last + guard + 1)
        cc, ee = pn_match(z[lo:hi], ref)
        ix = np.flatnonzero((timing >= first) & (timing <= last))
        if len(ix) == 0:
            continue
        positions = lag_samp[:, None] + timing[ix][None, :] - lo
        raw = sample_correlation(cc, positions)
        energy = sample_correlation(ee, positions)
        rho_iq[:, ix] = raw
        rho[:, ix] = raw / np.sqrt(np.maximum(energy, np.finfo(float).eps))
        sync_rms[ix] = np.sqrt(sample_correlation(ee, timing[ix] - lo) / len(ref))
    t_frame = timing / FSWORK                    # seconds from the first sample (DK8OK: (timingSamp-1)/FSWORK, 1-based)
    keep = np.isfinite(sync_rms)

    # -3 dB width of the reference's own autocorrelation (DK8OK psfWidth, used as an exclusion distance)
    psf = []
    for L in lag_samp:
        if L >= 0:
            a, b = ref[:len(ref) - L], ref[L:]
        else:
            a, b = ref[-L:], ref[:len(ref) + L]
        psf.append(abs(np.vdot(a, b)) / np.sqrt(np.sum(abs(a) ** 2) * np.sum(abs(b) ** 2)))
    psf = np.array(psf) / max(psf)
    above = np.flatnonzero(20 * np.log10(np.maximum(psf, 1e-4)) >= -3)
    psf_width = delay_ms[above[-1]] - delay_ms[above[0]]
    print(f'FFT PN80 global clock: {global_ppm:+.3f} ppm; residual tuning {residual_hz:+.1f} Hz; '
          f'{keep.sum()} frames; reference -3 dB width {psf_width:.3f} ms')
    return {'rho_iq': rho_iq[:, keep], 'rho': rho[:, keep], 'delay_ms': delay_ms, 't_frame': t_frame[keep],
            'sync_rms': sync_rms[keep], 'global_ppm': global_ppm, 'residual_hz': residual_hz, 'psf_width_ms': psf_width,
            'acquisition_start_s': acq_start, 'acquisition_score': best_score,
            'clock_obs_ids': clock_ids, 'clock_obs_residual_samples': clock_residual}


def prepare_spectrum(R, aperture_s=APERTURE_S):
    """DK8OK prepareSpectrum: time-Doppler image of the strongest delay cell per Doppler bin, 64-s Hann windows,
    with his noise-floor and competing-delay outputs (restored after audit P4). As in MATLAB, NaN cells stay NaN
    and are skipped by the maximum (MATLAB max omits NaN); nothing is zero-filled."""
    C = R['rho_iq']
    t = R['t_frame']
    d = R['delay_ms']
    fs_slow = 1 / np.median(np.diff(t))
    nwin = min(len(t), max(8, round(aperture_s * fs_slow)))
    nfft = max(128, 1 << int(np.ceil(np.log2(4 * nwin))))
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nwin) / nwin)          # MATLAB hann(nwin, 'periodic')
    starts = np.arange(0, len(t) - nwin + 1, max(1, round(nwin / 4)))
    raw_freq = np.arange(-nfft // 2, nfft // 2) * fs_slow / nfft
    raw_gate = np.abs(raw_freq) <= 0.48 * fs_slow
    delay_gate = (d >= DELAY_GATE_MS[0]) & (d <= DELAY_GATE_MS[1])
    gate_delay = d[delay_gate]
    exclusion = max(0.3, R['psf_width_ms']) if np.isfinite(R['psf_width_ms']) else 0.3
    DG = np.full((nfft, len(starts)), np.nan)
    best_delay = np.full((nfft, len(starts)), np.nan)
    noise = np.full((nfft, len(starts)), np.nan)
    competition = np.full((nfft, len(starts)), np.nan)
    tc = np.zeros(len(starts))
    Cg = C[delay_gate]
    columns = np.arange(nfft)
    for k, s in enumerate(starts):
        X = np.fft.fftshift(np.fft.fft(Cg[:, s:s + nwin] * w, nfft, axis=1), axes=1)
        P = np.abs(X) ** 2
        finite = np.isfinite(P)
        Pz = np.where(finite, P, -np.inf)
        ib = np.argmax(Pz, axis=0)
        peak = Pz[ib, columns]
        has = np.isfinite(peak)
        DG[has, k] = peak[has]
        best_delay[has, k] = gate_delay[ib[has]]
        noise_per_delay = np.nanmedian(np.where(finite[:, raw_gate], P[:, raw_gate], np.nan), axis=1) / np.log(2)
        noise[has, k] = noise_per_delay[ib[has]]
        eligible = np.abs(gate_delay[:, None] - gate_delay[ib][None, :]) >= exclusion
        alternative = np.where(eligible & finite, P, 0).max(axis=0)
        competition[has, k] = alternative[has] / np.maximum(peak[has], np.finfo(float).tiny)
        tc[k] = np.mean(t[s:s + nwin])
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
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(K['nwin']) / K['nwin'])
    C = R['rho_iq'][:, start:start + K['nwin']]                         # NaN stays NaN (as MATLAB)
    X = np.fft.fftshift(np.fft.fft(C * w, K['nfft'], axis=1), axes=1)
    return np.abs(X[:, K['order']]) ** 2


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
    middle = K['starts'][len(K['starts']) // 2]
    cube = cube_frame(R, K, middle)
    np.savez_compressed(out_prefix + '.npz', t0_gps=t0, fs_measured=fs, fs_header=header_rate,
                        kiwi_gps_fresh_fraction=gps_fresh_fraction,
                        rho_iq=R['rho_iq'].astype(np.complex64), delay_ms=R['delay_ms'], t_frame=R['t_frame'],
                        sync_rms=R['sync_rms'], global_ppm=R['global_ppm'], residual_hz=R['residual_hz'],
                        psf_width_ms=R['psf_width_ms'], clock_obs_ids=R['clock_obs_ids'],
                        clock_obs_residual_samples=R['clock_obs_residual_samples'],
                        DG=K['DG'].astype(np.float32), best_delay=K['best_delay'].astype(np.float32),
                        noise=K['noise'].astype(np.float32), competition=K['competition'].astype(np.float32), fd=K['fd'],
                        tc=K['tc'], reference_hz=K['reference_hz'], cube=cube.astype(np.float32),
                        cube_tc=np.mean(R['t_frame'][middle:middle + K['nwin']]))
    print(f'done in {time.time() - started:.0f} s -> {out_prefix}.npz')


if __name__ == '__main__':
    main()
