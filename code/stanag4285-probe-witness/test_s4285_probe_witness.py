#!/usr/bin/env python3
"""
Known-answer tests for s4285_probe_witness.py: synthetic STANAG 4285 signals with paths we choose, written as plain
12000-S/s IQ WAVs, run through the script, and the recovered response compared with what was put in.

Written by Claude (Anthropic AI) for Mike Burns, AB1LD, 2026-10-07.

Synthetic signal: 256-symbol frames (sync + scrambled probes from s4285_known_symbols, random 8PSK data), 2400 Bd,
root-raised-cosine pulse (roll-off 0.2, span 10 symbols), 5 samples per symbol at 12000 S/s. The transmitter's frame
clock runs +0.5 ppm fast. Each path is a delayed (fractional, in the frequency domain), Doppler-shifted copy.
White complex noise is added.
Cases
  A  three paths: main 0 ms (-1.1 Hz), -0.5 ms at -12 dB (-0.94 Hz), +2.3 ms at -9 dB (-0.7 Hz); centre at 0 Hz.
  B  the same signal moved to +1000 Hz in the IQ spectrum (tests the centre-offset argument).
  C  control: the main path alone.
Checks
  * frame clock: the applied period error recovered within 0.05 ppm (the transmitter runs about +0.5 ppm fast, so
    frames arrive early; the exact value applied is printed, because the resampled length is rounded);
  * main-path Doppler -1.100 Hz recovered;
  * the mean least-squares response, from the sync AND separately from the probes, matches the response put in:
    sum over paths of gain^2 * |p(tau - delay - t0)|^2, p = the RRC pulse (the receive filters are flat across the
    signal band), with t0 (sub-sample clock alignment) and the overall scale fitted on case C only. Every tap within
    25 dB of the peak must agree within 1.5 dB;
  * the "core + late" model (adds +1.5 ... +3.0 ms) must beat "core" by more than 3 dB in at least 95 % of frames
    when a +2.3 ms path exists (A, B), and by less than 0.5 dB when it does not (C). (In C the pulse tail, which
    reaches +2.08 ms, still gives a small but consistent gain.) "full" vs "core" is NOT a test for late paths: the pulse itself has tails beyond
    +-1 ms (RRC, roll-off 0.2), so "full" beats "core" even for a single path.
"""
import subprocess
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

import s4285_known_symbols

RATE = 12000
BAUD = 2400
SPS = RATE // BAUD
SECONDS = 120
CLOCK_PPM = 0.5
SNR_DB = 20


def make_symbols(n_frames, rng):
    values, index, role = s4285_known_symbols.frame_template()
    frames = []
    for f in range(n_frames):
        frame = values.copy()
        data = role == 'data'
        frame[data] = np.exp(1j * np.pi / 4 * rng.integers(0, 8, data.sum()))
        frames.append(frame)
    return np.concatenate(frames)


def rrc(beta, span, sps):
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    h = np.zeros(len(t))
    for i, x in enumerate(t):
        if abs(x) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(x) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - beta)) + 4 * beta * x * np.cos(np.pi * x * (1 + beta))) / (np.pi * x * (1 - (4 * beta * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def transmitted_baseband(rng):
    """Transmitted signal at 12000 S/s; the transmitter clock is CLOCK_PPM fast (symbols slightly closer together)."""
    n_frames = int(SECONDS * BAUD / 256) + 2
    symbols = make_symbols(n_frames, rng)
    upsampled = np.zeros(len(symbols) * SPS, complex)
    upsampled[::SPS] = symbols
    shaped = signal.convolve(upsampled, rrc(0.2, 10, SPS), mode='same')
    # band-limited (FFT) resampling from the transmitter's clock to the receiver's: the same symbols take
    # 1 / (1 + CLOCK_PPM) as many receiver samples
    n_receiver = int(round(len(shaped) / (1 + CLOCK_PPM * 1e-6)))
    received_clock = signal.resample(shaped, n_receiver)
    applied_ppm = (len(shaped) / n_receiver - 1) * 1e6      # n_receiver is rounded, so this differs from CLOCK_PPM
    return received_clock[:int(SECONDS * RATE)], applied_ppm


def apply_path(x, delay_ms, doppler_hz, gain_db):
    bins = np.fft.fftfreq(len(x), 1 / RATE)
    delayed = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * bins * delay_ms / 1000))
    t = np.arange(len(x)) / RATE
    return 10 ** (gain_db / 20) * delayed * np.exp(2j * np.pi * doppler_hz * t)


def write_case(name, paths, centre_hz, rng, base):
    received = np.zeros(len(base), complex)
    for delay_ms, doppler_hz, gain_db in paths:
        received += apply_path(base, delay_ms, doppler_hz, gain_db)
    power = np.mean(np.abs(received) ** 2)
    received += np.sqrt(power / 10 ** (SNR_DB / 10) / 2) * (rng.standard_normal(len(base)) + 1j * rng.standard_normal(len(base)))
    received *= np.exp(2j * np.pi * centre_hz * np.arange(len(base)) / RATE)
    received *= 0.25 / np.max(np.abs(received))
    stereo = np.stack([np.round(received.real * 32767), np.round(received.imag * 32767)], axis=1).astype(np.int16)
    wavfile.write(f'test_{name}.wav', RATE, stereo)


def rrc_continuous(t_symbols, beta=0.2):
    """Root-raised-cosine impulse response at arbitrary times (in symbols)."""
    out = np.zeros(len(t_symbols))
    for i, x in enumerate(t_symbols):
        if abs(x) < 1e-9:
            out[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(x) - 1 / (4 * beta)) < 1e-9:
            out[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            out[i] = (np.sin(np.pi * x * (1 - beta)) + 4 * beta * x * np.cos(np.pi * x * (1 + beta))) / (np.pi * x * (1 - (4 * beta * x) ** 2))
    return out


def expected_power(delay_ms, paths, t0_ms):
    total = np.zeros(len(delay_ms))
    for path_delay_ms, doppler_hz, gain_db in paths:
        t_symbols = (delay_ms - path_delay_ms - t0_ms) * BAUD / 1000
        total += 10 ** (gain_db / 10) * rrc_continuous(t_symbols) ** 2
    return total


def run(name, centre_hz):
    output = subprocess.run([sys.executable, 's4285_probe_witness.py', f'test_{name}.wav', str(centre_hz), f'test_{name}'],
                            capture_output=True, text=True, check=True).stdout
    result = np.load(f'test_{name}.npz')
    clock_line = [line for line in output.splitlines() if line.startswith('frame clock')][0]
    ppm = float(clock_line.split('(')[1].split(' ppm')[0])
    return result, ppm


def median_change_db(result, direction, model):
    core = result[f'error {direction} core']
    other = result[f'error {direction} {model}']
    change = 10 * np.log10(other / core)
    return np.median(change), np.mean(change < 0)


def main():
    rng = np.random.default_rng(4285)
    base, applied_ppm = transmitted_baseband(rng)
    expected_ppm = -applied_ppm
    print(f'transmitter clock applied {applied_ppm:+.3f} ppm fast -> expect frame period {expected_ppm:+.3f} ppm')
    three_paths = [(0.0, -1.1, 0.0), (-0.5, -0.94, -12.0), (2.3, -0.7, -9.0)]
    one_path = [(0.0, -1.1, 0.0)]
    write_case('A', three_paths, 0, rng, base)
    write_case('B', three_paths, 1000, rng, base)
    write_case('C', one_path, 0, rng, base)
    results = {}
    for name, centre_hz in (('A', 0), ('B', 1000), ('C', 0)):
        results[name] = run(name, centre_hz)

    # sub-sample alignment t0 and scale from the single-path case C
    result_c = results['C'][0]
    delay = result_c['delay_ms']
    measured_c = np.mean(np.abs(result_c['h_from_sync']) ** 2, axis=0)
    best = None
    for t0_ms in np.linspace(-0.1, 0.1, 401):
        model = expected_power(delay, one_path, t0_ms)
        scale = np.sum(measured_c * model) / np.sum(model ** 2)
        mismatch = np.sum((measured_c - scale * model) ** 2)
        if best is None or mismatch < best[0]:
            best = (mismatch, t0_ms, scale)
    t0_ms = best[1]
    print(f'alignment from case C: t0 = {t0_ms * 1000:+.1f} us ({t0_ms * 9.6:+.2f} samples at 9600 S/s)')

    failures = []
    for name, paths in (('A', three_paths), ('B', three_paths), ('C', one_path)):
        result, ppm = results[name]
        doppler = np.median(result['doppler_hz'])
        shape = expected_power(delay, paths, t0_ms)
        print(f'case {name}: frame period {ppm:+.3f} ppm (expect {expected_ppm:+.3f}), main-path Doppler {doppler:+.4f} Hz (put in -1.1000)')
        for witness in ('h_from_sync', 'h_from_probe'):
            measured = np.mean(np.abs(result[witness]) ** 2, axis=0)
            depth_db = 25 if witness == 'h_from_sync' else 20
            significant = shape > shape.max() * 10 ** (-depth_db / 10)
            # each file is normalised to its own peak level, so the overall scale is fitted per case and witness
            scale = np.sum(measured[significant] * shape[significant]) / np.sum(shape[significant] ** 2)
            expected = scale * shape
            peak = expected.max()
            difference_db = 10 * np.log10(measured[significant] / expected[significant])
            worst = np.max(np.abs(difference_db))
            print(f'  {witness[7:]:5s}: {significant.sum()} taps within {depth_db} dB of the peak; worst difference from the put-in '
                  f'response {worst:.2f} dB')
            for target in (-0.5, 0.0, 2.3):
                i = int(np.argmin(np.abs(delay - target)))
                print(f'    {delay[i]:+.2f} ms: measured {10 * np.log10(measured[i] / peak):6.1f} dB, '
                      f'put in {10 * np.log10(expected[i] / peak):6.1f} dB')
            if worst > 1.5:
                failures.append(f'{name} {witness}: response')
        for direction in ('sync->probe', 'probe->sync'):
            late_change, late_better = median_change_db(result, direction, 'core+late')
            full_change, full_better = median_change_db(result, direction, 'full')
            print(f'  {direction}: core+late vs core {late_change:+.2f} dB (better in {100 * late_better:.0f} %), '
                  f'full vs core {full_change:+.2f} dB (better in {100 * full_better:.0f} %)')
            if name in ('A', 'B') and (late_change > -3 or late_better < 0.95):
                failures.append(f'{name} {direction}: late path not detected')
            if name == 'C' and late_change < -0.5:
                failures.append(f'C {direction}: late path claimed where there is none')
        if abs(ppm - expected_ppm) > 0.05:
            failures.append(f'{name}: clock')
        if abs(doppler + 1.1) > 0.005:
            failures.append(f'{name}: Doppler')
    result_c = results['C'][0]
    shape_c = expected_power(result_c['delay_ms'], one_path, t0_ms)
    empty = shape_c < shape_c.max() * 1e-4
    for witness in ('h_from_sync', 'h_from_probe'):
        measured = np.mean(np.abs(result_c[witness]) ** 2, axis=0)
        floor_db = 10 * np.log10(np.median(measured[empty]) / measured.max())
        print(f'estimation-noise floor in case C ({SNR_DB} dB SNR), {witness[7:]}: {floor_db:.1f} dB below the peak')
    print('ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
