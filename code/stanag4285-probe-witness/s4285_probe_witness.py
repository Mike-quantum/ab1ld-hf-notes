#!/usr/bin/env python3
"""
STANAG 4285 "probe witness" test, standalone version for a plain IQ WAV file.

Question: is a feature in the delay profile (a bump, a shoulder, a second mode) really in the received signal, or is it
a sidelobe of the sync correlation? Test: estimate the received response twice, independently, by least squares
(which has no correlation sidelobes) -- once from the 80 sync symbols and once from the 48 probe symbols -- and use
each estimate to predict the OTHER set's received samples. A component is credible when it appears in both estimates
and when including it improves the prediction of symbols that were not used to fit it.

Credits
  * Test idea: Codex (OpenAI), 2026-10-06, in reply to Mike Burns' question on algorithm improvements.
  * Straight-line frame clock with a robust refit, and reading every frame at that clock: Nils Schiffhauer, DK8OK,
    DK8OK_STANAG4285_ESSENTIAL (2026), whose app credits Con Wassilieff, ZL2AFP, and Peter Martinez, G3PLX.
  * Code: Claude (Anthropic AI) for Mike Burns, AB1LD, 2026-10-07; same method as our probe_witness.py of 2026-10-06,
    rewritten to need nothing but this folder, numpy, scipy and matplotlib.

Model, per frame. Received sample k at 9600 S/s (4 samples per symbol, k = 0 at the first sync symbol, main-path
Doppler removed within the frame):
        r[k] = sum over taps m of h[m] * a[(k - m) / 4]
where a are the transmitted symbols (non-zero only at symbol instants) and the taps m run from -10 to +29
(-1.04 ... +3.0 ms). h is the whole received response: transmit pulse shaping, receiver filters and propagation
together. No pulse shape is assumed.

Unknown neighbouring symbols (the 128 data symbols per frame): they are NOT estimated, decided or treated as noise.
A sample k is used only if every symbol that any tap can reach from it, i.e. symbols (k - m) / 4 for m = -10 ... +29,
is known (sync or probe). Each usable sample is assigned to one witness -- "sync" if all those symbols are sync
symbols, "probe" if all are probe symbols; samples whose span mixes sync and probe symbols are dropped, so the two
witnesses share no samples. The same samples are used for every response model.
With taps -10 ... +29 this leaves 284 sync rows and 84 probe rows per frame. The probe blocks are short compared
with the tap span (40 taps = 10 symbols), so each 16-symbol block gives only 7 symbol positions x 4 samples = 28 rows.

Response models compared (tap sets): narrow -0.4 ... +0.4 ms, core -1.0 ... +1.0 ms, core + late (adds +1.5 ... +3.0
ms), full -1.04 ... +3.0 ms. For each frame and model: fit on the sync rows, predict the probe rows; and the reverse.
Reported: normalised held-out prediction error, median over frames, and the paired change against "core".

Usage: s4285_probe_witness.py IQ.wav STANAG_CENTRE_OFFSET_HZ OUT_PREFIX [FRAME_STEP]
  IQ.wav                 2-channel (I, Q) WAV, int16 or float, any integer sample rate
  STANAG_CENTRE_OFFSET_HZ  where the centre of the 4285 signal (USB carrier + 1800 Hz) lies in the IQ spectrum;
                         0 for the excerpt supplied (IQ centre 8554.4 kHz, USB carrier 8552.6 kHz)
  FRAME_STEP             use every n-th frame (default 1)
The input is converted to 9600 S/s with scipy's resample_poly. Runtime: about 25 s for the 10-minute excerpt.
"""
import sys
from fractions import Fraction
import numpy as np
from scipy import signal
from scipy.io import wavfile

import s4285_known_symbols

WORKING_RATE = 9600
SAMPLES_PER_SYMBOL = 4
SYMBOLS_PER_FRAME = 256
FRAME_SAMPLES = SAMPLES_PER_SYMBOL * SYMBOLS_PER_FRAME
TAPS = np.arange(-10, 30)
TAP_MS = TAPS / WORKING_RATE * 1000
PAD = 48
BLOCK = 1100
DOPPLER_SMOOTHING_FRAMES = 33


# ---------------------------------------------------------------- known symbols and the least-squares design rows

def known_symbols():
    """Complex value of every frame symbol, plus masks: known (sync or probe) and sync."""
    values, index, role = s4285_known_symbols.frame_template()
    known = role != 'data'
    is_sync = role == 'sync'
    symbols = np.where(known, values, 0)
    return symbols, known, is_sync


def design_rows(symbols, known, is_sync):
    """For every sample k whose whole tap span sees only known symbols of ONE kind: the transmitted symbol under each
    tap, the sample index k, and the witness the row belongs to ('sync' or 'probe')."""
    rows = []
    sample_index = []
    witness = []
    for k in range(-40, FRAME_SAMPLES - 24):
        row = np.zeros(len(TAPS), complex)
        usable = True
        kinds = set()
        for i, m in enumerate(TAPS):
            n = k - m
            if n % SAMPLES_PER_SYMBOL != 0:
                continue
            j = n // SAMPLES_PER_SYMBOL
            if j < 0 or j >= SYMBOLS_PER_FRAME or not known[j]:
                usable = False
                break
            row[i] = symbols[j]
            kinds.add('sync' if is_sync[j] else 'probe')
        if usable and len(kinds) == 1:
            rows.append(row)
            sample_index.append(k)
            witness.append(kinds.pop())
    return np.array(rows), np.array(sample_index), np.array(witness)


# ---------------------------------------------------------------- reading and channelizing

def read_iq(path):
    rate, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[1] != 2:
        sys.exit('expected a 2-channel (I, Q) WAV file')
    scale = 32768.0 if data.dtype == np.int16 else 1.0
    iq = (data[:, 0].astype(float) + 1j * data[:, 1].astype(float)) / scale
    return rate, iq


def channelize(iq, rate, centre_offset_hz):
    """Move the 4285 centre to 0 Hz, resample to 9600 S/s, and keep DK8OK's USB channel: 150-3600 Hz above the
    suppressed carrier, i.e. -1650 ... +1800 Hz around the 1800-Hz centre (complex band-pass FIR, 255 taps, delay
    compensated so the time axis is unchanged)."""
    n = np.arange(len(iq))
    mixed = iq * np.exp(-2j * np.pi * centre_offset_hz * n / rate)
    ratio = Fraction(WORKING_RATE, rate).limit_denominator(1000)
    z = signal.resample_poly(mixed, ratio.numerator, ratio.denominator)
    low_edge_hz = 150 - 1800
    high_edge_hz = 3600 - 1800
    half_width_hz = (high_edge_hz - low_edge_hz) / 2
    centre_hz = (high_edge_hz + low_edge_hz) / 2
    lowpass = signal.firwin(255, half_width_hz, fs=WORKING_RATE)
    taps = lowpass * np.exp(2j * np.pi * centre_hz * (np.arange(255) - 127) / WORKING_RATE)
    return signal.convolve(z, taps, mode='same')


# ---------------------------------------------------------------- frame clock (DK8OK's straight-line idea)

def sync_correlation(z, sync):
    """Complex correlation of z with the 80 sync symbols placed 4 samples apart; output index = first sync sample."""
    template = np.zeros(80 * SAMPLES_PER_SYMBOL - 3, complex)
    template[::SAMPLES_PER_SYMBOL] = sync
    return signal.correlate(z, template, mode='valid')


def parabolic_peak(values, i):
    """Fractional position of the maximum near integer index i (three-point parabola)."""
    if i <= 0 or i >= len(values) - 1:
        return float(i)
    left, centre, right = values[i - 1], values[i], values[i + 1]
    denominator = left - 2 * centre + right
    if denominator == 0:
        return float(i)
    return i + 0.5 * (left - right) / denominator


def fit_frame_clock(correlation):
    """Find each frame's sync peak and fit one straight line: frame number -> sample position.
    Frame phase from the correlation power folded on the 1024-sample frame; then, per frame, the strongest peak within
    +-8 samples of the expected position, refined by a parabola. Robust refit: drop peaks more than 3 robust sigma
    from the line, three times (DK8OK's global-clock approach)."""
    power = np.abs(correlation) ** 2
    frames_available = len(power) // FRAME_SAMPLES
    folded = power[:frames_available * FRAME_SAMPLES].reshape(frames_available, FRAME_SAMPLES).sum(axis=0)
    phase = int(np.argmax(folded))
    frame_numbers = []
    positions = []
    strengths = []
    for f in range(frames_available + 1):
        expected = phase + f * FRAME_SAMPLES
        low = expected - 8
        high = expected + 9
        if low < 1 or high >= len(power) - 1:
            continue
        i = low + int(np.argmax(power[low:high]))
        frame_numbers.append(f)
        positions.append(parabolic_peak(power, i))
        strengths.append(power[i])
    frame_numbers = np.array(frame_numbers, float)
    positions = np.array(positions)
    strengths = np.array(strengths)
    use = strengths > 0.25 * np.median(strengths)
    for iteration in range(3):
        slope, intercept = np.polyfit(frame_numbers[use], positions[use], 1)
        residual = positions - (intercept + slope * frame_numbers)
        spread = 1.4826 * np.median(np.abs(residual[use] - np.median(residual[use])))
        use = use & (np.abs(residual) < max(3 * spread, 0.05))
    slope, intercept = np.polyfit(frame_numbers[use], positions[use], 1)
    residual = positions - (intercept + slope * frame_numbers)
    return frame_numbers, intercept, slope, use, residual


def frame_block(z, position):
    """BLOCK samples with sample PAD exactly at fractional position `position` (FFT fractional shift)."""
    start = int(np.floor(position)) - PAD
    if start < 0 or start + BLOCK > len(z):
        return None
    piece = z[start:start + BLOCK]
    bins = np.fft.fftfreq(BLOCK) * BLOCK
    fraction = position - np.floor(position)
    return np.fft.ifft(np.fft.fft(piece) * np.exp(2j * np.pi * bins * fraction / BLOCK))


# ---------------------------------------------------------------- main

def main():
    wav_path = sys.argv[1]
    centre_offset_hz = float(sys.argv[2])
    out_prefix = sys.argv[3]
    frame_step = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    symbols, known, is_sync = known_symbols()
    rows, row_k, row_witness = design_rows(symbols, known, is_sync)
    sync_rows = row_witness == 'sync'
    probe_rows = row_witness == 'probe'
    print(f'design rows: sync {sync_rows.sum()}, probe {probe_rows.sum()}; taps {len(TAPS)} '
          f'({TAP_MS[0]:.2f} ... {TAP_MS[-1]:.2f} ms)')

    rate, iq = read_iq(wav_path)
    z = channelize(iq, rate, centre_offset_hz)
    print(f'{wav_path}: {len(iq)} samples at {rate} S/s -> {len(z)} at {WORKING_RATE} S/s ({len(z) / WORKING_RATE:.1f} s)')

    correlation = sync_correlation(z, symbols[:80])
    frame_numbers, intercept, slope, clock_used, clock_residual = fit_frame_clock(correlation)
    period_error_ppm = (slope / FRAME_SAMPLES - 1) * 1e6
    print(f'frame clock: {clock_used.sum()} of {len(frame_numbers)} frames in the fit; frame period '
          f'{slope:.4f} samples ({period_error_ppm:+.3f} ppm vs nominal 106.667 ms); rms residual '
          f'{np.sqrt(np.mean(clock_residual[clock_used] ** 2)):.3f} samples')

    # main-path phase of every frame at the clock, then Doppler (smoothed phase rate)
    blocks = []
    times = []
    main_phasor = []
    sync_template = symbols[:80]
    for f in frame_numbers:
        position = intercept + slope * f
        y = frame_block(z, position)
        if y is None:
            continue
        blocks.append(y)
        times.append(position / WORKING_RATE)
        main_phasor.append(np.sum(np.conj(sync_template) * y[PAD:PAD + 320:SAMPLES_PER_SYMBOL]))
    blocks = np.array(blocks)
    times = np.array(times)
    phase = np.unwrap(np.angle(np.array(main_phasor)))
    rate_rad_s = np.gradient(phase, times)
    kernel = np.ones(DOPPLER_SMOOTHING_FRAMES) / DOPPLER_SMOOTHING_FRAMES
    doppler_hz = np.convolve(rate_rad_s, kernel, mode='same') / (2 * np.pi)
    print(f'main-path Doppler (relative to the receiver tuning): median {np.median(doppler_hz):+.4f} Hz')

    models = {'narrow': (TAP_MS >= -0.4) & (TAP_MS <= 0.4),
              'core': (TAP_MS >= -1.0) & (TAP_MS <= 1.0),
              'core+late': ((TAP_MS >= -1.0) & (TAP_MS <= 1.0)) | ((TAP_MS >= 1.5) & (TAP_MS <= 3.0)),
              'full': np.ones(len(TAPS), bool)}
    directions = (('sync->probe', sync_rows, probe_rows), ('probe->sync', probe_rows, sync_rows))
    errors = {d: {name: [] for name in models} for d, _, _ in directions}
    h_sync_all = []
    h_probe_all = []
    k_relative = np.arange(BLOCK) - PAD
    for i in range(0, len(blocks), frame_step):
        derotated = blocks[i] * np.exp(-2j * np.pi * doppler_hz[i] * k_relative / WORKING_RATE)
        received = derotated[row_k + PAD]
        for direction, fit_rows, test_rows in directions:
            for name, taps in models.items():
                A = rows[fit_rows][:, taps]
                h, *_ = np.linalg.lstsq(A, received[fit_rows], rcond=None)
                predicted = rows[test_rows][:, taps] @ h
                error = np.sum(np.abs(received[test_rows] - predicted) ** 2) / np.sum(np.abs(received[test_rows]) ** 2)
                errors[direction][name].append(error)
        h_sync, *_ = np.linalg.lstsq(rows[sync_rows], received[sync_rows], rcond=None)
        h_probe, *_ = np.linalg.lstsq(rows[probe_rows], received[probe_rows], rcond=None)
        h_sync_all.append(h_sync)
        h_probe_all.append(h_probe)
    h_sync_all = np.array(h_sync_all)
    h_probe_all = np.array(h_probe_all)
    print(f'{len(h_sync_all)} frames analysed (every {frame_step})')

    lines = []
    for direction in errors:
        lines.append(f'{direction}: held-out prediction error, median over frames; paired change vs core')
        core = np.array(errors[direction]['core'])
        for name in models:
            e = np.array(errors[direction][name])
            change_db = 10 * np.log10(e / core)
            lines.append(f'  {name:10s} {10 * np.log10(np.median(e)):7.2f} dB   vs core {np.median(change_db):+6.2f} dB, '
                         f'better than core in {100 * np.mean(change_db < 0):5.1f} % of frames')
    report = '\n'.join(lines)
    print(report)

    sync_power = np.mean(np.abs(h_sync_all) ** 2, axis=0)
    probe_power = np.mean(np.abs(h_probe_all) ** 2, axis=0)
    correlation_profile = np.zeros(len(TAPS))
    for i, m in enumerate(TAPS):
        lag_positions = intercept + slope * frame_numbers + m
        integer = np.round(lag_positions).astype(int)
        valid = (integer >= 0) & (integer < len(correlation))
        correlation_profile[i] = np.mean(np.abs(correlation[integer[valid]]) ** 2)

    with open(out_prefix + '_profiles.csv', 'w') as f:
        f.write('tap,delay_ms,ls_from_sync_power,ls_from_probe_power,sync_correlation_power\n')
        for i in range(len(TAPS)):
            f.write(f'{TAPS[i]},{TAP_MS[i]:.4f},{sync_power[i]:.6e},{probe_power[i]:.6e},{correlation_profile[i]:.6e}\n')
    with open(out_prefix + '_summary.txt', 'w') as f:
        f.write(f'input {wav_path}, centre offset {centre_offset_hz} Hz, frame step {frame_step}\n')
        f.write(f'frame period error {period_error_ppm:+.3f} ppm; main-path Doppler median {np.median(doppler_hz):+.4f} Hz\n')
        f.write(f'frames analysed {len(h_sync_all)}; design rows sync {sync_rows.sum()}, probe {probe_rows.sum()}\n')
        f.write(report + '\n')
    np.savez_compressed(out_prefix + '.npz', tap=TAPS, delay_ms=TAP_MS, frame_time_s=times[::frame_step],
                        h_from_sync=h_sync_all.astype(np.complex64), h_from_probe=h_probe_all.astype(np.complex64),
                        doppler_hz=doppler_hz, **{f'error {d} {n}': np.array(errors[d][n]) for d in errors for n in models})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    ax = axes[0]
    ax.plot(TAP_MS, 10 * np.log10(sync_power / sync_power.max()), 'o-', ms=3, label='least squares from SYNC symbols')
    ax.plot(TAP_MS, 10 * np.log10(probe_power / probe_power.max()), 's-', ms=3, label='least squares from PROBE symbols')
    ax.plot(TAP_MS, 10 * np.log10(correlation_profile / correlation_profile.max()), '--', color='grey',
            label='plain sync correlation')
    ax.set_xlabel('delay relative to main path, ms')
    ax.set_ylabel('mean power, dB')
    ax.set_ylim(-35, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Received response from two independent known-symbol sets', fontsize=10)
    ax = axes[1]
    names = list(models)
    for offset, direction in ((-0.15, 'sync->probe'), (0.15, 'probe->sync')):
        medians = [10 * np.log10(np.median(errors[direction][n])) for n in names]
        fit_set, test_set = direction.split('->')
        ax.bar(np.arange(len(names)) + offset, medians, width=0.3, label=f'fit {fit_set}, predict {test_set}')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(['narrow\n+-0.4 ms', 'core\n+-1.0 ms', 'core + late\n+1.5...3.0 ms', 'full\n-1...+3 ms'])
    ax.set_ylabel('held-out prediction error, dB (lower = better)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)
    ax.set_title('Which response model predicts the OTHER symbols best?', fontsize=10)
    fig.suptitle(f'STANAG 4285 probe-witness test: {wav_path.split("/")[-1]}\n'
                 'Test idea: Codex (OpenAI). Frame clock: DK8OK straight-line method. Code: Claude for AB1LD, 2026-10-07.',
                 fontsize=9)
    plt.savefig(out_prefix + '.png', dpi=110)
    print('wrote ' + out_prefix + '.png, _profiles.csv, _summary.txt, .npz')


if __name__ == '__main__':
    main()
