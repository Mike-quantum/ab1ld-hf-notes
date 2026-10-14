"""
livepn_offline -- run the STANAG 4285 PN80 monitor over a WAV file: CSV files like DK8OK's, plus static plots.

Based on DK8OK's STANAG4285 LIVE PN V1.4 (MATLAB, (c) 2026 Nils Schiffhauer, DK8OK; Dopplergram group
2026-10-11; DK8OK credits Con Wassilieff, ZL2AFP, and Peter Martinez, G3PLX, and wrote it with the help of
ChatGPT).  The analysis is s4285_livepn.LivePN with his defaults.  The two CSV files have his columns and his
default file names (his "Save CSV" and "Save archive" buttons), prefixed with the recording's name; the plots are
static versions of his panels.
Python version: Claude (AI assistant) for Mike Burns, AB1LD, 2026-10.

Usage examples:
    python livepn_offline.py recording.wav
    python livepn_offline.py recording.wav --channel right --start-utc 2026-10-10T10:40:00 --note "FUF 8552"
    python livepn_offline.py recording.wav --complete --out results/

The WAV is fed to the analyser in exact 0.2-s blocks.  DK8OK's timer delivers audio about every 0.2 s plus the
time its own processing takes, so his live program cuts the audio a little differently; the scheduling thresholds
(analysis every 0.65 s or more, 0.85-s window, 2.5-s buffer) are his.  With exact 0.2-s blocks the analyses come
every 0.8 s, which can leave an 8-ms blind gap between analysis windows, so roughly one PN frame in 15 can go
unmeasured on some signals.  --complete shortens the minimum interval to 0.5 s, which closes that gap; nothing
else changes.  As in V1.4, the last fraction of a second of a file (after the last analysis) is not measured.  --cfo-guard (ours, off by default) keeps frames that DK8OK's per-frame
carrier-offset fit would throw away under strong multipath (see s4285_livepn.LivePN, cfo_method).
"""
import argparse
import csv
import datetime as dt
import os
import re
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s4285_reference import FB, SPS, AUDIO_CENTRE
from s4285_livepn import LivePN, LOG_COLUMNS, EQ_COLUMNS, ARCHIVE_COLUMNS, VERSION

SUPPORTED_RATES = (44100, 48000, 96000)     # the rates DK8OK's program offers
MARGIN = 240                                # samples at FB kept clear at each end of an analysis window


# ---------------------------------------------------------------------------------------------- input
def read_wav(path, channel='left'):
    """Read a WAV file as float audio in +/-1 and its sample rate; pick 'left' (or mono) or 'right'."""
    rate, data = wavfile.read(path)
    if data.dtype.kind == 'i':
        audio = data.astype(float) / float(np.iinfo(data.dtype).max + 1)
    elif data.dtype.kind == 'u':                                   # 8-bit WAV is unsigned
        audio = (data.astype(float) - 128.0) / 128.0
    else:
        audio = data.astype(float)
    if audio.ndim == 2:
        if channel == 'right':
            column = 1
        else:
            column = 0                                 # left channel, or the only one
        if column >= audio.shape[1]:
            raise SystemExit('the file has only %d channel(s)' % audio.shape[1])
        audio = audio[:, column]
    return audio, int(rate)


def to_supported_rate(audio, rate):
    """Keep 44.1/48/96 kHz (DK8OK's choices); resample anything else to 48 kHz."""
    if rate in SUPPORTED_RATES:
        return audio, rate
    divisor = np.gcd(48000, rate)
    audio = signal.resample_poly(audio, 48000 // divisor, rate // divisor)
    return audio, 48000


def utc_from_filename(path):
    """Start time from names like ..._20261010T104000Z... or 20261010_104000; None if there is none."""
    match = re.search(r'(20\d{6})[T_]?(\d{6})Z?', os.path.basename(path))
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
    except ValueError:                                 # digits that are not a date and time
        return None


def parse_utc(text):
    """ISO date-time (UTC) from the command line."""
    return dt.datetime.fromisoformat(text.replace('Z', ''))


# ---------------------------------------------------------------------------------------------- scheduling
def usable_span(window=0.85):
    """Seconds of PN-start positions that one analysis window can measure."""
    return window - (2 * MARGIN + 80 * SPS) / FB


def analysis_cadence(block_seconds, min_interval):
    """Audio time between analyses: min_interval rounded up to whole blocks."""
    return np.ceil(min_interval / block_seconds - 1e-9) * block_seconds


# ---------------------------------------------------------------------------------------------- run
def run(audio, rate, block_seconds, min_interval, **options):
    """Feed the whole file to a LivePN analyser in blocks; return the analyser."""
    analyser = LivePN(fs=rate, min_interval=min_interval, keep_spectra=True, keep_points=True, **options)
    block = int(round(block_seconds * rate))
    for first in range(0, audio.size, block):
        analyser.feed(audio[first:first + block])
    return analyser


# ---------------------------------------------------------------------------------------------- CSV output
def number(value):
    """Number as MATLAB writetable writes it (15 significant digits, NaN as NaN)."""
    if value is None:
        return ''
    if isinstance(value, float) and np.isnan(value):
        return 'NaN'
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return '%.15g' % value


def utc_text(start_utc, elapsed):
    """UTC column, DK8OK's format yyyy-MM-dd HH:mm:ss.SSS (empty if the start time is unknown)."""
    if start_utc is None or not np.isfinite(elapsed):
        return ''
    moment = start_utc + dt.timedelta(seconds=float(elapsed))
    with_microseconds = moment.strftime('%Y-%m-%d %H:%M:%S.%f')
    return with_microseconds[:-3]                     # keep milliseconds only


def save_measurements(analyser, path, start_utc, note):
    """DK8OK saveCSV: frame log joined with the EQ log, plus UTC and Note columns."""
    header = LOG_COLUMNS + EQ_COLUMNS[1:] + ['UTC', 'Note']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in analyser.joined_rows():
            values = [number(float(value)) for value in row]
            writer.writerow(values + [utc_text(start_utc, row[0]), note])


def save_archive(analyser, path, start_utc, note, centre):
    """DK8OK saveArchive: 1-second means, plus UTC, Note and AudioCentre_Hz columns."""
    header = ARCHIVE_COLUMNS + ['UTC', 'Note', 'AudioCentre_Hz']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in analyser.archive_values():
            values = [number(float(value)) for value in row]
            writer.writerow(values + [utc_text(start_utc, row[0]), note, number(float(centre))])


# ---------------------------------------------------------------------------------------------- plots
BLUE = (0, .35, .75)          # DK8OK's two colours
ORANGE = (.85, .45, 0)


def time_axis(times, start_utc):
    """Plot x values: UTC datetimes if the start is known, else elapsed seconds."""
    if start_utc is None:
        return np.asarray(times), 'Elapsed time (s)'
    moments = [start_utc + dt.timedelta(seconds=float(seconds)) for seconds in times]
    return moments, 'UTC'


def plot_overview(analyser, path, start_utc, title):
    """One page like his live tab: constellations before/after EQ, BER before/after, phase and CFO, spectrogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log = log_array(analyser)
    eq = eq_array(analyser)
    figure = plt.figure(figsize=(14, 11))
    grid = figure.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])
    figure.suptitle(title, fontsize=11)

    # Constellations of all measured frames (his live view shows only the last 'hold' seconds).
    # Each entry of analyser.points is (time, aligned PN symbols, equalizer output or None, received samples).
    before_eq = []
    after_eq = []
    for frame_time, aligned, equalized, received in analyser.points:
        before_eq.append(aligned)
        if equalized is not None:
            after_eq.append(equalized)
    plot_constellation(figure.add_subplot(grid[0, 0]), before_eq, BLUE, 'Before EQ (80 PN symbols / frame)')
    plot_constellation(figure.add_subplot(grid[0, 1]), after_eq, ORANGE, 'After EQ (inner symbols)')

    # Spectrogram from the 0.25-s spectra computed for each block (his paintAudio).
    axes = figure.add_subplot(grid[0, 2:])
    if analyser.spectra:
        times = np.array([spectrum[0] for spectrum in analyser.spectra])
        frequency = analyser.spectra[0][1]
        levels = np.array([spectrum[2] for spectrum in analyser.spectra]).T
        axes.pcolormesh(times, frequency, levels, shading='nearest', vmin=-100, vmax=0, cmap='viridis')
        axes.axhline(analyser.centre, color='w', lw=.6, ls='--')
    axes.set_ylim(0, 3600)
    axes.set_xlabel('Elapsed time (s)')
    axes.set_ylabel('Audio frequency (Hz)')
    axes.set_title('Audio spectrogram (dBFS / RBW, 0.25-s windows); dashed: audio centre', fontsize=9)

    # BER before / after EQ on the same inner symbols.
    axes = figure.add_subplot(grid[1, :])
    if len(eq):
        x, label = time_axis(eq[:, 0], start_utc)
        axes.plot(x, 100 * eq[:, 1], '.', color=(.55, .65, .8), ms=3, label='Before: frames')
        seconds = np.floor(eq[:, 0])
        unique_seconds = np.unique(seconds)
        means = np.full((unique_seconds.size, 2), np.nan)
        for j, second in enumerate(unique_seconds):
            for k in (0, 1):
                values = eq[seconds == second, k + 1]
                values = values[np.isfinite(values)]
                if values.size:
                    means[j, k] = np.mean(values)
        xs, _ = time_axis(unique_seconds + .5, start_utc)
        axes.plot(xs, 100 * means[:, 0], color=BLUE, lw=1.3, label='Before: 1 s')
        axes.plot(xs, 100 * means[:, 1], color=ORANGE, lw=1.3, label='After: 1 s')
        axes.set_xlabel(label)
        axes.set_ylim(-.3, ber_axis_top(100 * eq[:, 1:3]))
        axes.legend(loc='upper left', fontsize=8)
    axes.set_ylabel('PN BER (%)')
    axes.set_title('PN BER before / after EQ: same symbols', fontsize=9)
    axes.grid(True)

    # Channel phase at PN start and carrier offset (or, without lock, correlation).
    axes = figure.add_subplot(grid[2, :])
    if len(log):
        x, label = time_axis(log[:, 0], start_utc)
        locked = log[:, 6] > 0
        if locked.any():
            phase = np.where(locked, np.degrees(log[:, 3]), np.nan)
            axes.plot(x, phase, '.', color=BLUE, ms=3)
            axes.set_ylabel('Phase at PN start (deg)', color=BLUE)
            axes.set_ylim(-180, 180)
        else:
            axes.plot(x, log[:, 4], '.-', color=BLUE)
            axes.set_ylabel('PN acquisition correlation', color=BLUE)
            axes.set_ylim(0, 1)
        right = axes.twinx()
        right.plot(x, log[:, 2], color=ORANGE, lw=1)
        right.set_ylabel('CFO / candidate offset (Hz)', color=ORANGE)
        axes.set_xlabel(label)
    axes.set_title('Channel phase (modulo 360 deg) and residual carrier offset', fontsize=9)
    axes.grid(True)

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_constellation(axes, point_sets, colour, name):
    """Scatter all points of a list of complex arrays (one array per frame) on a fixed +/-2 square."""
    if point_sets:
        cloud = np.concatenate(point_sets)
        axes.scatter(cloud.real, cloud.imag, s=2, color=colour, alpha=.3, linewidths=0)
    axes.set_xlim(-2, 2)
    axes.set_ylim(-2, 2)
    axes.set_aspect('equal')
    axes.grid(True)
    axes.set_title('%s, %d frames' % (name, len(point_sets)), fontsize=9)
    axes.set_xlabel('I / fitted amplitude')
    axes.set_ylabel('Q / fitted amplitude')


def ber_axis_top(ber_percent):
    """DK8OK's rule for the top of the BER axis: at least 10 %, at most 100 %, else the largest value
    plus 1, rounded up to a multiple of 5."""
    finite = ber_percent[np.isfinite(ber_percent)]
    if finite.size == 0:
        return 10
    rounded_up = 5 * np.ceil((finite.max() + 1) / 5)
    return min(100, max(10, rounded_up))


def plot_archive(analyser, path, start_utc, title):
    """His archive tab: 1-second means of BER before/after, carrier offset, audio RMS and peak."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    archive = analyser.archive_values()
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    figure.suptitle(title, fontsize=11)
    if len(archive):
        x, label = time_axis(archive[:, 0], start_utc)
        axes[0].plot(x, archive[:, 1], color=BLUE, lw=1.2, label='Before EQ')
        axes[0].plot(x, archive[:, 2], color=ORANGE, lw=1.2, label='After EQ')
        axes[0].legend(loc='upper left', fontsize=8)
        axes[1].plot(x, archive[:, 3], color=BLUE, lw=1.2)
        axes[2].plot(x, archive[:, 4], color=BLUE, lw=1.2, label='RMS')
        axes[2].plot(x, archive[:, 5], color=ORANGE, lw=.8, label='Peak')
        axes[2].legend(loc='upper left', fontsize=8)
        axes[2].set_xlabel(label)
    axes[0].set_ylabel('PN BER (%)')
    axes[0].set_title('PN BER before / after EQ | 1-second means', fontsize=9)
    axes[1].set_ylabel('Carrier offset (Hz)')
    axes[1].set_title('Residual carrier offset relative to audio centre | locked frames only', fontsize=9)
    axes[2].set_ylabel('Audio level (dBFS)')
    axes[2].set_title('Audio RMS envelope and peak | before equalizer', fontsize=9)
    for panel in axes:
        panel.grid(True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_bits(analyser, path, title):
    """His PN bit comparison tab for the last measured frame: reference, received bits, EQ bits, errors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(13, 4))
    reference = (analyser.pn > 0).astype(int)
    symbol = np.arange(1, 81)
    axes.step(symbol, reference, where='post', color=BLUE, lw=1.5, label='Reference')
    last = analyser.last
    if last is not None:
        received = last['decoded']
        axes.plot(symbol, received, '.', color=ORANGE, label='Received')
        axes.plot(symbol, last['eq_bits'], 'o', mfc='none', color='k', ms=3, label='After EQ')
        bad = received != reference
        axes.plot(symbol[bad], received[bad], 'xk', ms=9, label='Error')
        title = '%s | frame at %.3f s: %d/80 errors' % (title, last['time'], last['err'])
    axes.set_xlim(1, 80)
    axes.set_ylim(-.2, 1.2)
    axes.set_xlabel('PN symbol')
    axes.set_ylabel('Bit')
    axes.set_title('PN80: reference (blue), received bits (orange) -- ' + title, fontsize=9)
    axes.legend(loc='upper right', fontsize=8, ncol=4)
    axes.grid(True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


# ---------------------------------------------------------------------------------------------- summary
def log_array(analyser):
    """The frame log as a 2-D array (one row per entry, LOG_COLUMNS), empty with 9 columns if there is none."""
    if analyser.logdata:
        return np.array(analyser.logdata)
    return np.zeros((0, len(LOG_COLUMNS)))


def eq_array(analyser):
    """The equalizer log as a 2-D array (EQ_COLUMNS), empty with 6 columns if there is none."""
    if analyser.eqdata:
        return np.array(analyser.eqdata)
    return np.zeros((0, len(EQ_COLUMNS)))


def summary(analyser, duration):
    """A few lines for the terminal."""
    log = log_array(analyser)
    eq = eq_array(analyser)
    locked = log[log[:, 6] > 0]
    expected_frames = duration * 2400 / 256          # 2400 symbols/s, 256 symbols per frame
    locked_percent = 100 * len(locked) / max(1, expected_frames)
    lines = ['%.1f s of audio, %d rows, %d locked frames (%.0f%% of %.0f expected)'
             % (duration, len(log), len(locked), locked_percent, expected_frames)]

    if len(locked):
        mean_ber = 100 * np.mean(locked[:, 1])
        median_cfo = np.median(locked[:, 2])
        upper_quartile, lower_quartile = np.percentile(locked[:, 2], [75, 25])
        interquartile_range = upper_quartile - lower_quartile
        median_evm = 100 * np.median(locked[:, 7])
        lines.append('PN BER mean %.2f%% | CFO median %.2f Hz (IQR %.2f) | EVM median %.1f%%'
                     % (mean_ber, median_cfo, interquartile_range, median_evm))

    ready = eq[eq[:, 5] == 1]
    if len(ready):
        lines.append('EQ on %d frames: BER %.2f%% -> %.2f%% (same inner symbols)'
                     % (len(ready), 100 * np.mean(ready[:, 1]), 100 * np.mean(ready[:, 2])))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wav', help='USB audio recording (WAV), STANAG 4285 centred near --centre')
    parser.add_argument('--channel', choices=('left', 'right'), default='left', help="'left' also means mono")
    parser.add_argument('--centre', type=float, default=AUDIO_CENTRE, help='audio centre in Hz (1800)')
    parser.add_argument('--taps', type=int, default=9, choices=(5, 9, 13, 17), help='EQ FIR taps (9)')
    parser.add_argument('--ridge', type=float, default=0.02, help='EQ ridge factor (0.02)')
    parser.add_argument('--no-eq', action='store_true', help='switch the equalizer off')
    parser.add_argument('--start-utc', help='UTC of the first sample, e.g. 2026-10-10T10:40:00 '
                                            '(default: from the file name if it carries a date-time)')
    parser.add_argument('--note', default='', help="text for the CSV 'Note' column")
    parser.add_argument('--block', type=float, default=0.2, help='seconds per block fed (his timer: 0.2)')
    parser.add_argument('--complete', action='store_true',
                        help='minimum interval 0.5 s instead of 0.65 s: closes the blind gap between windows')
    parser.add_argument('--cfo-guard', action='store_true',
                        help="ours, off by default: cfo_method 'guarded' -- when DK8OK's per-frame CFO fit lowers the "
                             "PN correlation, use the maximum-likelihood CFO instead (see s4285_livepn.LivePN)")
    parser.add_argument('--out', default=None, help='output folder (default: next to the WAV)')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if not args.block > 0:
        raise SystemExit('--block must be a positive number of seconds')

    audio, rate = read_wav(args.wav, args.channel)
    audio, rate = to_supported_rate(audio, rate)
    duration = audio.size / rate
    if args.start_utc:
        start_utc = parse_utc(args.start_utc)
    else:
        start_utc = utc_from_filename(args.wav)

    if args.complete:
        min_interval = 0.5
    else:
        min_interval = 0.65                          # DK8OK's value
    cadence = analysis_cadence(args.block, min_interval)
    if cadence > usable_span():
        print('note: analysis cadence %.3f s > usable span %.3f s -- some frames may go unmeasured '
              '(DK8OK V1.4 behaviour); use --complete to close the gap' % (cadence, usable_span()))

    if args.cfo_guard:
        cfo_method = 'guarded'
    else:
        cfo_method = 'dk8ok'
    analyser = run(audio, rate, args.block, min_interval, centre=args.centre, taps=args.taps,
                   ridge=args.ridge, eq_enable=not args.no_eq, cfo_method=cfo_method)

    # Output files go next to the WAV unless --out names a folder.
    if args.out:
        out = args.out
    else:
        out = os.path.dirname(os.path.abspath(args.wav))
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.wav))[0]
    measurements = os.path.join(out, stem + '_DK8OK_LIVE_PN.csv')
    archive = os.path.join(out, stem + '_DK8OK_LIVE_PN_ARCHIVE.csv')
    save_measurements(analyser, measurements, start_utc, args.note)
    save_archive(analyser, archive, start_utc, args.note, args.centre)
    written = [measurements, archive]

    if not args.no_plots:
        label = args.note or 'STANAG 4285'
        title = '%s | %s | centre %.1f Hz | %s' % (stem, label, args.centre, VERSION)
        overview_png = os.path.join(out, stem + '_overview.png')
        archive_png = os.path.join(out, stem + '_archive.png')
        bits_png = os.path.join(out, stem + '_bits.png')
        plot_overview(analyser, overview_png, start_utc, title)
        plot_archive(analyser, archive_png, start_utc, title)
        plot_bits(analyser, bits_png, stem)
        written += [overview_png, archive_png, bits_png]
    print(summary(analyser, duration))
    if args.cfo_guard:
        print('cfo guard: maximum-likelihood CFO used on %d frames' % analyser.cfo_fallbacks)
    print('status: ' + analyser.status)
    for path in written:
        print('wrote ' + path)


if __name__ == '__main__':
    main()
