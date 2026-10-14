"""
livepn_live -- live window for the STANAG 4285 PN80 monitor (based on DK8OK's STANAG4285 LIVE PN V1.4).

DK8OK's MATLAB program ((c) 2026 Nils Schiffhauer, DK8OK; Dopplergram group 2026-10-11; DK8OK credits Con
Wassilieff, ZL2AFP, and Peter Martinez, G3PLX, and wrote it with the help of ChatGPT) shows its measurements live
while audio arrives from a sound card.  This program does the same with the analyser in s4285_livepn.py, which has
his signal path and defaults.  The window follows his "Live PN / Equalizer" tab: audio spectrum, 60-s spectrogram,
constellations before and after the equalizer (last `hold` seconds), PN bit error rate before/after over the
`history` window, channel phase and carrier offset, and his status line.
Python version: Claude (AI assistant) for Mike Burns, AB1LD, 2026-10-12.

Sources (s4285_sources.py; no sound-card set-up needed)
  --kiwi HOST[:PORT] --dial KHZ   a live KiwiSDR, I/Q tuned to the station's USB dial frequency (through
                                  kiwirecorder.py's netcat mode), converted to USB audio as it arrives.
  --iq FILE --dial KHZ            an I/Q recording (plain 2-channel WAV, or a kiwirecorder file) converted to USB
                                  audio; --iq-centre KHZ gives the file's centre if its name does not.
  --replay FILE.wav               a USB-audio recording.
  Recordings play at real time (or --speed times faster), optionally --loop'ed; at each loop the stream jumps back,
  so a capture gap is flagged, as DK8OK's program does when its recorder is renewed every 60 s.

Usage
  python livepn/livepn_live.py --kiwi HOST:8073 --dial 8552.6 --user YOURCALL      (HOST: a public Kiwi, see README)
  python livepn/livepn_live.py --iq ../../data/2026-10-01_FUF_8552.6kHz/FUF_8552.6kHzUSB_IQcentre8554.4kHz_20261001T105500Z_12000Hz_iq.wav --iq-centre 8554.4 --dial 8552.6 --loop
  python livepn/livepn_live.py --replay usb_audio.wav --loop --speed 4 --cfo-guard --save results/
Close the window to stop; with --save the two CSV files (his columns) are written on closing.
--snapshot PNG runs --duration seconds of audio as fast as possible and saves one picture of the window (for tests).
"""
import argparse
import atexit
import datetime as dt
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s4285_livepn import LivePN, LOG_TIME, LOG_PN_BER, LOG_CFO, LOG_PHASE, LOG_CORRELATION, LOG_LOCKED
from s4285_livepn import EQ_TIME, EQ_BER_BEFORE, EQ_BER_AFTER
from livepn_offline import save_measurements, save_archive, ber_axis_top
from s4285_sources import ReplaySource, IqFileSource, KiwiSource

BLOCK_SECONDS = 0.2          # DK8OK's timer: audio arrives about every 0.2 s
REDRAW_SECONDS = 0.5         # window refresh interval (his paint() runs at most every 0.5 s)
SPECTROGRAM_SECONDS = 60     # his spectrogram shows the last 60 s
BLUE = (0, .35, .75)         # DK8OK's two colours
ORANGE = (.85, .45, 0)


# ---------------------------------------------------------------------------------------------- feeding
class Feeder:
    """Feeds the analyser with source blocks, paced by the wall clock (times `speed`)."""

    def __init__(self, source, analyser, speed=1.0):
        self.source = source
        self.analyser = analyser
        self.speed = speed
        self.block = int(round(BLOCK_SECONDS * source.rate))
        self.fed_samples = 0                    # audio samples delivered so far (continuous across loops)
        self.clock_start = time.monotonic()
        self.finished = False

    def feed_one_block(self):
        """Deliver one block; flag a capture gap when the replay wraps around."""
        block, wrapped = self.source.next_block(self.block)
        if wrapped:
            # The stream jumps back to the file start: like DK8OK's recorder renewal, a flagged gap.
            self.analyser.capture_gap(self.fed_seconds)
        if block.size == 0:
            # End of a recording, or a live stream that has ended (a live source otherwise just has nothing yet).
            if not self.source.realtime or getattr(self.source, 'ended', False):
                self.finished = True
            return
        self.analyser.feed(block)
        self.fed_samples += block.size

    @property
    def fed_seconds(self):
        """Audio seconds delivered so far (from a whole number of samples, so loop times are exact)."""
        return self.fed_samples / self.source.rate

    def catch_up(self):
        """Deliver the blocks due now: for a live source everything that has arrived; for a recording every
        block that would have arrived by now (real time x speed)."""
        if self.source.realtime:
            before = -1
            while not self.finished and self.fed_samples != before:
                before = self.fed_samples
                self.feed_one_block()
        else:
            due_seconds = (time.monotonic() - self.clock_start) * self.speed
            while not self.finished and self.fed_seconds + BLOCK_SECONDS <= due_seconds:
                self.feed_one_block()
        self._forget_old_spectra()

    def run_fast(self, seconds):
        """Deliver `seconds` of audio immediately (for --snapshot)."""
        while not self.finished and self.fed_seconds < seconds:
            before = self.fed_samples
            self.feed_one_block()
            if self.fed_samples == before and self.source.realtime:
                time.sleep(0.05)                        # a live source: wait for samples to arrive
        self._forget_old_spectra()

    def _forget_old_spectra(self):
        """Keep only the spectra shown in the spectrogram (one per block)."""
        keep = int(SPECTROGRAM_SECONDS / BLOCK_SECONDS)
        if len(self.analyser.spectra) > keep:
            del self.analyser.spectra[:-keep]


# ---------------------------------------------------------------------------------------------- window
def mean_of_finite(values):
    """Mean of the finite values (NaN marks frames without a value); NaN if there are none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return np.mean(finite)


class LiveWindow:
    """DK8OK's live tab, redrawn from the analyser's state."""

    def __init__(self, analyser, feeder, history, title):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.analyser = analyser
        self.feeder = feeder
        self.history = history

        self.figure = plt.figure(figsize=(14, 9.5))
        self.figure.suptitle(title, fontsize=10)
        grid = self.figure.add_gridspec(3, 4, height_ratios=[1, 1, 1])
        self.spectrum_axes = self.figure.add_subplot(grid[0, 0:2])
        self.spectrogram_axes = self.figure.add_subplot(grid[1, 0:2])
        self.before_axes = self.figure.add_subplot(grid[0:2, 2])
        self.after_axes = self.figure.add_subplot(grid[0:2, 3])
        self.ber_axes = self.figure.add_subplot(grid[2, 0:2])
        self.phase_axes = self.figure.add_subplot(grid[2, 2:4])
        self.cfo_axes = self.phase_axes.twinx()
        self.status_text = self.figure.text(0.01, 0.005, '', fontsize=9, family='monospace')
        self.figure.subplots_adjust(left=0.06, right=0.95, top=0.93, bottom=0.12, hspace=0.5, wspace=0.45)

    # -------------------------------------------------------------- panels
    def draw_spectrum(self):
        axes = self.spectrum_axes
        axes.clear()
        if self.analyser.spectra:
            moment, frequency, level_db = self.analyser.spectra[-1]
            axes.plot(frequency, level_db, color=BLUE, lw=1)
        axes.axvline(self.analyser.centre, color=ORANGE, ls='--', lw=1)
        axes.set_xlim(0, 3600)
        axes.set_ylim(-100, 0)
        axes.set_title('Live audio spectrum (last 0.25 s); dashed: audio centre', fontsize=9)
        axes.set_xlabel('Audio frequency (Hz)')
        axes.set_ylabel('dBFS / RBW')
        axes.grid(True)

    def draw_spectrogram(self):
        axes = self.spectrogram_axes
        axes.clear()
        spectra = self.analyser.spectra
        if len(spectra) > 1:
            times = np.array([entry[0] for entry in spectra])
            frequency = spectra[-1][1]
            levels = np.array([entry[2] for entry in spectra if entry[2].size == frequency.size]).T
            times = times[-levels.shape[1]:]
            axes.pcolormesh(times, frequency, levels, shading='nearest', vmin=-100, vmax=0, cmap='viridis')
        axes.set_ylim(0, 3600)
        axes.set_title(f'Audio spectrogram, last {SPECTROGRAM_SECONDS} s', fontsize=9)
        axes.set_xlabel('Elapsed time (s)')
        axes.set_ylabel('Audio frequency (Hz)')

    def draw_constellation(self, axes, cloud, colour, title):
        axes.clear()
        if cloud:
            points = np.concatenate([entry[1] for entry in cloud])
            axes.scatter(points.real, points.imag, s=6, color=colour, linewidths=0)
        axes.set_xlim(-2, 2)
        axes.set_ylim(-2, 2)
        axes.set_aspect('equal')
        axes.grid(True)
        axes.set_title(f'{title}, last {self.analyser.hold:g} s', fontsize=9)
        axes.set_xlabel('I / fitted amplitude')

    def draw_ber(self, now):
        axes = self.ber_axes
        axes.clear()
        rows = [row for row in self.analyser.eqdata if row[EQ_TIME] >= now - self.history]
        if rows:
            eq = np.array(rows)
            axes.plot(eq[:, EQ_TIME], 100 * eq[:, EQ_BER_BEFORE], '.', color=(.55, .65, .8), ms=3,
                      label='Before: frames')
            seconds = np.floor(eq[:, EQ_TIME])
            unique_seconds = np.unique(seconds)
            before_means = []
            after_means = []
            for second in unique_seconds:
                in_second = eq[seconds == second]
                before = in_second[:, EQ_BER_BEFORE]
                after = in_second[:, EQ_BER_AFTER]
                before_means.append(mean_of_finite(before))
                after_means.append(mean_of_finite(after))
            axes.plot(unique_seconds + .5, 100 * np.array(before_means), color=BLUE, lw=1.4, label='Before: 1 s')
            axes.plot(unique_seconds + .5, 100 * np.array(after_means), color=ORANGE, lw=1.4, label='After: 1 s')
            axes.legend(loc='upper left', fontsize=7)
        axes.set_xlim(max(0, now - self.history), max(10, now))
        if rows:
            top = ber_axis_top(100 * np.array(rows)[:, [EQ_BER_BEFORE, EQ_BER_AFTER]].ravel())
        else:
            top = 10
        axes.set_ylim(-0.3, top)                        # DK8OK's rule: 10 % at least, more if needed
        axes.set_title('PN BER before / after EQ: same symbols', fontsize=9)
        axes.set_xlabel('Elapsed time (s)')
        axes.set_ylabel('PN BER (%)')
        axes.grid(True)

    def draw_phase_and_cfo(self, now):
        axes = self.phase_axes
        right = self.cfo_axes
        axes.clear()
        right.clear()
        rows = [row for row in self.analyser.logdata if row[LOG_TIME] >= now - self.history]
        if rows:
            log = np.array(rows)
            locked = log[:, LOG_LOCKED] > 0
            if locked.any():
                axes.plot(log[locked, LOG_TIME], np.degrees(log[locked, LOG_PHASE]), '.', color=BLUE, ms=3)
                axes.set_ylim(-180, 180)
                axes.set_ylabel('Phase at PN start (deg)', color=BLUE)
            else:
                axes.plot(log[:, LOG_TIME], log[:, LOG_CORRELATION], '.-', color=BLUE)
                axes.set_ylim(0, 1)
                axes.set_ylabel('PN acquisition correlation', color=BLUE)
            right.plot(log[:, LOG_TIME], log[:, LOG_CFO], color=ORANGE, lw=1)
        # clear() puts a twin axis's ticks and label back on the left; move them to the right again
        right.yaxis.tick_right()
        right.yaxis.set_label_position('right')
        right.set_ylabel('CFO / candidate offset (Hz)', color=ORANGE)
        axes.set_xlim(max(0, now - self.history), max(10, now))
        axes.set_title('Channel phase (modulo 360 deg) and carrier offset', fontsize=9)
        axes.set_xlabel('Elapsed time (s)')
        axes.grid(True)

    def draw_status(self):
        source = self.feeder.source
        if source.loop:
            loop_text = f' | pass {source.passes + 1}'
        else:
            loop_text = ''
        if source.realtime:
            speed_text = ''
        else:
            speed_text = f' | speed x{self.feeder.speed:g}'
        if getattr(source, 'ended', False):
            end_text = ' | STREAM ENDED: ' + source.end_reason
        else:
            end_text = ''
        self.status_text.set_text(f'{self.analyser.status}\n{source.where_text()}{loop_text}{speed_text}{end_text}')

    def redraw(self):
        now = self.feeder.fed_seconds
        self.draw_spectrum()
        self.draw_spectrogram()
        self.draw_constellation(self.before_axes, self.analyser.cloud, BLUE, 'Before EQ (80 PN symbols)')
        self.draw_constellation(self.after_axes, self.analyser.eq_cloud, ORANGE, 'After EQ (inner symbols)')
        self.draw_ber(now)
        self.draw_phase_and_cfo(now)
        self.draw_status()

    # -------------------------------------------------------------- running
    def run_live(self):
        """Feed and redraw every REDRAW_SECONDS until the window is closed (or a non-looping replay ends)."""
        timer = self.figure.canvas.new_timer(interval=int(REDRAW_SECONDS * 1000))

        def tick():
            self.feeder.catch_up()
            self.redraw()
            self.figure.canvas.draw_idle()
            if self.feeder.finished:
                timer.stop()

        timer.add_callback(tick)
        timer.start()
        self.plt.show()

    def save_snapshot(self, path):
        self.redraw()
        self.figure.savefig(path, dpi=100)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--kiwi', help='live KiwiSDR: HOST or HOST:PORT (port 8073 if not given)')
    parser.add_argument('--iq', help='I/Q recording: plain 2-channel WAV or kiwirecorder file')
    parser.add_argument('--replay', help='USB audio WAV (4285 centred near 1800 Hz)')
    parser.add_argument('--dial', type=float, help='station USB dial frequency, kHz (for --kiwi and --iq)')
    parser.add_argument('--iq-centre', type=float, help='I/Q file centre frequency, kHz (if its name does not say)')
    parser.add_argument('--gain', type=int, default=50, help='Kiwi fixed gain, AGC off (default 50)')
    parser.add_argument('--kiwiclient', help='folder with kiwirecorder.py (default: kiwiclient next to the README)')
    parser.add_argument('--user', default='LIVE-PN', help="name shown in the Kiwi's user list (e.g. your call)")
    parser.add_argument('--allow-last-channel', action='store_true',
                        help="take a Kiwi's last free channel (only for your own Kiwi)")
    parser.add_argument('--loop', action='store_true', help='start again at the end of the file')
    parser.add_argument('--speed', type=float, default=1.0, help='replay speed (1 = real time)')
    parser.add_argument('--channel', choices=('left', 'right'), default='left')
    parser.add_argument('--centre', type=float, default=1800.0, help='audio centre (Hz)')
    parser.add_argument('--taps', type=int, default=9, choices=(5, 9, 13, 17))
    parser.add_argument('--ridge', type=float, default=0.02)
    parser.add_argument('--no-eq', action='store_true')
    parser.add_argument('--hold', type=float, default=3.0, help='seconds of constellation points shown (his Hold)')
    parser.add_argument('--history', type=float, default=300.0, help='seconds shown in the time plots (his History)')
    parser.add_argument('--cfo-guard', action='store_true', help="ours, off by default: cfo_method 'guarded'")
    parser.add_argument('--complete', action='store_true',
                        help='min interval 0.5 s: closes the blind gap between windows')
    parser.add_argument('--save', default=None, help='folder for the two CSV files, written when the window closes')
    parser.add_argument('--note', default='', help="text for the CSV 'Note' column")
    parser.add_argument('--snapshot', default=None, help='(tests) run --duration s of audio fast, save a PNG, exit')
    parser.add_argument('--duration', type=float, default=60.0)
    args = parser.parse_args(argv)

    if args.kiwi:
        if args.dial is None:
            raise SystemExit('--kiwi needs --dial (the station USB dial frequency in kHz)')
        host, _, port = args.kiwi.partition(':')
        source = KiwiSource(host, int(port or 8073), args.dial, args.gain, args.kiwiclient, args.user,
                            args.allow_last_channel)
    elif args.iq:
        if args.dial is None:
            raise SystemExit('--iq needs --dial (the station USB dial frequency in kHz)')
        source = IqFileSource(args.iq, args.dial, args.iq_centre, args.loop)
    elif args.replay:
        source = ReplaySource(args.replay, args.channel, args.loop)
    else:
        raise SystemExit('give a source: --kiwi HOST --dial KHZ, --iq FILE --dial KHZ, or --replay FILE.wav')
    # A live source runs kiwirecorder as a separate process: make sure it is stopped however we exit
    # (a snapshot run on 2026-10-12 left one running for five hours).
    atexit.register(source.close)
    # The source's audio centre (1800 Hz, scaled for a non-round sample rate), moved if --centre is given.
    centre = source.centre * args.centre / 1800.0
    if args.complete:
        min_interval = 0.5
    else:
        min_interval = 0.65
    if args.cfo_guard:
        cfo_method = 'guarded'
    else:
        cfo_method = 'dk8ok'
    analyser = LivePN(fs=source.rate, centre=centre, taps=args.taps, ridge=args.ridge,
                      eq_enable=not args.no_eq, hold=args.hold, min_interval=min_interval,
                      keep_spectra=True, cfo_method=cfo_method)
    feeder = Feeder(source, analyser, args.speed)

    if args.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    title = f'STANAG 4285 PN monitor (based on DK8OK LIVE PN V1.4) -- {source.name}'
    window = LiveWindow(analyser, feeder, args.history, title)

    # Run; whatever ends the run (window closed, stream ended, Ctrl-C, an error), stop the source and save what
    # was measured if --save was given.
    try:
        if args.snapshot:
            feeder.run_fast(args.duration)
            window.save_snapshot(args.snapshot)
            print('wrote ' + args.snapshot)
        else:
            window.run_live()
    finally:
        source.close()
        if args.save:
            save_results(analyser, source, args)
    print(analyser.status)


def save_results(analyser, source, args):
    """Write the two CSV files (his columns) to the --save folder.  A failure here is reported but does not hide
    whatever ended the run."""
    try:
        os.makedirs(args.save, exist_ok=True)
        stem = source.file_stem + '_live'
        measurements = os.path.join(args.save, stem + '_DK8OK_LIVE_PN.csv')
        archive = os.path.join(args.save, stem + '_DK8OK_LIVE_PN_ARCHIVE.csv')
        save_measurements(analyser, measurements, source.start_utc, args.note)
        save_archive(analyser, archive, source.start_utc, args.note, args.centre)
        print('saved ' + measurements)
        print('saved ' + archive)
    except Exception as error:
        print('could not save the CSV files: %s' % error)


if __name__ == '__main__':
    main()
