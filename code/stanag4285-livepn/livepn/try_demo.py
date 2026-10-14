"""
try_demo -- run DK8OK's built-in demo signal through the analyser and print how well it is measured.

The demo signal is DK8OK's (his demoAudio in LIVE PN V1.4): carrier offset +7 Hz, channel phase
0.8 + 0.4 sin(2 pi t / 15) rad.  The checks with pass/fail limits are in test_livepn.py; this script just prints.
Claude (AI assistant) for Mike Burns, AB1LD, 2026-10-12.

Usage: python try_demo.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s4285_reference import FB
from s4285_livepn import LivePN, LOG_TIME, LOG_PN_BER, LOG_CFO, LOG_PHASE, LOG_LOCKED, LOG_EVM
from s4285_livepn import EQ_BER_AFTER, EQ_READY
from s4285_testsignals import demo_audio, demo_truth, demo_pn_starts

FS = 48000
DURATION = 20          # seconds of demo audio


def make_demo_audio():
    """20 s of DK8OK's demo signal, generated in 0.2-s pieces as his program does."""
    rng = np.random.default_rng(0)
    pieces = []
    for first in range(0, DURATION * FS, 9600):
        last = min(first + 9600, DURATION * FS)
        pieces.append(demo_audio(first, last, rng=rng))
    return np.concatenate(pieces)


def main():
    audio = make_demo_audio()
    true_starts = demo_pn_starts(DURATION)

    for block_seconds in (0.2, 0.05):
        analyser = LivePN()
        block = int(block_seconds * FS)
        started = time.time()
        for first in range(0, audio.size, block):
            analyser.feed(audio[first:first + block])
        elapsed = time.time() - started

        log = np.array(analyser.logdata)
        eq = np.array(analyser.eqdata)
        locked = log[log[:, LOG_LOCKED] == 1]

        # Match each measured frame to the nearest true PN start.
        time_errors = []
        phase_errors = []
        for row in locked:
            differences = row[LOG_TIME] - true_starts
            nearest = np.argmin(np.abs(differences))
            time_errors.append(differences[nearest])
            true_offset, true_phase = demo_truth(true_starts[nearest])
            phase_errors.append(np.angle(np.exp(1j * (row[LOG_PHASE] - true_phase))))
        time_errors_samples = np.array(time_errors) * FB
        phase_errors_degrees = np.degrees(np.array(phase_errors))

        print('blocks of %.2f s: %.1f s CPU for %d s of audio' % (block_seconds, elapsed, DURATION))
        print('  rows %d, locked %d, true frames %d' % (len(log), len(locked), len(true_starts)))
        print('  PN BER mean %.4f' % np.mean(locked[:, LOG_PN_BER]))
        print('  CFO %.3f +/- %.3f Hz (truth 7)' % (np.mean(locked[:, LOG_CFO]), np.std(locked[:, LOG_CFO])))
        print('  frame time error %.3f +/- %.3f samples at 19.2 kHz (+1 expected: MATLAB 1-based index)'
              % (np.mean(time_errors_samples), np.std(time_errors_samples)))
        print('  phase error %.2f deg mean, %.2f deg rms'
              % (np.mean(phase_errors_degrees), np.sqrt(np.mean(phase_errors_degrees ** 2))))
        print('  EQ ready on %d frames, after-EQ BER %.4f, EVM %.3f'
              % (np.sum(eq[:, EQ_READY]), np.nanmean(eq[:, EQ_BER_AFTER]), np.mean(locked[:, LOG_EVM])))


if __name__ == '__main__':
    main()
