"""
iq_to_usb_audio -- turn an I/Q WAV (I left, Q right) into the real USB audio a receiver would give, at 48 kHz.

Ours (Claude for Mike Burns, AB1LD, 2026-10-12), no DK8OK code: a helper so that recordings made as I/Q (KiwiSDR,
RSPduo) can be fed to livepn_offline.py, which, like DK8OK's LIVE PN program, expects USB audio from a sound card.

What it does: the I/Q is shifted so that the USB suppressed carrier is at 0 Hz, everything outside
LOW..HIGH Hz above the carrier is removed with an FFT mask (no LSB, no image), the result is resampled to 48 kHz
and its real part is the audio.  Levels: scaled to -12 dBFS RMS.

Usage: python iq_to_usb_audio.py IQ.wav IQ_CENTRE_KHZ USB_CARRIER_KHZ OUT.wav [IQ2.wav ...]
       (extra I/Q files are appended in order -- for a recording cut into consecutive parts)
"""
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

AUDIO_RATE = 48000
LOW, HIGH = 100.0, 3900.0          # audio passband kept (Hz above the carrier)
RMS_DBFS = -12.0


def is_kiwi_file(path):
    """True for a GPS-stamped kiwirecorder file (a 'kiwi' chunk at byte 36), which scipy cannot read whole."""
    with open(path, 'rb') as handle:
        handle.seek(36)
        return handle.read(4) == b'kiwi'


def read_iq(paths):
    """Concatenate I/Q WAV files; returns complex samples (I + jQ) and the sample rate.
    Plain 2-channel WAV files are read with scipy; GPS-stamped kiwirecorder files with kiwiwav.py (through
    s4285_sources.read_iq_file, the same reader livepn_live.py --iq uses)."""
    pieces, rate = [], None
    for path in paths:
        if is_kiwi_file(path):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from s4285_sources import read_iq_file
            iq, this_rate = read_iq_file(path)
            this_rate = int(round(this_rate))
        else:
            this_rate, data = wavfile.read(path)
            if data.ndim != 2 or data.shape[1] != 2:
                raise SystemExit('expected a 2-channel (I, Q) WAV file: ' + path)
            data = data.astype(float)
            iq = data[:, 0] + 1j * data[:, 1]
        if rate is not None and this_rate != rate:
            raise SystemExit('sample rates differ between parts')
        rate = this_rate
        pieces.append(iq)
    return np.concatenate(pieces), int(rate)


def usb_audio(iq, rate, centre_hz, carrier_hz):
    """Real USB audio at AUDIO_RATE: carrier to 0 Hz, keep LOW..HIGH Hz, resample, take the real part."""
    n = np.arange(iq.size)
    shifted = iq * np.exp(-2j * np.pi * (carrier_hz - centre_hz) * n / rate)
    spectrum = np.fft.fft(shifted)
    frequency = np.fft.fftfreq(iq.size, 1 / rate)
    spectrum[(frequency < LOW) | (frequency > HIGH)] = 0
    baseband = np.fft.ifft(spectrum)
    divisor = np.gcd(AUDIO_RATE, rate)
    upsampled = signal.resample_poly(baseband, AUDIO_RATE // divisor, rate // divisor)
    audio = np.real(upsampled)
    target = 10 ** (RMS_DBFS / 20)
    rms = np.sqrt(np.mean(audio ** 2))
    if not rms > 0:
        raise SystemExit('the recording is silent in the USB band: check the centre and carrier frequencies')
    return audio * target / rms


def main(argv):
    if len(argv) < 4:
        raise SystemExit(__doc__)
    first, centre_khz, carrier_khz, out = argv[:4]
    iq, rate = read_iq([first] + list(argv[4:]))
    audio = usb_audio(iq, rate, float(centre_khz) * 1e3, float(carrier_khz) * 1e3)
    wavfile.write(out, AUDIO_RATE, np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16))
    print('wrote %s: %.1f s at %d Hz' % (out, audio.size / AUDIO_RATE, AUDIO_RATE))


if __name__ == '__main__':
    main(sys.argv[1:])
