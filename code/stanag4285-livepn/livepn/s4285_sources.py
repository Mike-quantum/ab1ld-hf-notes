"""
s4285_sources -- where the USB audio for the STANAG 4285 PN monitor comes from.

Claude (AI assistant) for Mike Burns, AB1LD, 2026-10-12.  No DK8OK code: DK8OK's LIVE PN V1.4 takes USB audio from a
sound card (virtual audio cable from the receiver software).  These sources give the analyser the same USB audio
without any audio set-up:
  * ReplaySource  -- a USB-audio WAV recording, played at real time (or faster), optionally looped.
  * IqFileSource  -- an I/Q recording (plain 2-channel WAV, or a KiwiSDR kiwirecorder file), converted to USB audio.
  * KiwiSource    -- a live KiwiSDR on the network, through kiwirecorder.py's netcat mode (kiwiclient, John Seamons,
                     jks-prv), I/Q tuned to the station's USB dial frequency, converted to USB audio as it arrives.

Tuning.  Everything is described by the station's USB dial frequency (the suppressed carrier; DK8OK's audio centre
1800 Hz lies 1.8 kHz above it).  For an I/Q file the program also needs the file's centre frequency: KiwiSDR file
names carry it (YYYYMMDDTHHMMSSZ_<centre in Hz>_...), otherwise give it.  The dial needs to be right only within
about +/-400 Hz: the analyser searches +/-450 Hz for the signal.

Conversion to USB audio (UsbConverter): shift the carrier to 0 Hz, keep 100..3900 Hz above it with a complex
band-pass FIR (255 taps, run with filter state so blocks join seamlessly), take the real part.  This is what a USB
demodulator does.  The FIR delays everything by 127 samples (~10.6 ms at 12 kS/s): frame times are late by that.

Sample rate.  The analyser resamples its input to 19200 S/s by a ratio of small integers, which needs a 'round' input
rate (e.g. 12000).  A Kiwi delivers about 11999 S/s.  The audio is therefore declared at the nearest multiple of 50
S/s (analysis_rate) and the analyser's audio centre is scaled by the same ratio (scaled_centre), so frequencies and
carrier offsets come out right; elapsed times are then stretched by that ratio (83 ppm for 11999 -> 12000:
0.3 s per hour), which only matters for the display.
"""
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import datetime as dt

import numpy as np
from scipy import signal
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_CENTRE = 1800.0             # DK8OK's audio centre above the USB carrier (Hz)
PASSBAND = (100.0, 3900.0)        # audio band kept above the carrier (Hz)
FIR_TAPS = 255
HEADER_TIMEOUT_S = 30             # a Kiwi that sends no samples within this time is given up (kiwirecorder would
                                  # otherwise reconnect every 5 s, silently, for ever)
SILENCE_TIMEOUT_S = 30            # a live stream that delivers nothing for this long is ended (normally, see below)


def utc_now():
    """The current UTC time as a naive datetime (the form used throughout these programs)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def safe_file_stem(text):
    """Text usable as a file name on Windows, macOS and Linux: letters, digits, '-' and '_' only."""
    return re.sub(r'[^A-Za-z0-9_-]+', '-', text).strip('-')


# ---------------------------------------------------------------------------------------------- rates
def analysis_rate(true_rate):
    """The 'round' rate the audio is declared at: the nearest multiple of 50 S/s (11999 -> 12000)."""
    return int(round(true_rate / 50.0) * 50)


def scaled_centre(true_rate, centre=AUDIO_CENTRE):
    """Audio centre to give the analyser when audio at true_rate is declared at analysis_rate(true_rate)."""
    return centre * analysis_rate(true_rate) / true_rate


def kiwi_file_centre_khz(path):
    """I/Q centre (kHz) from a kiwirecorder file name YYYYMMDDTHHMMSSZ_<Hz>_...; None if the name has none."""
    match = re.match(r'\d{8}T\d{6}Z_(\d+)_', os.path.basename(path))
    if not match:
        return None
    return int(match.group(1)) / 1000.0


def start_utc_from_name(path):
    """Start time (naive UTC datetime) from a YYYYMMDDTHHMMSSZ stamp in the file name; None if there is none."""
    match = re.search(r'(20\d{6})T(\d{6})Z', os.path.basename(path))
    if not match:
        return None
    return dt.datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')


# ---------------------------------------------------------------------------------------------- conversion
class UsbConverter:
    """Streaming I/Q -> USB audio.  Feed consecutive complex blocks; get real audio blocks of the same length."""

    def __init__(self, rate, carrier_offset_hz):
        """rate: I/Q sample rate; carrier_offset_hz: USB carrier minus I/Q centre frequency (0 if tuned to it)."""
        self.rate = rate
        self.carrier_offset_hz = carrier_offset_hz
        self.samples_done = 0                     # keeps the mixing oscillator continuous across blocks

        # Complex band-pass for PASSBAND above the carrier: a low-pass of half the band width, shifted up.
        band_centre = (PASSBAND[0] + PASSBAND[1]) / 2
        half_width = (PASSBAND[1] - PASSBAND[0]) / 2
        lowpass = signal.firwin(FIR_TAPS, half_width, fs=rate)
        tap_index = np.arange(FIR_TAPS)
        self.taps = lowpass * np.exp(2j * np.pi * band_centre * tap_index / rate)
        self.state = np.zeros(FIR_TAPS - 1, complex)

    def convert(self, iq_block):
        """One block of complex I/Q -> the same number of real USB-audio samples."""
        sample_number = self.samples_done + np.arange(iq_block.size)
        self.samples_done += iq_block.size
        carrier_at_zero = iq_block * np.exp(-2j * np.pi * self.carrier_offset_hz * sample_number / self.rate)
        band, self.state = signal.lfilter(self.taps, 1.0, carrier_at_zero, zi=self.state)
        return np.real(band)


# ---------------------------------------------------------------------------------------------- recordings
class ReplaySource:
    """A USB-audio WAV recording delivered block by block, as if it were arriving from a receiver."""

    realtime = False          # the feeder paces it by the wall clock

    def __init__(self, path, channel='left', loop=False):
        from livepn_offline import read_wav, to_supported_rate
        audio, rate = read_wav(path, channel)
        self.audio, self.rate = to_supported_rate(audio, rate)
        self.centre = AUDIO_CENTRE
        self.loop = loop
        self.position = 0
        self.passes = 0
        self.start_utc = start_utc_from_name(path)
        self.name = os.path.basename(path)
        self.file_stem = safe_file_stem(os.path.splitext(self.name)[0])

    def next_block(self, samples):
        """Up to `samples` new samples; (block, wrapped).  At the end: wrap around if looping, else empty."""
        if self.position >= self.audio.size:
            if not self.loop:
                return np.zeros(0), False
            self.position = 0
            self.passes += 1
            wrapped = True
        else:
            wrapped = False
        block = self.audio[self.position:self.position + samples]
        self.position += block.size
        return block, wrapped

    def where_text(self):
        """Where in the recording we are (UTC if the file name gives the start)."""
        seconds = self.position / self.rate
        if self.start_utc is None:
            return f'{seconds:6.1f} s into {self.name}'
        moment = self.start_utc + dt.timedelta(seconds=seconds)
        return f'{moment:%Y-%m-%d %H:%M:%S} UTC in {self.name}'

    def close(self):
        pass


class IqFileSource(ReplaySource):
    """An I/Q recording converted to USB audio, then delivered like ReplaySource.

    Plain 2-channel WAV files are read with scipy; KiwiSDR kiwirecorder files (GPS-stamped) with kiwiwav.py, at the
    WAV header rate.  dial_khz: the station's USB dial frequency; centre_khz: the file's I/Q centre (taken from a
    kiwirecorder file name if not given).
    """

    def __init__(self, path, dial_khz, centre_khz=None, loop=False):
        if centre_khz is None:
            centre_khz = kiwi_file_centre_khz(path)
        if centre_khz is None:
            raise SystemExit('give the I/Q centre frequency (--iq-centre) for ' + os.path.basename(path))
        iq, true_rate = read_iq_file(path)
        converter = UsbConverter(true_rate, (dial_khz - centre_khz) * 1e3)
        audio = converter.convert(iq)
        audio = audio * (0.25 / max(np.sqrt(np.mean(audio ** 2)), 1e-12))     # -12 dBFS RMS

        self.audio = audio
        self.rate = analysis_rate(true_rate)
        self.centre = scaled_centre(true_rate)
        self.true_rate = true_rate
        self.loop = loop
        self.position = 0
        self.passes = 0
        self.start_utc = start_utc_from_name(path)
        self.name = os.path.basename(path)
        self.file_stem = safe_file_stem(os.path.splitext(self.name)[0])


def read_iq_file(path):
    """Complex samples (full scale 1) and sample rate of a plain or KiwiSDR I/Q WAV file."""
    with open(path, 'rb') as handle:
        handle.seek(36)
        is_kiwi = handle.read(4) == b'kiwi'
        handle.seek(24)
        header_rate = struct.unpack('<I', handle.read(4))[0]
    if is_kiwi:
        sys.path.insert(0, HERE)                                     # a kiwiwav.py next to this file, if any
        sys.path.append(os.path.join(HERE, '..', '..', 'kiwi-tools'))   # ab1ld-hf-notes: code/kiwi-tools/kiwiwav.py
        import kiwiwav
        recording = kiwiwav.KiwiWav(path, fs=float(header_rate))
        try:
            if recording.nb < 3:
                raise SystemExit('the KiwiSDR recording is too short to read: ' + os.path.basename(path))
            # Record 0 carries no usable GPS stamp (kiwirecorder), so reading starts at record 1 (about 43 ms in).
            first_time = recording.stamp(1)
            duration = (recording.nb - 2) * kiwiwav.NS / header_rate
            x, t, ok = recording.read(first_time, duration)
        finally:
            handle = getattr(recording, 'f', None)
            if handle is not None:
                handle.close()
        if not ok:
            print('note: the recording has a gap or a bad GPS stamp; the analyser sees the audio joined across it, '
                  'so frame times after it are early by the length of the gap')
        return x.astype(complex), float(header_rate)
    rate, data = wavfile.read(path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise SystemExit('expected a 2-channel (I, Q) WAV file: ' + path)
    if data.dtype == np.int16:
        scale = 32768.0
    elif data.dtype == np.int32:
        scale = 2147483648.0
    elif data.dtype in (np.float32, np.float64):
        scale = 1.0
    else:
        raise SystemExit(f'unsupported I/Q sample type {data.dtype} (16-bit, 32-bit or float WAV please): ' + path)
    in_phase = data[:, 0].astype(float)
    quadrature = data[:, 1].astype(float)
    return (in_phase + 1j * quadrature) / scale, float(rate)


# ---------------------------------------------------------------------------------------------- live KiwiSDR
def kiwi_status(host, port, timeout=8):
    """A Kiwi's /status page as a dict of strings; None if it does not answer."""
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/status', timeout=timeout) as response:
            text = response.read().decode('utf-8', errors='replace')
    except Exception:
        return None
    status = {}
    for line in text.splitlines():
        key, separator, value = line.partition('=')
        if separator:
            status[key.strip()] = value.strip()
    return status


def check_kiwi_before_connecting(host, port, allow_last_channel=False):
    """Refuse at once, with a plain reason, when the Kiwi will not serve us: down, offline, scripts not allowed by
    its owner (ext_api=0), or no free channel.  Out of courtesy to other listeners we also do not take a public
    Kiwi's last free channel, unless allow_last_channel is set (e.g. for your own Kiwi).
    Returns a warning text ('' if none), e.g. for time limits."""
    status = kiwi_status(host, port)
    if status is None:
        raise SystemExit(f'{host}:{port} does not answer (down, unreachable, or wrong port; proxy Kiwis use 8073)')
    if status.get('offline', 'no') != 'no' or status.get('status', 'active') != 'active':
        raise SystemExit(f'{host}:{port} is offline / not accepting users')
    if status.get('ext_api', '').strip() == '0':
        raise SystemExit(f'{host}:{port}: its owner allows no scripted connections (ext_api=0); use a browser there')
    users = int(status.get('users', '0') or 0)
    users_max = int(status.get('users_max', '0') or 0)
    if not users_max and not allow_last_channel:
        raise SystemExit(f'{host}:{port} does not say how many channels it has, so we cannot tell whether one is '
                         f'free for us (--allow-last-channel if this is your own Kiwi)')
    if users_max and users >= users_max:
        raise SystemExit(f'{host}:{port} has all {users_max} channels in use; try later or another Kiwi '
                         f'(your own Kiwi: is another program of yours using it?)')
    if users_max and users_max - users < 2 and not allow_last_channel:
        raise SystemExit(f'{host}:{port} has only one free channel left; leaving it for a listener '
                         f'(--allow-last-channel if this is your own Kiwi)')
    if 'Limits' in status.get('sdr_hw', ''):
        return 'note: this Kiwi has time limits set by its owner; the session may be cut off'
    return ''


class KiwiSource:
    """A live KiwiSDR: I/Q tuned to the USB dial frequency, streamed by kiwirecorder.py --nc --nc-wav, converted
    to USB audio as it arrives.  Blocks are delivered as they come; there is no wall-clock pacing to do."""

    realtime = True

    def __init__(self, host, port, dial_khz, gain=50, kiwiclient_dir=None, user='LIVE-PN',
                 allow_last_channel=False):
        if kiwiclient_dir is None:
            # Where to look, in order: the folder above this one (README: git clone kiwiclient there), then this
            # folder.  If neither has it, the error names the first.
            candidates = [os.path.join(os.path.dirname(HERE), 'kiwiclient'),
                          os.path.join(HERE, 'kiwiclient')]
            kiwiclient_dir = candidates[0]
            for candidate in candidates:
                if os.path.exists(os.path.join(candidate, 'kiwirecorder.py')):
                    kiwiclient_dir = candidate
                    break
        kiwiclient_dir = os.path.abspath(os.path.expanduser(kiwiclient_dir))   # used both as cwd and in the path
        recorder = os.path.join(kiwiclient_dir, 'kiwirecorder.py')
        if not os.path.exists(recorder):
            raise SystemExit('kiwirecorder.py not found in ' + kiwiclient_dir + ' -- see README "Live KiwiSDR" '
                             '(git clone https://github.com/jks-prv/kiwiclient), or give --kiwiclient DIR')
        warning = check_kiwi_before_connecting(host, port, allow_last_channel)
        if warning:
            print(warning)
        self.name = f'KiwiSDR {host}:{port} at {dial_khz:.2f} kHz USB'
        # For saved files: no ':' (not allowed in Windows file names), the frequency kept whole, and the start time.
        frequency_text = f'{dial_khz:.2f}'.replace('.', 'p')
        self.file_stem = safe_file_stem(f'kiwi_{host}_{port}_{frequency_text}kHz_{utc_now():%Y%m%dT%H%M%SZ}')
        self.dial_khz = dial_khz
        self.loop = False
        self.passes = 0
        self.ended = False                        # set when the stream has ended and everything has been delivered
        self.end_reason = ''
        self.start_utc = None                     # set when the first samples arrive
        self.process = None
        self.samples = queue.Queue()
        self.pending = np.zeros(0)
        self.error_lines = []
        # I/Q tuned to the dial: the USB band is then 0..+3.9 kHz in the I/Q spectrum (offset 0).
        command = [sys.executable, '-u', recorder,
                   '-s', host,                         # the Kiwi
                   '-p', str(port),
                   '-f', f'{dial_khz:.3f}',            # tuned to the USB dial frequency (kHz)
                   '-m', 'iq',                         # I/Q mode
                   '-u', user,                         # name shown in the Kiwi's user list
                   '-L', '0', '-H', '4000',            # passband 0 .. 4000 Hz above the dial
                   f'--agc-gain={gain}',               # AGC off, fixed gain
                   '--nc', '--nc-wav',                 # stream the samples to us, with a WAV header first
                   '-q']                               # quiet
        self.process = subprocess.Popen(command, cwd=kiwiclient_dir, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        # From here on the recorder is running on someone's receiver: whatever goes wrong (including Ctrl-C while
        # waiting for the first samples), stop it before passing the error on.
        try:
            threading.Thread(target=self._error_reader, daemon=True).start()
            self.true_rate = self._read_header()
            self.start_utc = utc_now()
            self.rate = analysis_rate(self.true_rate)
            self.centre = scaled_centre(self.true_rate)
            self.converter = UsbConverter(self.true_rate, 0.0)
            self.received = 0
            self.last_arrival = time.monotonic()
            threading.Thread(target=self._reader, daemon=True).start()
        except BaseException:
            self.close()
            raise

    def _read_header(self):
        """The WAV header kiwirecorder writes first (--nc-wav): 44 bytes, sample rate at bytes 24..27.
        Gives up after HEADER_TIMEOUT_S (kiwirecorder writes it only when the first samples arrive).
        Checks that it is a 2-channel, 16-bit WAV header with a sensible sample rate."""
        result = {}

        def read():
            result['header'] = self.process.stdout.read(44)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(HEADER_TIMEOUT_S)
        header = result.get('header', b'')
        if len(header) < 44:
            message = ' | '.join(self.error_lines[-3:])
            raise SystemExit(f'the KiwiSDR sent no samples within {HEADER_TIMEOUT_S} s (busy, refusing scripted '
                             f'clients, or unreachable). ' + message[-300:])
        # The fields of a standard 44-byte WAV header that matter here.
        channels = struct.unpack('<H', header[22:24])[0]
        rate = struct.unpack('<I', header[24:28])[0]
        bits = struct.unpack('<H', header[34:36])[0]
        advice = '; is the kiwiclient version a current one?'
        if header[0:4] != b'RIFF' or header[8:12] != b'WAVE':
            raise SystemExit('kiwirecorder did not start its stream with a WAV header' + advice)
        if channels != 2:
            raise SystemExit(f'kiwirecorder sent {channels} channel(s), not 2 (I and Q)' + advice)
        if bits != 16:
            raise SystemExit(f'kiwirecorder sent {bits}-bit samples, not 16-bit' + advice)
        if rate < 1000 or rate > 100000:
            raise SystemExit(f'kiwirecorder sent an implausible sample rate, {rate} S/s' + advice)
        return float(rate)

    def _reader(self):
        """Background thread: raw int16 I, Q pairs from kiwirecorder -> complex blocks in the queue."""
        while True:
            try:
                data = self.process.stdout.read(4096)      # 1024 I/Q pairs
            except (OSError, ValueError):                  # the pipe was closed by close()
                data = b''
            if not data:
                self.samples.put(None)                     # end of stream
                return
            pairs = np.frombuffer(data[:len(data) // 4 * 4], '<i2').astype(float).reshape(-1, 2) / 32768
            self.samples.put(pairs[:, 0] + 1j * pairs[:, 1])

    def _error_reader(self):
        """Background thread: keep kiwirecorder's last messages, for error reports."""
        try:
            for line in self.process.stderr:
                self.error_lines.append(line.decode(errors='replace').rstrip())
                del self.error_lines[:-20]
        except (OSError, ValueError):                      # the pipe was closed by close()
            pass

    def next_block(self, samples):
        """Up to `samples` of USB audio that have arrived; (block, False).  The block is empty while not enough
        has arrived.  When the stream ends (the Kiwi closed it, e.g. an owner's time limit, or nothing arrived for
        SILENCE_TIMEOUT_S) the remaining audio is delivered as a last, shorter block and .ended is set: a normal
        end, so the measurements so far can still be saved."""
        if self.ended:
            return np.zeros(0), False
        pieces = [self.pending]
        have = self.pending.size
        stream_over = False
        while have < samples:
            try:
                iq = self.samples.get_nowait()
            except queue.Empty:
                break
            if iq is None:
                stream_over = True
                self.end_reason = 'the Kiwi closed the stream: ' + ' | '.join(self.error_lines[-3:])
                break
            self.last_arrival = time.monotonic()
            audio = self.converter.convert(iq)
            pieces.append(audio)
            have += audio.size
        if not stream_over and time.monotonic() - self.last_arrival > SILENCE_TIMEOUT_S:
            stream_over = True
            self.end_reason = f'no samples for {SILENCE_TIMEOUT_S} s'
        joined = np.concatenate(pieces)
        if stream_over:
            self.ended = True
            self.pending = np.zeros(0)
            print('KiwiSDR stream ended (' + self.end_reason + ')')
            self.received += joined.size
            return joined, False
        if joined.size < samples:
            self.pending = joined
            return np.zeros(0), False
        self.pending = joined[samples:]
        self.received += samples
        return joined[:samples], False

    def where_text(self):
        seconds = self.received / self.true_rate
        return f'{self.name} | {seconds:7.1f} s received | {utc_now():%H:%M:%S} UTC'

    def close(self):
        """Stop kiwirecorder and wait for it, so no recorder outlives this program.  Safe to call more than once."""
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print('warning: kiwirecorder (process %d) did not stop; please end it yourself' % process.pid)
        for pipe in (process.stdout, process.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
