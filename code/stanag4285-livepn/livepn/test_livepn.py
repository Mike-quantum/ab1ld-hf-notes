"""
Tests for s4285-livepn (Python version of DK8OK's STANAG4285 LIVE PN V1.4).

The analyser is based on DK8OK's MATLAB program DK8OK_STANAG4285_LIVE_PN_V1_4.m ((c) 2026 Nils Schiffhauer,
DK8OK; credits Con Wassilieff, ZL2AFP, and Peter Martinez, G3PLX).  Tests: Claude (AI assistant) for Mike Burns,
AB1LD, 2026-10-12.

Two kinds of check:
  * DK8OK's own known answer: his demo source (demoAudio, carrier offset +7 Hz, phase 0.8 + 0.4 sin(2 pi t/15)).
  * Our independent STANAG 4285 generator (s4285_testsignals.stanag4285_audio), which shares nothing with the
    analyser except the published sync sequence: off-grid carrier offsets, phase wander, multipath, noise.
Two tests document properties of V1.4 itself (not of the port) so that we notice if they ever change:
  * scheduling: with 0.2-s audio blocks, PN starts can fall in a short blind gap between analysis windows;
  * the per-frame carrier-offset refinement (unwrap + straight-line fit) can fail under strong ISI.

Run:  python test_livepn.py        (plain script, prints a summary)
  or: python -m pytest test_livepn.py
About 1-2 minutes in total (more on a slow computer).
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s4285_reference import FB, SPS, FRAME, PN31_BITS, pn31_bits, pn80_symbols, rrc_sqrt, rrc_pulse
from s4285_livepn import LivePN, LOG_COLUMNS, EQ_COLUMNS, ARCHIVE_COLUMNS
from s4285_testsignals import demo_audio, demo_truth, demo_pn_starts, stanag4285_audio

FS = 48000

# Column positions in logdata / eqdata rows (names are DK8OK's).
TIME = LOG_COLUMNS.index('Elapsed_s')
PN_BER = LOG_COLUMNS.index('PN_BER')
CFO = LOG_COLUMNS.index('CFO_Hz')
PHASE = LOG_COLUMNS.index('FittedPhase_rad')
CORR = LOG_COLUMNS.index('Correlation')
LOCKED = LOG_COLUMNS.index('Locked')
EVM = LOG_COLUMNS.index('EVM')
EQ_BEFORE = EQ_COLUMNS.index('BER_before_same_bits')
EQ_AFTER = EQ_COLUMNS.index('BER_after_EQ')
EQ_READY = EQ_COLUMNS.index('EQ_ready')


# ---------------------------------------------------------------------------------------------- helpers
def run_analyser(audio, block_seconds=0.2, **options):
    """Feed audio to a fresh LivePN in blocks of block_seconds (DK8OK's timer delivers about 0.2 s)."""
    analyser = LivePN(fs=FS, **options)
    block = int(round(block_seconds * FS))
    for first in range(0, audio.size, block):
        analyser.feed(audio[first:first + block])
    return analyser


def locked_rows(analyser):
    """logdata rows of measured (locked) frames, as an array."""
    log = np.array(analyser.logdata)
    return log[log[:, LOCKED] == 1]


def match_to_truth(frame_times, truth_times):
    """For each measured frame time, the index of the nearest true PN start and the time difference (s)."""
    nearest = np.zeros(frame_times.size, dtype=int)
    time_difference = np.zeros(frame_times.size)
    for i, frame_time in enumerate(frame_times):
        differences = frame_time - truth_times
        nearest[i] = np.argmin(np.abs(differences))
        time_difference[i] = differences[nearest[i]]
    return nearest, time_difference


def wrapped(angle):
    """Angle wrapped to (-pi, pi]."""
    return np.angle(np.exp(1j * angle))


def missed_frames(analyser, truth_times, t_from, t_to):
    """Number of true PN starts between t_from and t_to for which the analyser wrote no row at all."""
    row_times = np.array([row[TIME] for row in analyser.logdata])
    in_range = (truth_times > t_from) & (truth_times < t_to)
    wanted = truth_times[in_range]
    missed = 0
    for true_time in wanted:
        closest_row = np.min(np.abs(row_times - true_time))
        if closest_row >= 2 / FB:                     # no row within 2 samples of this PN start
            missed += 1
    return missed, wanted.size


def paths(*delay_gain_pairs):
    """Multipath description for stanag4285_audio: (delay_ms, complex gain) pairs, no Doppler."""
    channel = []
    for delay, gain in delay_gain_pairs:
        channel.append(dict(delay_ms=delay, gain=gain, doppler_hz=0.0))
    return channel


# ---------------------------------------------------------------------------------------------- reference
def test_pn31_and_pn80():
    """PN31 from DK8OK's LFSR equals the STANAG 4285 Annex A sequence; PN80 = PN31 + PN31 + first 18."""
    bits = pn31_bits()
    assert ''.join(map(str, bits)) == PN31_BITS
    assert bits.sum() == 16                                  # m-sequence: 16 ones, 15 zeros
    pn = pn80_symbols()
    assert pn.size == 80
    assert np.array_equal(pn[:31], pn[31:62])
    assert np.array_equal(pn[62:], pn[:18])
    assert np.array_equal(pn[:31], 2.0 * bits - 1)


def test_rrc():
    """rrc_sqrt: 81 taps, symmetric, unit energy, equal to the analytic pulse; RRC*RRC is Nyquist at 8 sps."""
    h = rrc_sqrt()
    assert h.size == 10 * SPS + 1
    assert np.allclose(h, h[::-1])
    assert abs(np.sum(h ** 2) - 1) < 1e-12
    n = np.arange(-40, 41) / SPS
    analytic = rrc_pulse(n) / np.sqrt(np.sum(rrc_pulse(n) ** 2))
    assert np.allclose(h, analytic)
    raised_cosine = np.convolve(h, h)
    centre = raised_cosine.size // 2
    other_symbols = raised_cosine[centre % SPS::SPS]
    other_symbols = np.delete(other_symbols, centre // SPS)
    assert np.max(np.abs(other_symbols)) < 0.01 * raised_cosine[centre]   # truncation to 10 symbols


# ---------------------------------------------------------------------------------------------- DK8OK demo
def test_demo_known_answer():
    """DK8OK's built-in demo: BER 0, CFO +7 Hz, channel phase, frame time (+1 sample: MATLAB 1-based index)."""
    duration = 10
    rng = np.random.default_rng(0)
    pieces = []
    for first in range(0, duration * FS, 9600):                  # his demo, generated in 0.2-s pieces
        last = min(first + 9600, duration * FS)
        pieces.append(demo_audio(first, last, rng=rng))
    audio = np.concatenate(pieces)
    analyser = run_analyser(audio)
    rows = locked_rows(analyser)
    assert len(rows) == len(analyser.logdata)          # every analysed frame locked
    assert len(rows) >= 80
    assert np.all(rows[:, PN_BER] == 0)
    assert abs(np.mean(rows[:, CFO]) - 7) < 0.1
    truth_times = demo_pn_starts(duration)
    nearest, dt = match_to_truth(rows[:, TIME], truth_times)
    assert np.allclose(dt * FB, 1.0, atol=1e-6)               # his absindex is 1-based
    _, true_phase = demo_truth(truth_times[nearest])
    phase_error = wrapped(rows[:, PHASE] - true_phase)
    assert np.degrees(np.sqrt(np.mean(phase_error ** 2))) < 2
    eq = np.array(analyser.eqdata)
    ready = eq[eq[:, EQ_READY] == 1]
    assert len(ready) == len(eq) - 1                          # all but the first frame
    assert np.all(ready[:, EQ_AFTER] == 0)
    assert np.mean(rows[:, EVM]) < 0.1


# ---------------------------------------------------------------------------------------------- independent
def test_offgrid_cfo_and_phase_wander():
    """Our generator, one path: CFO -23.4 Hz (between 15-Hz grid points), phase wander 0.5 rad / 7 s."""
    audio, truth = stanag4285_audio(10, cfo=-23.4, wander=(0.5, 7.0))
    rows = locked_rows(run_analyser(audio))
    assert len(rows) >= 80
    assert np.all(rows[:, PN_BER] == 0)
    assert abs(np.mean(rows[:, CFO]) + 23.4) < 0.1
    assert np.std(rows[:, CFO]) < 0.2
    nearest, dt = match_to_truth(rows[:, TIME], truth['pn_start'])
    assert np.all(np.abs(dt * FB) <= 1.5)                     # within the sample grid (+1-sample convention)
    assert np.std(dt * FB) < 1e-6
    phase_error = wrapped(rows[:, PHASE] - truth['phase'][nearest])
    assert np.degrees(np.sqrt(np.mean(phase_error ** 2))) < 1


def test_lock_range():
    """Search is +/-450 Hz in 15-Hz steps: 440 and -455 Hz lock, 600 Hz does not."""
    for cfo in (440.0, -455.0):
        audio, _ = stanag4285_audio(4, cfo=cfo, snr_db=30)
        rows = locked_rows(run_analyser(audio))
        assert len(rows) >= 25, cfo
        assert abs(np.mean(rows[:, CFO]) - cfo) < 0.5, cfo
    audio, _ = stanag4285_audio(4, cfo=600.0, snr_db=30)
    assert len(locked_rows(run_analyser(audio))) == 0


def test_low_snr():
    """10 dB SNR in 3 kHz: every frame locks, PN BER 0, CFO within 0.5 Hz on average."""
    audio, truth = stanag4285_audio(10, cfo=12.3, snr_db=10)
    analyser = run_analyser(audio)
    rows = locked_rows(analyser)
    assert len(rows) == len(analyser.logdata)          # every analysed frame locked
    assert len(rows) >= 80
    assert np.all(rows[:, PN_BER] == 0)
    assert abs(np.mean(rows[:, CFO]) - 12.3) < 0.5


def test_pure_noise_no_lock():
    """60 s of white noise: no lock; the best coherent score stays well under the 0.48 gate."""
    noise = 0.1 * np.random.default_rng(7).standard_normal(60 * FS)
    analyser = run_analyser(noise)
    log = np.array(analyser.logdata)
    assert len(log) > 50
    assert np.sum(log[:, LOCKED]) == 0
    assert np.nanmax(log[:, CORR]) < 0.40


def test_two_path_multipath():
    """Two paths (main + 0.6 at 0.42 ms, and main + 0.6j at 1 ms): locks, CFO within 0.5 Hz, PN BER < 1 %."""
    for channel in (paths((0, 1), (0.42, -0.6)), paths((0, 1), (1.0, 0.6j))):
        audio, _ = stanag4285_audio(8, cfo=5, snr_db=25, paths=channel)
        analyser = run_analyser(audio)
        rows = locked_rows(analyser)
        assert len(rows) == len(analyser.logdata)
        assert len(rows) >= 65
        assert abs(np.mean(rows[:, CFO]) - 5) < 0.5
        assert np.mean(rows[:, PN_BER]) < 0.01


def test_equalizer_helps():
    """Symbol-spaced ISI (gains 1, -0.55, +0.5 at 0, 1, 2 symbols): EQ after-BER far below before-BER."""
    channel = paths((0, 1), (1 / 2.4, -0.55), (2 / 2.4, 0.5))
    audio, _ = stanag4285_audio(8, cfo=5, snr_db=25, paths=channel)
    analyser = run_analyser(audio)
    eq = np.array(analyser.eqdata)
    ready = eq[eq[:, EQ_READY] == 1]
    assert len(ready) >= 50
    before = np.mean(ready[:, EQ_BEFORE])
    after = np.mean(ready[:, EQ_AFTER])
    assert before > 0.005
    assert after < before / 5


def test_equalizer_off():
    """eq_enable=False: no EQ results, EQ_ready always 0."""
    audio, _ = stanag4285_audio(3, cfo=5, snr_db=25)
    eq = np.array(run_analyser(audio, eq_enable=False).eqdata)
    assert np.all(eq[:, EQ_READY] == 0)
    assert np.all(np.isnan(eq[:, EQ_AFTER]))


# ---------------------------------------------------------------------------------------------- scheduling
def usable_span():
    """Seconds of PN-start positions one analysis can measure: window minus 2 margins and one PN80 (at FB)."""
    return 0.85 - (2 * 240 + 80 * SPS) / FB


def analysis_cadence(block_seconds, min_interval=0.65):
    """Audio time between analyses when blocks arrive every block_seconds (min_interval rounded up)."""
    return np.ceil(min_interval / block_seconds - 1e-9) * block_seconds


def test_scheduling_coverage_v14():
    """DK8OK V1.4 scheduling, documented: cadence 0.8 s (0.2-s blocks) > usable span 0.792 s leaves an 8-ms
    blind gap; 0.8 s is 7.5 frames, so the stream either loses 1 frame in 15 or none, depending on where the
    frames fall.  Signal starting at 0.025 s loses 1 in 15; 0.05-s blocks or min_interval 0.5 lose none."""
    assert analysis_cadence(0.2) > usable_span()
    assert analysis_cadence(0.05) <= usable_span()
    assert analysis_cadence(0.2, 0.5) <= usable_span()
    audio, truth = stanag4285_audio(20, cfo=3, snr_db=30, start=0.025)
    missed, total = missed_frames(run_analyser(audio, 0.2), truth['pn_start'], 1.0, 19.0)
    assert missed == round(total / 15), (missed, total)
    for block_seconds, min_interval in ((0.05, 0.65), (0.2, 0.5)):
        analyser = run_analyser(audio, block_seconds, min_interval=min_interval)
        assert missed_frames(analyser, truth['pn_start'], 1.0, 19.0)[0] == 0
    audio, truth = stanag4285_audio(20, cfo=3, snr_db=30, start=0.0123)
    assert missed_frames(run_analyser(audio, 0.2), truth['pn_start'], 1.0, 19.0)[0] == 0


def test_cfo_refinement_fragility_v14():
    """DK8OK V1.4 behaviour, documented: under strong ISI the PN-stripped samples pass near zero, the
    unwrap + straight-line phase fit slips cycles, the 'refined' CFO lands tens of Hz off and every frame is
    rejected -- although the correlation at the acquisition CFO is about 0.7, well above the 0.48 gate."""
    channel = paths((0, 1), (0.42, -0.9), (1.0, 0.7j))
    audio, _ = stanag4285_audio(4, cfo=5, snr_db=30, paths=channel)
    analyser = run_analyser(audio)
    log = np.array(analyser.logdata)
    rejected = log[(log[:, LOCKED] == 0) & np.isnan(log[:, CFO])]    # rejected after acquisition passed
    assert len(rejected) == len(log)                   # every row is such a rejection
    assert len(rejected) > 20
    assert np.max(log[:, CORR]) < 0.48


def test_cfo_guard_option():
    """Ours, off by default: cfo_method='guarded' changes nothing on a clean signal, rescues the strong-ISI
    frames that V1.4 rejects (with a CFO measurement near the truth), and does not lock on noise."""
    audio, _ = stanag4285_audio(4, cfo=-23.4, wander=(0.5, 7.0))
    plain = run_analyser(audio)
    guarded = run_analyser(audio, cfo_method='guarded')
    assert guarded.cfo_fallbacks == 0
    assert np.array_equal(np.array(plain.logdata), np.array(guarded.logdata), equal_nan=True)
    channel = paths((0, 1), (0.42, -0.9), (1.0, 0.7j))
    audio, _ = stanag4285_audio(4, cfo=5, snr_db=30, paths=channel)
    analyser = run_analyser(audio, cfo_method='guarded')
    rows = locked_rows(analyser)
    assert len(rows) == len(analyser.logdata)
    assert len(rows) > 30
    assert abs(np.median(rows[:, CFO]) - 5) < 0.5
    eq = np.array(analyser.eqdata)
    ready = eq[eq[:, EQ_READY] == 1]
    assert np.mean(ready[:, EQ_AFTER]) < np.mean(ready[:, EQ_BEFORE]) / 5
    noise = 0.1 * np.random.default_rng(8).standard_normal(20 * FS)
    assert len(locked_rows(run_analyser(noise, cfo_method='guarded'))) == 0


# ---------------------------------------------------------------------------------------------- outputs
def test_outputs_and_capture_gap():
    """joined_rows (his saveCSV) and archive_values (1-s bins); a capture gap blanks its second."""
    audio, _ = stanag4285_audio(5, cfo=5, snr_db=25)
    analyser = run_analyser(audio)
    joined = analyser.joined_rows()
    assert len(joined) == len(analyser.logdata)
    assert len(joined[0]) == len(LOG_COLUMNS) + len(EQ_COLUMNS) - 1
    archive = analyser.archive_values()
    assert archive.shape[1] == len(ARCHIVE_COLUMNS)
    assert np.all(np.diff(archive[:, 0]) == 1)
    full_seconds = archive[(archive[:, 0] > 1) & (archive[:, 0] < 4)]
    assert np.all(full_seconds[:, ARCHIVE_COLUMNS.index('LockFraction')] == 1)
    assert np.allclose(full_seconds[:, ARCHIVE_COLUMNS.index('CarrierOffset_Hz')], 5, atol=1)
    analyser.capture_gap(10.0)
    analyser.feed(audio[:FS])
    archive = analyser.archive_values()
    gap_row = archive[archive[:, 0] == 10.5][0]
    assert gap_row[ARCHIVE_COLUMNS.index('CaptureGaps')] >= 1
    assert np.all(np.isnan(gap_row[1:6]))


def test_offline_cli(tmp_path=None):
    """livepn_offline: 16-bit stereo WAV at 44.1 kHz (signal on the right) -> both CSVs with his columns."""
    import csv
    import tempfile
    from scipy.io import wavfile
    import livepn_offline
    if tmp_path is not None:
        folder = str(tmp_path)              # pytest's temporary folder
    else:
        folder = tempfile.mkdtemp()
    audio, _ = stanag4285_audio(4, fs=44100, cfo=-8.3, snr_db=25)
    stereo = np.zeros((audio.size, 2), np.int16)
    stereo[:, 1] = np.round(audio * 0.8 * 32767).astype(np.int16)
    wav = os.path.join(folder, 'test_20261012T120000Z.wav')
    wavfile.write(wav, 44100, stereo)
    livepn_offline.main([wav, '--channel', 'right', '--no-plots', '--note', 'test'])
    with open(os.path.join(folder, 'test_20261012T120000Z_DK8OK_LIVE_PN.csv')) as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == LOG_COLUMNS + EQ_COLUMNS[1:] + ['UTC', 'Note']
    assert len(rows) > 25
    assert rows[5][-2].startswith('2026-10-12 12:00:0') and rows[5][-1] == 'test'
    assert abs(float(rows[5][LOG_COLUMNS.index('CFO_Hz')]) + 8.3) < 1
    with open(os.path.join(folder, 'test_20261012T120000Z_DK8OK_LIVE_PN_ARCHIVE.csv')) as handle:
        header = next(csv.reader(handle))
    assert header == ARCHIVE_COLUMNS + ['UTC', 'Note', 'AudioCentre_Hz']


# ---------------------------------------------------------------------------------------------- review fixes
# Added 2026-10-14 after the reviews by GPT-6 Astra and Claude Fable (findings A3, A4, A8, A10, A11, B1).

FAKE_RECORDER = """
# A stand-in for kiwirecorder.py --nc --nc-wav, for tests only.
import os
import struct
import sys
import time

mode = os.environ.get('FAKE_MODE', 'short')
out = sys.stdout.buffer
rate = 11999

# A standard 44-byte WAV header: 2 channels (I, Q), 16 bit, `rate` samples/s.
header = b'RIFF'
header += struct.pack('<I', 36)
header += b'WAVE'
header += b'fmt '
header += struct.pack('<I', 16)                 # size of the format block
header += struct.pack('<H', 1)                  # PCM
header += struct.pack('<H', 2)                  # channels
header += struct.pack('<I', rate)               # samples per second
header += struct.pack('<I', rate * 4)           # bytes per second
header += struct.pack('<H', 4)                  # bytes per sample pair
header += struct.pack('<H', 16)                 # bits per sample
header += b'data'
header += struct.pack('<I', 0)

if mode == 'zeros':
    # A broken header (all zeros), then stay alive like a recorder that hangs.
    out.write(bytes(44))
    out.flush()
    time.sleep(60)
else:
    # A good header, 3 s of silence, then the stream ends.
    out.write(header)
    out.write(bytes(3 * rate * 4))
    out.flush()
"""


def no_status_check(*arguments):
    """Stand-in for check_kiwi_before_connecting in tests: no network, no warning."""
    return ''


def fake_kiwiclient(folder):
    """A folder with a stand-in kiwirecorder.py (no network): 'zeros' sends a bad header and stays alive,
    'short' sends a header and 3 s of silence, then ends the stream."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'kiwirecorder.py'), 'w') as handle:
        handle.write(FAKE_RECORDER)
    return folder


def test_kiwi_source_stops_recorder_on_bad_start():
    """A recorder that sends a bad header is stopped before the error is passed on (none left running)."""
    import tempfile
    import s4285_sources
    folder = fake_kiwiclient(os.path.join(tempfile.mkdtemp(), 'fake kiwi'))
    original_check = s4285_sources.check_kiwi_before_connecting
    s4285_sources.check_kiwi_before_connecting = no_status_check
    os.environ['FAKE_MODE'] = 'zeros'
    started = []
    original_popen = s4285_sources.subprocess.Popen

    def remembering_popen(*arguments, **keywords):
        process = original_popen(*arguments, **keywords)
        started.append(process)
        return process

    s4285_sources.subprocess.Popen = remembering_popen
    try:
        try:
            s4285_sources.KiwiSource('fake.host', 8073, 8552.6, kiwiclient_dir=folder)
            raised = False
        except SystemExit:
            raised = True
    finally:
        s4285_sources.subprocess.Popen = original_popen
        s4285_sources.check_kiwi_before_connecting = original_check
    assert raised
    assert started and started[0].poll() is not None


def test_live_save_when_stream_ends():
    """When a Kiwi stream ends on its own, --save still writes both CSVs, with Windows-safe names."""
    import glob
    import tempfile
    import s4285_sources
    import livepn_live
    base = tempfile.mkdtemp()
    folder = fake_kiwiclient(os.path.join(base, 'fake kiwi'))
    save = os.path.join(base, 'saved')
    original_check = s4285_sources.check_kiwi_before_connecting
    s4285_sources.check_kiwi_before_connecting = no_status_check
    os.environ['FAKE_MODE'] = 'short'
    try:
        livepn_live.main(['--kiwi', 'fake.host:8073', '--dial', '8552.6', '--kiwiclient', folder,
                          '--snapshot', os.path.join(base, 'snap.png'), '--duration', '20', '--save', save])
    finally:
        s4285_sources.check_kiwi_before_connecting = original_check
    names = sorted(os.path.basename(path) for path in glob.glob(os.path.join(save, '*.csv')))
    assert len(names) == 2
    for name in names:
        assert ':' not in name and '8552p60kHz' in name


def test_compare_finds_start_time():
    """compare_with_matlab: a whole 'MATLAB-like' run (irregular 0.20-0.28-s pieces) and the Python run on its
    last 10 s; the start time is found exactly, and a capture gap inside the 10 s is reported."""
    import tempfile
    import contextlib
    import io
    from scipy.io import wavfile
    import livepn_offline
    import compare_with_matlab
    folder = tempfile.mkdtemp()
    duration = 30.0
    paths = [dict(delay_ms=0, gain=1.0, doppler_hz=0.0), dict(delay_ms=0.42, gain=-0.3, doppler_hz=0.0)]
    audio, _ = stanag4285_audio(duration, cfo=12.3, snr_db=25, paths=paths, seed=2)
    audio = np.clip(np.round(audio * 32768), -32768, 32767) / 32768.0
    analyser = LivePN()
    rng = np.random.default_rng(102)
    position = 0
    while position < audio.size:
        block = int(round((0.2 + rng.uniform(0, 0.08)) * FS))
        analyser.feed(audio[position:position + block])
        position += block
    matlab_csv = os.path.join(folder, 'DK8OK_LIVE_PN.csv')
    livepn_offline.save_measurements(analyser, matlab_csv, None, 'test')
    wav = os.path.join(folder, 'DK8OK_LIVE_PN_INPUT.wav')
    wavfile.write(wav, FS, np.clip(np.round(audio[-10 * FS:] * 32768), -32768, 32767).astype(np.int16))
    with contextlib.redirect_stdout(io.StringIO()):
        livepn_offline.main([wav, '--no-plots'])
    python_csv = os.path.join(folder, 'DK8OK_LIVE_PN_INPUT_DK8OK_LIVE_PN.csv')

    matlab = compare_with_matlab.read_measurements(matlab_csv)
    python = compare_with_matlab.read_measurements(python_csv)
    candidates = compare_with_matlab.find_offset(matlab, python, compare_with_matlab.locked_rows(matlab),
                                                 compare_with_matlab.locked_rows(python), 0.005, False)
    offset, pairs, score = candidates[0]
    assert abs(offset - (duration - 10)) < 0.001
    assert score < 0.6 and candidates[1][2] >= 1.5 * score

    # A capture gap row inside the WAV's span must be reported.
    gap_time = duration - 5
    with open(matlab_csv, 'a', newline='') as handle:
        handle.write(f'{gap_time},NaN,NaN,NaN,NaN,NaN,0,NaN,1,NaN,NaN,NaN,NaN,NaN,,test\n')
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        compare_with_matlab.main([matlab_csv, python_csv, '--offset', str(duration - 10), '--rows', '0'])
    assert 'capture gap' in report.getvalue()


# ---------------------------------------------------------------------------------------------- runner
def main():
    """Run every test_ function in this file; print one line each and a summary."""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    tests = []
    for name, value in list(globals().items()):
        if name.startswith('test_') and callable(value):
            tests.append((name, value))
    failures = 0
    for name, test in tests:
        start = time.time()
        try:
            test()
            result = 'ok'
        except AssertionError as error:
            failures += 1
            result = 'FAIL %s' % (error,)
        print('%-40s %-6s %5.1f s' % (name, result, time.time() - start))
    print('%d tests, %d failed' % (len(tests), failures))
    return failures


if __name__ == '__main__':
    failures = main()
    if failures:
        sys.exit(1)
