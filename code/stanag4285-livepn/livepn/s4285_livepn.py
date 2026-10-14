"""
s4285_livepn -- STANAG 4285 PN80 monitor with a prior-frame ridge equalizer (block-fed core).

Python version of DK8OK's MATLAB program DK8OK_STANAG4285_LIVE_PN_V1_4.m ("STANAG4285: P/N Sequence and
Equalizer", Dopplergram group, 2026-10-11).  (c) 2026 Nils Schiffhauer, DK8OK -- method, constants, thresholds
and order of operations are his; DK8OK credits Con Wassilieff, ZL2AFP, and Peter Martinez, G3PLX, and wrote the
MATLAB program with the help of ChatGPT.  Python version: Claude (AI assistant) for Mike Burns, AB1LD, 2026-10,
to be donated back to DK8OK.  It is not a line-by-line translation: the measurement core is the same, the user
interface is replaced by a block-fed class that the offline WAV reader (livepn_offline.py) and the live window
(livepn_live.py, fed from recordings or a KiwiSDR) drive.

What one analysis does (DK8OK's analyse()):
  1. Take the last 0.85 s of real USB audio, make it complex (Hilbert transform), and mix it down so that the
     audio centre (1800 Hz) lands at 0 Hz.  The mixing oscillator runs on the ABSOLUTE sample index, so its
     phase is continuous from one analysis to the next.
  2. Resample 48000 -> 19200 samples/s (8 samples per symbol) and apply the root-raised-cosine matched filter
     (beta 0.2, span 10 symbols).
  3. Acquisition: search for the PN80 sync over carrier offsets -450..+450 Hz in 15-Hz steps and over all 8
     sample phases.  The score is the normalised correlation; a second PN80 exactly one frame (256 symbols)
     later is required, and the smaller of the two scores counts.  Lock needs a score of at least 0.48.
  4. For every frame on that grid not measured before: residual carrier offset from a straight-line fit to the
     PN-stripped phase, channel phase at the PN start, PN bit error rate (hard decisions on the 80 sync
     symbols) and error-vector magnitude.
  5. Equalizer: a symmetric, symbol-spaced FIR filter (default 9 taps), fitted by ridge regression (ridge
     0.02 x mean tap energy) on the PREVIOUS accepted frame and tested on this one before it is retrained.
     Before/after bit error rates are counted on the same inner symbols.
Scheduling (DK8OK's tick()): audio arrives in blocks; a 2.5-s raw buffer keeps absolute sample indices; an
analysis runs when more than 0.4 s is buffered and at least 0.65 s of audio has arrived since the last one.

Python equivalents of the MATLAB calls: hilbert -> scipy.signal.hilbert (same FFT method); resample(z,fb,fs) ->
scipy.signal.resample_poly (same Kaiser-windowed design, beta 5, 10 x max(p,q) half length); conv(...,'same')
-> numpy.convolve(...,'same') (identical for an odd-length filter); polyfit/unwrap -> numpy; '\\' -> solve.
MATLAB counts samples from 1; where such an index reaches the output (absolute frame index, frame times) it is
kept 1-based here too, so times match his program.
"""
import numpy as np
from scipy import signal

from s4285_reference import FB, SPS, FRAME, AUDIO_CENTRE, pn31_bits, pn80_symbols, rrc_sqrt

VERSION = '0.3 (2026-10-12), based on DK8OK STANAG4285 LIVE PN V1.4'

# ---------------------------------------------------------------------------------------------- output columns
# Column names are DK8OK's (his CSV export).
LOG_COLUMNS = ['Elapsed_s', 'PN_BER', 'CFO_Hz', 'FittedPhase_rad', 'Correlation', 'ClippedFraction',
               'Locked', 'EVM', 'CaptureGap']
EQ_COLUMNS = ['Elapsed_s', 'BER_before_same_bits', 'BER_after_EQ', 'EVM_after_EQ', 'ComparedBits', 'EQ_ready']
ARCHIVE_COLUMNS = ['Elapsed_s', 'BER_before_pct', 'BER_after_pct', 'CarrierOffset_Hz', 'AudioRMS_dBFS',
                   'AudioPeak_dBFS', 'LockFraction', 'PN_Correlation', 'CaptureGaps']

# Positions within one row of .logdata (same order as LOG_COLUMNS).
LOG_TIME = 0
LOG_PN_BER = 1
LOG_CFO = 2
LOG_PHASE = 3
LOG_CORRELATION = 4
LOG_CLIPPED = 5
LOG_LOCKED = 6
LOG_EVM = 7
LOG_CAPTURE_GAP = 8

# Positions within one row of .eqdata (same order as EQ_COLUMNS).
EQ_TIME = 0
EQ_BER_BEFORE = 1
EQ_BER_AFTER = 2
EQ_EVM_AFTER = 3
EQ_COMPARED_BITS = 4
EQ_READY = 5

# The 1-second archive keeps running sums per second (DK8OK's 15 archive columns); positions in one row:
ARC_SECOND = 0                # the whole second this row describes
ARC_BER_BEFORE_SUM = 1        # sum of before-EQ BER over frames in this second ...
ARC_BER_BEFORE_COUNT = 2      # ... and how many frames contributed
ARC_BER_AFTER_SUM = 3         # same for after-EQ BER
ARC_BER_AFTER_COUNT = 4
ARC_CFO_SUM = 5               # carrier offset, locked frames only
ARC_CFO_COUNT = 6
ARC_POWER_SUM = 7             # sum of squared audio samples ...
ARC_SAMPLE_COUNT = 8          # ... and the number of samples (-> RMS level)
ARC_PEAK_POWER = 9            # largest squared audio sample (-> peak level)
ARC_LOCKED_FRAMES = 10        # frames locked in this second ...
ARC_ALL_FRAMES = 11           # ... out of all rows written in this second (-> lock fraction)
ARC_CORRELATION_SUM = 12      # PN correlation (finite values only)
ARC_CORRELATION_COUNT = 13
ARC_CAPTURE_GAPS = 14         # capture gaps flagged in this second
ARCHIVE_WIDTH = 15

# ---------------------------------------------------------------------------------------------- DK8OK constants
LOCK_GATE = 0.48              # minimum normalised PN80 correlation for lock
MAX_CFO_HZ = 470              # a frame whose refined carrier offset is beyond this is rejected
SEARCH_LOW_HZ = -450          # acquisition search range and step (Hz)
SEARCH_HIGH_HZ = 450
SEARCH_STEP_HZ = 15
EDGE_MARGIN = 240             # samples at FB kept clear at both ends of an analysis window
PN_LENGTH = 80                # sync symbols per frame
FRAME_SYMBOLS = 256           # symbols per frame (one frame later = 256 symbols later)
EQ_MAX_AGE_S = 0.3            # equalizer taps older than this are not used (next frame comes after 0.107 s)
LOG_LIMIT = 100000            # rows kept in the logs and seconds kept in the archive
EPS = np.finfo(float).eps     # tiny number that keeps divisions finite


# ---------------------------------------------------------------------------------------------- helpers (ours)
def pn_correlation(y, pn, t, f):
    """|sum of y * PN80| after removing a carrier offset f (Hz) -- the quantity DK8OK's per-frame gate uses.

    y: the 80 received sync samples; pn: the +/-1 reference; t: their times (s) from the first sync symbol.
    Used by cfo_method 'guarded' (ours) to compare two carrier-offset estimates.
    """
    offset_removed = y * np.exp(-2j * np.pi * f * t)
    return abs(np.sum(offset_removed * pn))


def ml_offset(yd, t, half_range=7.5, step=0.05):
    """Maximum-likelihood residual carrier offset (Hz) of PN-stripped samples yd at times t (ours).

    For known symbols in white noise the maximum-likelihood frequency estimate is the frequency df that
    maximises |sum over k of yd[k] exp(-j 2 pi df t[k])| -- a periodogram peak.  It needs no phase unwrapping,
    so it cannot slip a cycle.  Searched over +/-half_range Hz (half the 15-Hz acquisition grid) in steps of
    `step`, then refined by a parabola through the peak and its two neighbours.
    Used by cfo_method 'guarded' only.
    """
    trial_offsets = np.arange(-half_range, half_range + step / 2, step)

    # One row per trial offset: the phasors that remove that offset from every sample.
    rotations = np.exp(-2j * np.pi * trial_offsets[:, None] * t[None, :])
    response = np.abs(rotations @ yd)

    peak = int(np.argmax(response))
    peak_is_inside = 0 < peak < trial_offsets.size - 1
    if peak_is_inside:
        left = response[peak - 1]
        centre = response[peak]
        right = response[peak + 1]
        curvature = left - 2 * centre + right
        if curvature < 0:
            # Vertex of the parabola through the three points (shift from the centre point, in Hz).
            vertex_shift = 0.5 * step * (left - right) / curvature
            return trial_offsets[peak] + vertex_shift
    return trial_offsets[peak]


# ---------------------------------------------------------------------------------------------- the analyser
def row_time(row):
    """Sort key: a log row's elapsed time (its first value)."""
    return row[0]


class LivePN:
    """Block-fed STANAG 4285 PN80 analyser (DK8OK LIVE PN V1.4 signal path).

    Feed real audio blocks with feed(block).  Results accumulate in .logdata (one row per measured frame or per
    failed acquisition, columns LOG_COLUMNS), .eqdata (EQ_COLUMNS) and the 1-second archive (archive_values(),
    ARCHIVE_COLUMNS).  .last holds the most recent frame for displays.

    Defaults are DK8OK's.  Parameters he exposes in his window: centre, taps, ridge, eq_enable, hold, bits
    (PN31 reference).

    Option (ours, off by default): cfo_method='guarded'.  DK8OK refines each frame's carrier offset with a
    straight-line fit to the unwrapped PN-stripped phase.  Under strong ISI or fading some of those samples pass
    near zero, the unwrap slips a cycle, the 'refined' offset lands tens of Hz away and the frame is rejected
    although it correlates well at the acquisition offset.  'guarded' keeps his estimate whenever it does not
    lower the PN correlation; otherwise it uses the maximum-likelihood offset for the known PN80 (ml_offset).
    'dk8ok' (default) is his method unchanged.

    The scheduling constants (window 0.85 s, min_interval 0.65 s, buffer 2.5 s, min_buffer 0.4 s) are fixed in
    his program; they are parameters here so that an offline run can choose complete frame coverage --
    leaving them at the defaults reproduces his program.
    """

    def __init__(self, fs=48000, centre=AUDIO_CENTRE, taps=9, ridge=0.02, eq_enable=True, hold=3.0,
                 bits=None, window=0.85, min_interval=0.65, buffer=2.5, min_buffer=0.4,
                 keep_spectra=False, keep_points=False, cfo_method='dk8ok'):
        if taps not in (5, 9, 13, 17):
            raise ValueError('taps must be 5, 9, 13 or 17 (DK8OK offers these)')
        if cfo_method not in ('dk8ok', 'guarded'):
            raise ValueError("cfo_method must be 'dk8ok' (his) or 'guarded'")

        # Settings.
        self.fs = fs                          # audio sample rate (Hz)
        self.centre = float(centre)           # audio frequency of the signal centre (Hz)
        self.taps = int(taps)                 # equalizer length (symbols)
        self.ridge = float(ridge)             # equalizer ridge factor
        self.eq_enable = bool(eq_enable)
        self.hold = float(hold)               # seconds of constellation points kept for display
        self.cfo_method = cfo_method

        # Scheduling (DK8OK's fixed values by default).
        self.window = window                  # seconds of audio analysed each time
        self.min_interval = min_interval      # minimum audio time between analyses (s)
        self.buffer = buffer                  # seconds of raw audio kept
        self.min_buffer = min_buffer          # audio needed before the first analysis (s)

        # What to keep for plots.
        self.keep_spectra = keep_spectra
        self.keep_points = keep_points

        # Sync reference and matched filter.
        if bits is None:
            self.bits = pn31_bits()
        else:
            self.bits = np.asarray(bits, int)
        self.pn = pn80_symbols(self.bits)
        self.h = rrc_sqrt()

        # Resampling ratio audio rate -> FB as a reduced fraction up/down (48000 -> 19200 is 2/5).
        common = np.gcd(FB, fs)
        self.up = FB // common
        self.down = fs // common

        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """DK8OK startRun(): clear all buffers, logs and the equalizer."""
        # Raw audio buffer.  raw[0] is absolute sample number raw_start of the stream.
        self.raw = np.zeros(0)
        self.raw_start = 0
        self.origin = 0.0                     # elapsed seconds at absolute sample 0; moved by capture_gap()
        self.samples_in = 0

        # Bookkeeping between analyses.
        self.consumed = -np.inf               # absolute index (at FB) of the last frame measured
        self.last_analysis = -np.inf          # audio time of the last analysis (s)
        self.losses = 0                       # analyses without lock
        self.cfo_fallbacks = 0                # frames where cfo_method 'guarded' replaced his estimate

        # Results.
        self.logdata = []                     # rows of LOG_COLUMNS
        self.eqdata = []                      # rows of EQ_COLUMNS
        self.archive = {}                     # whole second -> ARCHIVE_WIDTH running sums
        self.archive_last_frame = -np.inf     # newest log time already folded into the archive

        # Equalizer: taps from the previous accepted frame and that frame's time.
        self.eqW = None
        self.eq_last = -np.inf

        # Display material.
        self.cloud = []                       # (time, aligned PN symbols) within the hold time
        self.eq_cloud = []                    # (time, equalizer output) within the hold time
        self.points = []                      # every frame if keep_points: (time, aligned, eq output, received)
        self.spectra = []                     # every block if keep_spectra: (time, frequencies, dB)
        self.last = None                      # the most recent measured frame (dictionary)
        self.status = 'Ready.'

    # ------------------------------------------------------------------ tick
    def feed(self, x):
        """DK8OK tick(): accept one block of new audio (mono, full scale +/-1) and analyse when it is time."""
        x = np.asarray(x, dtype=float).ravel()
        if x.size == 0:
            return
        self.samples_in += x.size

        # Fraction of samples at or near full scale (his "ClippedFraction").
        clip = float(np.mean(np.abs(x) >= .99))

        # Append to the raw buffer; drop the oldest samples beyond the buffer length.
        self.raw = np.concatenate([self.raw, x])
        max_samples = round(self.buffer * self.fs)
        if self.raw.size > max_samples:
            excess = self.raw.size - max_samples
            self.raw = self.raw[excess:]
            self.raw_start += excess

        # Audio level and spectrum are recorded for every block, locked or not (his paintAudio).
        self._audio(x)

        # Analyse when enough audio is buffered and enough time has passed since the last analysis.
        audio_time = self.origin + (self.raw_start + self.raw.size) / self.fs
        enough_buffered = self.raw.size > self.min_buffer * self.fs
        interval_passed = audio_time - self.last_analysis >= self.min_interval
        if enough_buffered and interval_passed:
            self.last_analysis = audio_time
            self.analyse(clip)

        self._collect_archive()

    def capture_gap(self, origin):
        """DK8OK's recorder renewal (every 60 s in his live program): flag a capture gap and restart the
        buffer, with absolute sample 0 now at elapsed time `origin` (s)."""
        self.origin = float(origin)
        gap_row = [self.origin, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, 1]
        self.logdata.append(gap_row)
        self._collect_archive()

        self.raw = np.zeros(0)
        self.raw_start = 0
        self.consumed = -np.inf
        self.eqW = None
        self.eq_last = -np.inf
        self.status = 'Recorder renewed: capture gap flagged; reacquiring.'

    # ------------------------------------------------------------------ audio level / spectrum
    def _archive_row(self, second):
        """The archive row (running sums) for one whole second; created empty when first needed."""
        second = int(second)
        if second not in self.archive:
            new_row = np.zeros(ARCHIVE_WIDTH)
            new_row[ARC_SECOND] = second
            self.archive[second] = new_row
            if len(self.archive) > LOG_LIMIT:
                oldest = min(self.archive)
                del self.archive[oldest]
        return self.archive[second]

    def _audio(self, block):
        """DK8OK paintAudio(): per-second power and peak of the new block; optionally the spectrum of the
        last 0.25 s."""
        fs = self.fs
        spectrum_samples = min(self.raw.size, round(.25 * fs))
        if spectrum_samples < 64:
            return

        # Elapsed time at the end of the buffer, and the whole second each sample of the block falls in.
        audio_now = self.origin + (self.raw_start + self.raw.size) / fs
        block_start_time = max(0.0, audio_now - block.size / fs)
        sample_seconds = np.floor(block_start_time + np.arange(block.size) / fs)

        for second in np.unique(sample_seconds):
            row = self._archive_row(second)
            samples = block[sample_seconds == second]
            row[ARC_POWER_SUM] += np.sum(samples ** 2)
            row[ARC_SAMPLE_COUNT] += samples.size
            peak_power = np.max(np.abs(samples)) ** 2
            row[ARC_PEAK_POWER] = max(row[ARC_PEAK_POWER], peak_power)

        if self.keep_spectra:
            recent = self.raw[-spectrum_samples:]
            nfft = int(2 ** np.ceil(np.log2(round(.25 * fs))))
            window = signal.get_window('hann', spectrum_samples, fftbins=True)     # MATLAB hann(n,'periodic')
            frequency, density = signal.periodogram(recent - recent.mean(), fs, window=window, nfft=nfft,
                                                    return_onesided=True, scaling='density', detrend=False)
            # Convert density to power per resolution bandwidth so that 0 dBFS = full-scale sine (DK8OK).
            resolution_bandwidth = fs * np.sum(window ** 2) / np.sum(window) ** 2
            level_db = 10 * np.log10(np.maximum(2 * density * resolution_bandwidth, 1e-15))
            visible = frequency <= 3600
            self.spectra.append((audio_now, frequency[visible], level_db[visible]))

    # ------------------------------------------------------------------ analyse
    def analyse(self, clip=0.0):
        """DK8OK analyse(): acquisition on the last 0.85 s and measurement of every new frame in it."""
        fs = self.fs

        # --- Step 1 and 2: the analysis window as complex baseband at FB, matched-filtered.
        window_samples = round(self.window * fs)
        audio = self.raw[max(0, self.raw.size - window_samples):]
        segment_start = self.raw_start + self.raw.size - audio.size      # absolute index of audio[0]
        z = self._baseband(audio, segment_start)

        # Limits for a PN start (1-based sample numbers, as in MATLAB): keep EDGE_MARGIN clear at the start, and
        # leave room for the 80 sync symbols plus the margin at the end.
        length = z.size
        max_start = length - EDGE_MARGIN - PN_LENGTH * SPS
        now = self.origin + (self.raw_start + self.raw.size) / fs
        if max_start < EDGE_MARGIN + FRAME:
            return

        # --- Step 3: acquisition.
        best_score, best_start, best_f = self._acquire(z, max_start)

        if best_score < LOCK_GATE:
            self._no_lock(now, best_score, best_f, clip)
            return

        # --- Step 4 and 5: measure every frame on the acquired grid inside the window.
        sync_offsets = np.arange(PN_LENGTH) * SPS          # sample offsets of the 80 sync symbols in a frame
        sync_times = sync_offsets / FB                     # the same, in seconds

        # Step back to the earliest frame start in the window, then walk forward one frame at a time.
        start = best_start
        while start - FRAME >= EDGE_MARGIN:
            start -= FRAME
        while start <= max_start:
            absolute_index = round(segment_start * FB / fs) + start     # 1-based, as DK8OK's absindex
            is_new_frame = absolute_index > self.consumed + FRAME / 2
            if is_new_frame:
                sync_samples = z[start - 1 + sync_offsets]               # -1: MATLAB 1-based -> Python 0-based
                self._frame(sync_samples, sync_times, best_f, absolute_index, clip)
                self.consumed = absolute_index
            start += FRAME

        # Forget display points older than the hold time.
        self.cloud = [entry for entry in self.cloud if entry[0] >= now - self.hold]
        self.eq_cloud = [entry for entry in self.eq_cloud if entry[0] >= now - self.hold]
        self._trim_logs()

    def _baseband(self, audio, segment_start):
        """Real audio -> complex baseband at FB with the audio centre at 0 Hz, matched-filtered (steps 1-2)."""
        absolute_sample = segment_start + np.arange(audio.size)
        oscillator = np.exp(-2j * np.pi * self.centre * absolute_sample / self.fs)

        # hilbert() returns x + j*H{x}; doubling it keeps the signal's amplitude as DK8OK does.
        z = 2 * signal.hilbert(audio) * oscillator
        z = signal.resample_poly(z, self.up, self.down)          # MATLAB resample(z, fb, fs)
        z = np.convolve(z, self.h, 'same')                       # MATLAB conv(z, h(:), 'same')
        return z

    def _acquire(self, z, max_start):
        """Search all carrier offsets and sample phases for a PN80 followed by another PN80 one frame later.

        Returns (best score, best start as a 1-based sample number, best carrier offset in Hz).
        """
        best_score = -np.inf
        best_start = 0
        best_f = 0.0

        reversed_pn = self.pn[::-1]          # convolving with the reversed sequence = correlating with it
        sample_number = np.arange(z.size)
        ones = np.ones(PN_LENGTH)

        for f in range(SEARCH_LOW_HZ, SEARCH_HIGH_HZ + 1, SEARCH_STEP_HZ):
            shifted = z * np.exp(-2j * np.pi * f * sample_number / FB)

            for phase in range(1, SPS + 1):                      # DK8OK's q = 1..8
                # One sample per symbol, starting at this sample phase.
                symbols = shifted[phase - 1::SPS]

                # Correlation with PN80 at every symbol position, and the energy of the same 80 symbols.
                correlation = np.convolve(symbols, reversed_pn, 'valid')
                energy = np.convolve(np.abs(symbols) ** 2, ones, 'valid')
                score = np.abs(correlation) / np.sqrt(80 * energy + EPS)     # 1.0 = perfect match

                # 1-based sample number of each candidate start; candidates outside the limits score 0.
                candidate_start = phase + np.arange(score.size) * SPS
                outside = (candidate_start < EDGE_MARGIN) | (candidate_start > max_start)
                score[outside] = 0

                # Require a second PN80 one frame later: the score is the smaller of the two.
                coherent = np.zeros(score.size)
                if score.size > FRAME_SYMBOLS:
                    this_frame = score[:-FRAME_SYMBOLS]
                    next_frame = score[FRAME_SYMBOLS:]
                    coherent[:-FRAME_SYMBOLS] = np.minimum(this_frame, next_frame)

                best_here = int(np.argmax(coherent))
                if coherent[best_here] > best_score:
                    best_score = coherent[best_here]
                    best_start = int(candidate_start[best_here])
                    best_f = float(f)

        return best_score, best_start, best_f

    def _no_lock(self, now, best_score, best_f, clip):
        """Record a failed acquisition and reset the equalizer and displays."""
        self.losses += 1
        self.logdata.append([now, np.nan, best_f, np.nan, best_score, clip, 0, np.nan, 0])
        self.eqdata.append([now, np.nan, np.nan, np.nan, 0, 0])
        self.cloud = []
        self.eq_cloud = []
        self.eqW = None
        self.eq_last = -np.inf
        self.last = None
        candidate_centre = self.centre + best_f
        self.status = ('NO LOCK | corr %.3f / gate 0.480 | best candidate centre %.1f Hz'
                       % (best_score, candidate_centre))
        self._trim_logs()

    def _frame(self, y, t, best_f, absindex, clip):
        """Measure one PN80 (step 4), then test and retrain the equalizer (step 5).

        y: the 80 received sync samples; t: their times (s) from the first; best_f: acquisition offset (Hz);
        absindex: 1-based absolute index of the first sync sample at FB.
        """
        pn = self.pn

        # --- Carrier offset.  Multiplying by the known PN removes the BPSK modulation ("PN-stripped").
        yd = y * pn * np.exp(-2j * np.pi * best_f * t)
        # DK8OK: straight-line fit to the unwrapped phase; its slope (rad/s) is the residual offset.
        unwrapped_phase = np.unwrap(np.angle(yd))
        slope = np.polyfit(t, unwrapped_phase, 1)[0]
        f = best_f + slope / (2 * np.pi)
        if self.cfo_method == 'guarded':                  # ours, off by default
            if pn_correlation(y, pn, t, f) < pn_correlation(y, pn, t, best_f):
                f = best_f + ml_offset(yd, t)
                self.cfo_fallbacks += 1

        # --- Remove the offset, then the mean channel phase and amplitude ("aligned" symbols ideally = PN).
        yc = y * np.exp(-2j * np.pi * f * t)
        channel = np.mean(yc * pn)                        # complex channel gain at the PN start
        amplitude = max(abs(channel), EPS)
        aligned = yc * np.exp(-1j * np.angle(channel)) / amplitude

        correlation = abs(np.sum(yc * pn)) / np.sqrt(80 * np.sum(np.abs(yc) ** 2) + EPS)
        tm = self.origin + absindex / FB                  # elapsed time of this frame (s)

        accepted = correlation >= LOCK_GATE and abs(f) <= MAX_CFO_HZ
        if not accepted:
            self.logdata.append([tm, np.nan, np.nan, np.nan, correlation, clip, 0, np.nan, 0])
            self.eqdata.append([tm, np.nan, np.nan, np.nan, 0, 0])
            self.eqW = None
            self.eq_last = -np.inf
            return

        # --- PN bit errors (hard decisions), error-vector magnitude, channel phase.
        decoded = aligned.real >= 0                       # received bit: True = 1
        reference = pn > 0
        errors = int(np.sum(decoded != reference))
        ber = errors / 80
        evm = float(np.sqrt(np.mean(np.abs(aligned - pn) ** 2)))
        phase = float(np.angle(channel))                  # channel phase at PN start, modulo 2 pi
        self.logdata.append([tm, ber, f, phase, correlation, clip, 1, evm, 0])
        self.cloud.append((tm, aligned))

        # --- Equalizer.  Inner symbols only: the FIR needs `half` neighbours on each side.
        n_taps = self.taps
        half = (n_taps - 1) // 2
        inner = np.arange(half, 80 - half)                # 0-based indices of the inner symbols
        # Row r of X holds the n_taps aligned symbols centred on inner symbol r (the FIR's input).
        tap_offsets = np.arange(-half, half + 1)
        X = aligned[inner[:, None] + tap_offsets[None, :]]
        target = pn[inner]
        compared = inner.size
        before = float(np.sum(decoded[inner] != (target > 0)) / compared)

        # Taps are usable if they exist, have the right length and come from the frame just before this one.
        ready = (self.eq_enable and self.eqW is not None and self.eqW.size == n_taps
                 and tm - self.eq_last < EQ_MAX_AGE_S)
        after_ber = np.nan
        after_evm = np.nan
        eq_bits = np.full(80, np.nan)
        output = None
        if ready:
            output = X @ self.eqW                         # equalizer output for the inner symbols
            after_bits = output.real >= 0
            eq_bits[inner] = after_bits
            after_ber = float(np.sum(after_bits != (target > 0)) / compared)
            after_evm = float(np.sqrt(np.mean(np.abs(output - target) ** 2)))
            self.eq_cloud.append((tm, output))
        self.eqdata.append([tm, before, after_ber, after_evm, compared, int(ready)])

        # Retrain AFTER this frame's comparison: ridge regression  W = (X^H X + lambda I)^-1 X^H target,
        # with lambda = ridge x mean tap energy.
        if self.eq_enable:
            XH = X.conj().T
            gram = XH @ X
            ridge_lambda = self.ridge * np.real(np.trace(gram)) / n_taps
            self.eqW = np.linalg.solve(gram + ridge_lambda * np.eye(n_taps), XH @ target)
            self.eq_last = tm
        else:
            self.eqW = None
            self.eq_last = -np.inf

        # --- Material for displays.
        if self.keep_points:
            if output is None:
                output_copy = None
            else:
                output_copy = output.copy()
            self.points.append((tm, aligned.copy(), output_copy, y.copy()))
        if self.eqW is None:
            taps_copy = None
        else:
            taps_copy = self.eqW.copy()
        self.last = dict(time=tm, received=y / amplitude, aligned=aligned,
                         decoded=decoded.astype(int), eq_bits=eq_bits, tested=np.isin(np.arange(80), inner),
                         ber=ber, err=errors, corr=correlation, cfo=f, phase=phase, before=before,
                         after=after_ber, eq_evm=after_evm, ready=ready, taps=taps_copy)

        if ready:
            eq_text = 'EQ BER %.2f -> %.2f%% on %d bits | EQ EVM %.1f%%' % (100 * before, 100 * after_ber,
                                                                           compared, 100 * after_evm)
        elif self.eq_enable:
            eq_text = 'EQ training / awaiting next consecutive frame'
        else:
            eq_text = 'EQ OFF'
        self.status = 'LOCK | raw PN80 %.2f%% (%d/80) | %s | corr %.3f' % (100 * ber, errors, eq_text,
                                                                           correlation)

    def _trim_logs(self):
        """Keep at most LOG_LIMIT rows in each log."""
        if len(self.logdata) > LOG_LIMIT:
            self.logdata = self.logdata[-LOG_LIMIT:]
        if len(self.eqdata) > LOG_LIMIT:
            self.eqdata = self.eqdata[-LOG_LIMIT:]

    # ------------------------------------------------------------------ archive
    def _collect_archive(self):
        """DK8OK collectArchive(): add log rows not yet archived to their 1-second archive rows."""
        fresh_rows = [row for row in self.logdata if row[LOG_TIME] > self.archive_last_frame]
        if not fresh_rows:
            return

        if self.eqdata:
            eq_times = np.array([eq_row[EQ_TIME] for eq_row in self.eqdata])
        else:
            eq_times = np.zeros(0)

        for row in fresh_rows:
            archive_row = self._archive_row(np.floor(row[LOG_TIME]))
            archive_row[ARC_ALL_FRAMES] += 1
            archive_row[ARC_CAPTURE_GAPS] += row[LOG_CAPTURE_GAP]
            if row[LOG_LOCKED] > 0:
                archive_row[ARC_LOCKED_FRAMES] += 1
                archive_row[ARC_CFO_SUM] += row[LOG_CFO]
                archive_row[ARC_CFO_COUNT] += 1
            if np.isfinite(row[LOG_CORRELATION]):
                archive_row[ARC_CORRELATION_SUM] += row[LOG_CORRELATION]
                archive_row[ARC_CORRELATION_COUNT] += 1

            # The equalizer row written for the same frame (same time), if any: its before/after BER.
            matching = np.nonzero(np.abs(eq_times - row[LOG_TIME]) < 1e-7)[0]
            if matching.size:
                eq_row = self.eqdata[matching[-1]]
                if np.isfinite(eq_row[EQ_BER_BEFORE]):
                    archive_row[ARC_BER_BEFORE_SUM] += eq_row[EQ_BER_BEFORE]
                    archive_row[ARC_BER_BEFORE_COUNT] += 1
                if np.isfinite(eq_row[EQ_BER_AFTER]):
                    archive_row[ARC_BER_AFTER_SUM] += eq_row[EQ_BER_AFTER]
                    archive_row[ARC_BER_AFTER_COUNT] += 1

        self.archive_last_frame = max(row[LOG_TIME] for row in fresh_rows)

    def archive_values(self):
        """DK8OK archiveValues(): per-second means, columns ARCHIVE_COLUMNS.

        Seconds that contain a capture gap have their measurements blanked (NaN).
        """
        if not self.archive:
            return np.zeros((0, 9))
        sums = np.array([self.archive[second] for second in sorted(self.archive)])

        values = np.full((sums.shape[0], 9), np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):      # 0/0 -> NaN is intended
            values[:, 0] = sums[:, ARC_SECOND] + .5                                   # middle of the second
            values[:, 1] = 100 * sums[:, ARC_BER_BEFORE_SUM] / sums[:, ARC_BER_BEFORE_COUNT]   # percent
            values[:, 2] = 100 * sums[:, ARC_BER_AFTER_SUM] / sums[:, ARC_BER_AFTER_COUNT]
            values[:, 3] = sums[:, ARC_CFO_SUM] / sums[:, ARC_CFO_COUNT]
            mean_power = sums[:, ARC_POWER_SUM] / sums[:, ARC_SAMPLE_COUNT]
            values[:, 4] = 10 * np.log10(np.maximum(mean_power, 1e-15))               # RMS level, dBFS
            values[sums[:, ARC_SAMPLE_COUNT] == 0, 4] = np.nan
            values[:, 5] = 10 * np.log10(np.maximum(sums[:, ARC_PEAK_POWER], 1e-15))  # peak level, dBFS
            values[sums[:, ARC_SAMPLE_COUNT] == 0, 5] = np.nan
            values[:, 6] = sums[:, ARC_LOCKED_FRAMES] / sums[:, ARC_ALL_FRAMES]
            values[:, 7] = sums[:, ARC_CORRELATION_SUM] / sums[:, ARC_CORRELATION_COUNT]
            values[:, 8] = sums[:, ARC_CAPTURE_GAPS]

        has_gap = values[:, 8] > 0
        values[has_gap, 1:6] = np.nan
        return values

    # ------------------------------------------------------------------ export
    def joined_rows(self):
        """DK8OK saveCSV(): each log row followed by the equalizer row with the same time, sorted by time.

        Equivalent to his outerjoin(..., 'Type', 'left'): log rows without an equalizer row get NaNs.
        """
        eq_rows_by_time = {}
        for eq_row in self.eqdata:
            eq_rows_by_time.setdefault(eq_row[EQ_TIME], []).append(eq_row)

        empty_eq_values = [np.nan] * 5
        rows = []
        for log_row in self.logdata:
            matches = eq_rows_by_time.get(log_row[LOG_TIME])
            if not matches:
                matches = [[log_row[LOG_TIME]] + empty_eq_values]
            for eq_row in matches:
                rows.append(list(log_row) + list(eq_row[1:]))

        rows.sort(key=row_time)                  # by elapsed time; rows with equal times keep their order
        return rows
