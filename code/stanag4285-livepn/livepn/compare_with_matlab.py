"""
compare_with_matlab -- compare the PN measurements CSV of DK8OK's MATLAB LIVE PN with the CSV of this Python port.

The test it is made for
  1. In the MATLAB program: run LIVE PN on a STANAG 4285 signal as usual, press Stop, then "Save CSV"
     (DK8OK_LIVE_PN.csv) and "Save WAV" (DK8OK_LIVE_PN_INPUT.wav, the last 10 s of audio it analysed).
  2. Run the Python port on that WAV:      python livepn/livepn_offline.py DK8OK_LIVE_PN_INPUT.wav
     which writes DK8OK_LIVE_PN_INPUT_DK8OK_LIVE_PN.csv (same columns as the MATLAB file).
  3. Compare the two:
         python livepn/compare_with_matlab.py DK8OK_LIVE_PN.csv DK8OK_LIVE_PN_INPUT_DK8OK_LIVE_PN.csv

What it does
  * The MATLAB file covers the whole run; the WAV only its last 10 s.  Times in the MATLAB file count from the start
    of the run, times in the Python file from the start of the WAV.  The difference (the WAV's start time within
    the run) is found from the locked frames:
      - because the WAV is the END of the run, only start times that put the two files' last frames within a
        couple of seconds of each other are considered (--anywhere searches the whole run);
      - PN frames come every 106.7 ms, so several start times, one frame apart, line the frames up equally well.
        Among those, the one where the measured values agree best, relative to how much they normally change from
        frame to frame, is taken.  If the values cannot tell the candidates apart (a very steady signal), the
        report says "ambiguous" and lists them.  --offset gives the start time directly.
  * Locked frames are paired one to one (the MATLAB frame nearest in time, within --tolerance-ms).  Rows that are
    not locked frames -- failed acquisitions and capture gaps -- are not paired; they are counted separately.
  * For each measurement column it reports how many pairs agree within the normal variation (see below), the
    median and the largest difference, then lists the pairs side by side.  Frames that only one program measured
    are listed too.
  * A MATLAB capture gap (its recorder is renewed after each minute of audio) inside the span of the WAV is
    reported: across a renewal the WAV and the MATLAB times no longer share one offset.

What to expect
  The two programs see the same audio but do not cut it into the same analysis windows: the MATLAB timer delivers
  audio in pieces whose length depends on the computer, the Python port in exact 0.2-s blocks.  Where a window
  starts moves V1.4's resampled sample grid by a fraction of a sample, so the same frame measured from two
  different windows -- by either program, or by the same program twice -- differs a little (measured on our own
  program run twice with different pieces: carrier offset by 0.01-0.02 Hz typically, 0.13 Hz at most; correlation
  by about 0.001; EVM by up to a few hundredths).  Pairs within those amounts "agree"; the bit error rates
  should be identical.  The frames near the start of the WAV have less history (the equalizer needs the frame
  before).  The fitted phase is relative to each program's own time origin, so a constant phase difference is
  expected; the report removes it and shows what is left.

Exit status: 0 when a report was produced (whatever it shows), 1 when the files could not be compared.

Claude (AI assistant) for Mike Burns, AB1LD, 2026-10-13 (revised 2026-10-14 after reviews by GPT-6 Astra and
Claude Fable).  No DK8OK code; the columns are those of his "Save CSV".
"""
import argparse
import csv
import io
import math
import sys

import numpy as np

FRAME_SECONDS = 256 / 2400.0           # one STANAG 4285 frame: 256 symbols at 2400 Bd
PRIOR_WINDOW_S = 2.0                   # how far from "the WAV ends where the run ends" the start time may lie
MIN_PAIRED_FRACTION = 0.5              # a candidate start time must pair at least this share of the Python frames
AMBIGUITY_RATIO = 1.5                  # the best candidate must beat the next by this factor to be "clear"

# Columns compared, with the difference that still counts as agreement (in the column's unit).  The bit error
# rates are counts of whole bits, so they should agree exactly; the other limits are the normal variation of
# V1.4's estimator between two analysis windows (see "What to expect" above).
COMPARED_COLUMNS = [
    ('PN_BER', 1e-9),
    ('CFO_Hz', 0.2),
    ('FittedPhase_rad', 0.03),
    ('Correlation', 0.01),
    ('EVM', 0.05),
    ('BER_before_same_bits', 1e-9),
    ('BER_after_EQ', 1e-9),
    ('EVM_after_EQ', 0.15),
]
PHASE_COLUMN = 'FittedPhase_rad'
# Columns used to tell candidate start times apart (they change from frame to frame with noise and fading).
SCORING_COLUMNS = ['Correlation', 'CFO_Hz', 'EVM', 'PN_BER']
REQUIRED_COLUMNS = ['Elapsed_s', 'Locked', 'CaptureGap']


# ---------------------------------------------------------------------------------------------- reading
def read_text(path):
    """The file's text.  Python writes UTF-8; MATLAB's writetable writes the system encoding, which on Windows is
    often Windows-1252.  Returns (text, encoding used)."""
    with open(path, 'rb') as handle:
        data = handle.read()
    try:
        return data.decode('utf-8-sig'), 'UTF-8'
    except UnicodeDecodeError:
        return data.decode('cp1252', errors='replace'), 'Windows-1252'


def read_measurements(path):
    """The rows of a LIVE PN measurements CSV as a dict: column name -> numpy array (NaN for empty or NaN)."""
    try:
        text, encoding = read_text(path)
    except OSError as error:
        raise SystemExit(f'cannot read {path}: {error}')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    rows = list(reader)
    names = reader.fieldnames or []
    missing = [name for name in REQUIRED_COLUMNS if name not in names]
    if missing:
        raise SystemExit(f'{path}: missing column(s) {", ".join(missing)} -- is this a LIVE PN "Save CSV" file?')
    columns = {}
    for name in names:
        values = []
        for row in rows:
            values.append(to_number(row.get(name)))
        columns[name] = np.array(values, dtype=float)
    if encoding != 'UTF-8':
        print(f'note: {path} is not UTF-8; read as {encoding} (only the Note column could be affected)')
    return columns


def to_number(text):
    """A CSV field as a float; empty, NaN or text -> NaN."""
    if text is None:
        return math.nan
    text = text.strip()
    if text == '':
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def locked_rows(columns):
    """Row numbers of the locked frames, in time order."""
    times = columns['Elapsed_s']
    is_locked = (columns['Locked'] == 1) & (columns['CaptureGap'] != 1) & np.isfinite(times)
    rows = np.flatnonzero(is_locked)
    return rows[np.argsort(times[rows], kind='stable')]


# ---------------------------------------------------------------------------------------------- pairing
def pair_frames(matlab, python, matlab_rows, python_rows, offset, tolerance_s):
    """Pair locked frames one to one, in time order: each Python frame with the nearest unused MATLAB frame within
    the tolerance.  Returns (pairs, MATLAB rows left over within the Python span, Python rows left over)."""
    matlab_times = matlab['Elapsed_s'][matlab_rows]
    python_times = python['Elapsed_s'][python_rows] + offset
    pairs = []
    python_only = []
    used = np.zeros(matlab_rows.size, bool)
    start = 0
    for index, time in enumerate(python_times):
        while start < matlab_times.size and matlab_times[start] < time - tolerance_s:
            start += 1
        best = None
        candidate = start
        while candidate < matlab_times.size and matlab_times[candidate] <= time + tolerance_s:
            if not used[candidate]:
                if best is None or abs(matlab_times[candidate] - time) < abs(matlab_times[best] - time):
                    best = candidate
            candidate += 1
        if best is None:
            python_only.append(python_rows[index])
        else:
            used[best] = True
            pairs.append((matlab_rows[best], python_rows[index]))
    matlab_only = []
    if python_times.size:
        first = python_times[0] - tolerance_s
        last = python_times[-1] + tolerance_s
        for position in range(matlab_rows.size):
            if not used[position] and first <= matlab_times[position] <= last:
                matlab_only.append(matlab_rows[position])
    return pairs, matlab_only, python_only


# ---------------------------------------------------------------------------------------------- start time
FRAMES_APART = 8                       # about one analysis (0.8 s) apart, so usually from different windows


def frame_to_frame_scale(columns, rows, name):
    """Typical difference of a column between two locked frames FRAMES_APART frames apart (median absolute
    difference); 0 if the column does not change."""
    values = columns[name][rows]
    values = values[np.isfinite(values)]
    if values.size < FRAMES_APART + 3:
        return 0.0
    return float(np.median(np.abs(values[FRAMES_APART:] - values[:-FRAMES_APART])))


def values_score(matlab, python, pairs, scales):
    """How well the paired values agree.  For each informative column: the mean absolute difference over the
    pairs divided by that column's normal frame-to-frame change.  Pairing the wrong frames gives about 1 in every
    column; the right frames give clearly less than 1 in at least one column (in practice the carrier offset,
    which changes from frame to frame much more than between two analysis windows).  The score is the smallest
    of the column ratios; NaN if no column is informative."""
    ratios = []
    for name, scale in scales.items():
        if scale <= 0:
            continue
        differences = []
        for matlab_row, python_row in pairs:
            a = matlab[name][matlab_row]
            b = python[name][python_row]
            if np.isfinite(a) and np.isfinite(b):
                differences.append(abs(a - b))
        if differences:
            ratios.append(np.mean(differences) / scale)
    if not ratios:
        return math.nan
    return float(min(ratios))


def candidate_start_time(candidate):
    """Sort key: a candidate's start time.  (A candidate is (start time, pairs, score).)"""
    return candidate[0]


def candidate_rank(candidate):
    """Sort key: best candidate first -- the lowest score, then the most pairs.  No score counts as worst."""
    score = candidate[2]
    if math.isnan(score):
        score = math.inf
    pair_count = len(candidate[1])
    return (score, -pair_count)


def find_offset(matlab, python, matlab_rows, python_rows, tolerance_s, anywhere):
    """Candidate start times of the WAV within the MATLAB run, best first, as (offset, pairs, score).

    Step 1, timing: every (MATLAB frame, Python frame) pair suggests a start time; suggestions are counted in bins
    of the tolerance.  Unless `anywhere`, only start times that put the two files' last frames within
    PRIOR_WINDOW_S of each other are kept (the WAV is the end of the run).
    Step 2, frame count: the MATLAB run analysed all of the WAV's audio, so the right start time pairs (nearly)
    all the Python frames; candidates pairing more than 2 frames fewer than the best are dropped.
    Step 3, values: the remaining candidates are scored by values_score.
    """
    matlab_times = matlab['Elapsed_s'][matlab_rows]
    python_times = python['Elapsed_s'][python_rows]
    suggestions = (matlab_times[:, None] - python_times[None, :]).ravel()
    if not anywhere:
        expected = np.nanmax(matlab['Elapsed_s']) - np.nanmax(python['Elapsed_s'])
        suggestions = suggestions[np.abs(suggestions - expected) <= PRIOR_WINDOW_S]
    if suggestions.size == 0:
        return []
    bins = np.round(suggestions / tolerance_s).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    strong = values[counts >= 0.5 * counts.max()]

    scales = {}
    for name in SCORING_COLUMNS:
        if name in matlab and name in python:
            scales[name] = frame_to_frame_scale(matlab, matlab_rows, name)

    candidates = []
    for value in strong:
        near = suggestions[np.abs(bins - value) <= 1]
        offset = float(np.median(near))
        pairs, _, _ = pair_frames(matlab, python, matlab_rows, python_rows, offset, tolerance_s)
        if len(pairs) < max(3, MIN_PAIRED_FRACTION * python_rows.size):
            continue
        candidates.append((offset, pairs, values_score(matlab, python, pairs, scales)))

    # Merge candidates that are really the same start time (neighbouring bins).
    candidates.sort(key=candidate_start_time)
    merged = []
    for candidate in candidates:
        if merged and abs(candidate[0] - merged[-1][0]) < 0.5 * FRAME_SECONDS:
            if len(candidate[1]) > len(merged[-1][1]):
                merged[-1] = candidate
        else:
            merged.append(candidate)

    if merged:
        most_pairs = max(len(candidate[1]) for candidate in merged)
        merged = [candidate for candidate in merged if len(candidate[1]) >= most_pairs - 2]

    merged.sort(key=candidate_rank)
    return merged


# ---------------------------------------------------------------------------------------------- report
def wrapped(angle):
    """Angle(s) wrapped into -pi .. pi."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def column_report(name, matlab_values, python_values, agree_within):
    """One line: pairs with a value in both, how many agree, median and largest difference."""
    both = np.isfinite(matlab_values) & np.isfinite(python_values)
    one_only = int(np.sum(np.isfinite(matlab_values) != np.isfinite(python_values)))
    if not np.any(both):
        text = f'  {name:22s} no pairs with a value in both files'
        if one_only:
            text += f'  ({one_only} with a value in one only)'
        return text
    difference = python_values[both] - matlab_values[both]
    note = ''
    if name == PHASE_COLUMN:
        difference = wrapped(difference)
        constant = float(np.angle(np.mean(np.exp(1j * difference))))
        difference = wrapped(difference - constant)
        note = f'  (after removing a constant {constant:+.3f} rad)'
    agree = int(np.sum(np.abs(difference) <= agree_within))
    text = (f'  {name:22s} {agree:4d} of {both.sum():4d} agree (within {agree_within:g});'
            f'  median |diff| {np.median(np.abs(difference)):.3g},  largest {np.max(np.abs(difference)):.3g}')
    if one_only:
        text += f';  {one_only} with a value in one file only'
    return text + note


def value_text(columns, name, row, width, digits):
    """One value for the side-by-side list, right-aligned in `width` characters; '-' if missing."""
    if name not in columns or not np.isfinite(columns[name][row]):
        return f'{"-":>{width}}'
    return f'{columns[name][row]:{width}.{digits}f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('matlab_csv', help="the MATLAB program's DK8OK_LIVE_PN.csv")
    parser.add_argument('python_csv', help="the Python port's CSV for the saved WAV")
    parser.add_argument('--offset', type=float, help='start of the WAV within the MATLAB run, s (default: found)')
    parser.add_argument('--anywhere', action='store_true',
                        help='search the whole MATLAB run, not only near its end (for a WAV that is not the end)')
    parser.add_argument('--tolerance-ms', type=float, default=5.0, help='frames this close in time are paired (5)')
    parser.add_argument('--rows', type=int, default=40, help='pairs listed side by side (40; 0 = none)')
    args = parser.parse_args(argv)
    if not (args.tolerance_ms > 0 and math.isfinite(args.tolerance_ms)):
        raise SystemExit('--tolerance-ms must be a positive number')
    tolerance_s = args.tolerance_ms / 1000.0

    matlab = read_measurements(args.matlab_csv)
    python = read_measurements(args.python_csv)
    matlab_rows = locked_rows(matlab)
    python_rows = locked_rows(python)
    for label, path, columns, rows in (('MATLAB', args.matlab_csv, matlab, matlab_rows),
                                        ('Python', args.python_csv, python, python_rows)):
        times = columns['Elapsed_s'][np.isfinite(columns['Elapsed_s'])]
        if times.size == 0:
            print(f'{label}: {path} has no measurements.')
            print('Nothing to compare (a recording shorter than about 1 s gives no measurements).')
            return 1
        print(f'{label}: {path}  ({times.size} rows, {rows.size} locked frames, '
              f'{times.min():.2f} .. {times.max():.2f} s)')
    if matlab_rows.size == 0 or python_rows.size == 0:
        print('No locked frames in one of the files: nothing to compare.')
        return 1

    # ---- the WAV's start time within the MATLAB run
    if args.offset is not None:
        offset = args.offset
        print(f'The WAV starts {offset:.4f} s into the MATLAB run (given)')
    else:
        candidates = find_offset(matlab, python, matlab_rows, python_rows, tolerance_s, args.anywhere)
        if not candidates:
            if args.anywhere:
                print('No start time lines up the locked frames of the two files.')
            else:
                print('No start time near the end of the MATLAB run lines up the frames.  Is the WAV the end of '
                      'this run?  (Save CSV and Save WAV right after Stop; --anywhere searches the whole run.)')
            return 1
        offset, _, score = candidates[0]
        if len(candidates) > 1:
            runner_up = candidates[1][2]
        else:
            runner_up = math.nan
        # A clear match: a good score, and the next-best candidate clearly worse (or none).
        good_score = not math.isnan(score) and score < 0.6
        no_close_rival = math.isnan(runner_up) or runner_up >= AMBIGUITY_RATIO * score
        if good_score and no_close_rival:
            print(f'The WAV starts {offset:.4f} s into the MATLAB run (found: the paired values differ by '
                  f'{score:.2f} of their normal frame-to-frame change; '
                  f'the next-best start time scores {runner_up:.2f})')
        else:
            print(f'The WAV starts {offset:.4f} s into the MATLAB run -- AMBIGUOUS: the measured values do not '
                  f'single out one start time.')
            print('  PN frames come every 106.7 ms, and on a very steady signal the frames look alike, so several '
                  'start times one or more frames apart fit equally well.  Candidates, best first '
                  '(score: paired differences relative to the normal frame-to-frame change; below about 0.5 is a '
                  'clear match):')
            for candidate_offset, candidate_pairs, candidate_score in candidates[:5]:
                print(f'    {candidate_offset:10.4f} s   {len(candidate_pairs)} frames paired   '
                      f'score {candidate_score:.2f}')
            print('  The report below uses the first.  If the signal really is that steady, the agreement figures '
                  'are still meaningful (neighbouring frames carry nearly the same values); --offset chooses one.')

    # ---- a MATLAB recorder renewal inside the WAV?
    python_times = python['Elapsed_s'][python_rows] + offset
    gap_times = matlab['Elapsed_s'][(matlab['CaptureGap'] == 1) & np.isfinite(matlab['Elapsed_s'])]
    inside = gap_times[(gap_times >= python_times[0] - 1.0) & (gap_times <= python_times[-1] + 1.0)]
    for gap in inside:
        print(f'  WARNING: the MATLAB run has a capture gap (recorder renewal) at {gap:.2f} s, inside the span of '
              f'the WAV.  Across it the WAV and the MATLAB times do not share one offset, so frames on one side '
              f'will not pair.  Stop MATLAB so that its last 10 s contain no renewal (see the README).')

    # ---- pairing and the report
    pairs, matlab_only, python_only = pair_frames(matlab, python, matlab_rows, python_rows, offset, tolerance_s)
    print(f'\nLocked frames paired: {len(pairs)}   locked only in MATLAB (same time span): {len(matlab_only)}   '
          f'locked only in Python: {len(python_only)}')
    if not pairs:
        print('No frames could be paired: check that the WAV and the CSV come from the same run.')
        return 1

    paired_matlab = np.array([pair[0] for pair in pairs])
    paired_python = np.array([pair[1] for pair in pairs])
    time_difference_ms = 1000 * (python['Elapsed_s'][paired_python] + offset - matlab['Elapsed_s'][paired_matlab])
    print(f'Frame time differences: median {np.median(np.abs(time_difference_ms)):.3f} ms, '
          f'largest {np.max(np.abs(time_difference_ms)):.3f} ms')

    print('\nMeasurements (Python minus MATLAB); "agree" means within the normal window-to-window variation:')
    for name, agree_within in COMPARED_COLUMNS:
        if name in matlab and name in python:
            print(column_report(name, matlab[name][paired_matlab], python[name][paired_python], agree_within))
        else:
            print(f'  {name:22s} not in both files')

    if args.rows:
        print(f'\nFirst {min(args.rows, len(pairs))} pairs (time in the MATLAB run):')
        print(f'  {"time_s":>9}  {"PN_BER M":>8} {"P":>8}   {"CFO_Hz M":>9} {"P":>9}   '
              f'{"BER_EQ M":>8} {"P":>8}')
        for matlab_row, python_row in pairs[:args.rows]:
            time_text = f'{matlab["Elapsed_s"][matlab_row]:9.3f}'
            ber_text = (value_text(matlab, 'PN_BER', matlab_row, 8, 4) + ' '
                        + value_text(python, 'PN_BER', python_row, 8, 4))
            cfo_text = (value_text(matlab, 'CFO_Hz', matlab_row, 9, 3) + ' '
                        + value_text(python, 'CFO_Hz', python_row, 9, 3))
            eq_text = (value_text(matlab, 'BER_after_EQ', matlab_row, 8, 4) + ' '
                       + value_text(python, 'BER_after_EQ', python_row, 8, 4))
            print(f'  {time_text}  {ber_text}   {cfo_text}   {eq_text}')

    for label, rows, columns, shift in (('locked only in MATLAB', matlab_only, matlab, 0.0),
                                        ('locked only in Python', python_only, python, offset)):
        if rows:
            time_texts = []
            for row in rows[:20]:
                time_texts.append(f'{columns["Elapsed_s"][row] + shift:.3f}')
            if len(rows) > 20:
                more = f' ... ({len(rows)} in all)'
            else:
                more = ''
            print(f'\nFrames {label} (time in the MATLAB run, s): {", ".join(time_texts)}{more}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
