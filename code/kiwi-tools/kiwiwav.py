"""Shared reader for KiwiSDR IQ wav files (kiwirecorder -w, GPS-stamped). [2026-10-01, after Codex audit]

Written by Claude (Anthropic AI) for Mike Burns, AB1LD.  File format as written by kiwirecorder (kiwiclient,
John Seamons, jks-prv).  2026-10-12: rewritten for readability (one step per line, descriptive names, comments);
names, attributes and results unchanged, verified bit-identical on a clean and a deliberately damaged file.
The previous version is in this repository's history.

File layout:
    RIFF header (12 bytes) + fmt chunk (24 bytes), then N records of 2074 bytes, starting at byte 36:
        'kiwi' chunk: 8-byte chunk header + 10 bytes (last GPS solution, dummy, GPS seconds of week u32, GPS ns u32)
        'data' chunk: 8-byte chunk header + 2048 bytes = 512 complex int16 samples (I, Q interleaved)
    Every record carries the GPS time of its first sample.

Fixes vs the earlier per-script copies:
  * GPS week derived from the recording start in the FILENAME (YYYYMMDDTHHMMSSZ), not hard-coded; GPS-UTC = 18 s
    (valid since 2017-01-01; checked against the filename start, which must agree within 120 s).
  * Every sample time comes from ITS OWN block stamp (t_block + j/fs), never extrapolated from block 0.
  * Non-monotonic / corrupt / gapped stamps are detected; windows that touch one are flagged bad.
  * Seek uses a sparse monotonic index (every STRIDE blocks) built once and cached next to the wav (.idx.npz).
"""
import datetime as dt
import os
import re
import struct

import numpy as np

# ---------------------------------------------------------------------------------------------- file layout
BLK = 2074                     # bytes per record ('kiwi' chunk + 'data' chunk)
NS = 512                       # complex samples per record
OFF0 = 36                      # byte offset of the first record (after the RIFF and fmt headers)
GPS_UTC = 18.0                 # GPS time - UTC (s), valid since 2017-01-01
GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
FS_NOMINAL = 11998.9554        # measured 2026-09-30/10-01 (header says 11999)

STAMP_OFFSET = 10              # GPS seconds-of-week and nanoseconds start 10 bytes into a record
DATA_OFFSET = 26               # samples start 26 bytes into a record (18-byte 'kiwi' chunk + 8-byte 'data' header)
SECONDS_PER_WEEK = 604800


def _stamp_seconds(week0_unix, gps_seconds, gps_nanoseconds, t_name):
    """Unix time of a record from its GPS seconds-of-week and nanoseconds (NaN when the stamp is missing, i.e. 0).

    A recording that crosses a GPS-week boundary has small seconds-of-week after the boundary; those are moved
    into the next week.
    """
    if gps_seconds == 0:
        return np.nan
    t = week0_unix + gps_seconds + gps_nanoseconds * 1e-9
    if t < t_name - 3600:
        t += SECONDS_PER_WEEK
    return t


def _samples_of_record(record_bytes):
    """The 512 (I, Q) int16 pairs of one record, as an array of shape (512, 2)."""
    data = record_bytes[DATA_OFFSET:DATA_OFFSET + 4 * NS]
    return np.frombuffer(data, '<i2').reshape(-1, 2)


class KiwiWav:
    """A kiwirecorder IQ file, opened for random access by GPS time.

    Attributes: path, fs (sample rate used to place samples between stamps), f (open file), nb (number of records),
    t_name (Unix time of the start stamp in the file name), week0_unix (Unix time of the start of the GPS week),
    ik / it (sparse index: record numbers and their Unix times).
    """

    def __init__(self, path, fs=FS_NOMINAL, stride=256):
        self.path = path
        self.fs = fs

        # Number of whole records in the file.
        self.f = open(path, 'rb')
        self.f.seek(0, 2)
        file_size = self.f.tell()
        self.nb = (file_size - OFF0) // BLK

        # Start time from the file name.
        match = re.search(r'(\d{8}T\d{6})Z', os.path.basename(path))
        if not match:
            raise ValueError('filename lacks YYYYMMDDTHHMMSSZ start stamp: ' + path)
        start = dt.datetime.strptime(match.group(1), '%Y%m%dT%H%M%S').replace(tzinfo=dt.timezone.utc)
        self.t_name = start.timestamp()

        # Unix time at which the GPS week containing the file-name start began.
        gps_now = self.t_name - GPS_EPOCH.timestamp() + GPS_UTC
        whole_weeks = gps_now // SECONDS_PER_WEEK
        self.week0_unix = GPS_EPOCH.timestamp() + whole_weeks * SECONDS_PER_WEEK - GPS_UTC

        self._index(stride)

    # ------------------------------------------------------------------ stamps
    def _raw_stamp(self, k):
        """(GPS seconds of week, nanoseconds) of record k, or None if the stamp is missing (seconds = 0)."""
        self.f.seek(OFF0 + k * BLK)
        chunk = self.f.read(18)
        if chunk[:4] != b'kiwi':
            raise ValueError(f'layout break at block {k}')
        gps_seconds, gps_nanoseconds = struct.unpack('<II', chunk[10:18])
        if gps_seconds == 0:
            return None
        return (gps_seconds, gps_nanoseconds)

    def stamp(self, k):
        """Unix time of the first sample of record k (NaN if its stamp is missing)."""
        raw = self._raw_stamp(k)
        if raw is None:
            return np.nan
        t = self.week0_unix + raw[0] + raw[1] * 1e-9
        if t < self.t_name - 3600:
            t += SECONDS_PER_WEEK                              # recording crossed a GPS-week boundary
        return t

    def _index(self, stride):
        """Sparse index: the time of every stride-th record (from record 1), cached in <wav>.idx<stride>.npz.

        Entries with missing stamps, or more than an hour away from where a steady clock would put them, are
        dropped (e.g. a corrupt +-68,613 s pair seen in practice).
        """
        cache = self.path + f'.idx{stride}.npz'
        cache_is_current = os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(self.path)
        if cache_is_current:
            saved = np.load(cache)
            self.ik = saved['k']
            self.it = saved['t']
        else:
            record_numbers = np.arange(1, self.nb, stride)
            times = np.array([self.stamp(int(k)) for k in record_numbers])
            self.ik = record_numbers
            self.it = times
            try:
                np.savez(cache, k=record_numbers, t=times)
            except OSError:
                pass                                           # read-only folder: work without a cache

        first = self.it[np.isfinite(self.it)][0]
        if abs(first - self.t_name) > 120:
            raise ValueError(f'GPS stamps ({first:.0f}) disagree with filename start ({self.t_name:.0f}) by >120 s')

        expected = first + (self.ik - self.ik[0]) * NS / self.fs
        sensible = np.isfinite(self.it) & (np.abs(self.it - expected) < 3600)
        self.ik = self.ik[sensible]
        self.it = self.it[sensible]

    def find_block(self, t):
        """Last block whose stamp <= t (sparse index + local scan; robust to isolated bad stamps)."""
        # Start from the last index entry before t ...
        j = int(np.searchsorted(self.it, t)) - 1
        start = int(self.ik[max(j, 0)])
        best = start
        # ... and scan forward record by record (at most 2 index strides + 2), stopping at the first stamp after t.
        for k in range(start, min(start + 2 * 256 + 2, self.nb)):
            s = self.stamp(k)
            if np.isfinite(s) and s <= t:
                best = k
            elif np.isfinite(s) and s > t:
                break
        return best

    # ------------------------------------------------------------------ reading
    def _read_records(self, first, last):
        """Records first .. last-1: their samples (one (512, 2) int16 array each) and stamp times (NaN where
        missing)."""
        sample_blocks = []
        stamp_times = []
        for k in range(first, last):
            self.f.seek(OFF0 + k * BLK)
            record = self.f.read(BLK)
            gps_seconds, gps_nanoseconds = struct.unpack('<II', record[10:18])
            stamp_times.append(_stamp_seconds(self.week0_unix, gps_seconds, gps_nanoseconds, self.t_name))
            sample_blocks.append(_samples_of_record(record))
        return sample_blocks, np.array(stamp_times)

    def read(self, t0, dur):
        """IQ (complex64, full scale 1) and per-sample unix times for [t0, t0+dur]; ok=False if the window touches a
        missing/non-monotonic stamp or a gap (> 1.5 samples between blocks)."""
        first = self.find_block(t0)
        record_count = int(np.ceil(dur * self.fs / NS)) + 2
        last = min(first + record_count, self.nb)
        sample_blocks, stamps = self._read_records(first, last)

        samples = np.concatenate(sample_blocks).astype(np.float32)
        x = (samples[:, 0] + 1j * samples[:, 1]) / 32768

        # Every stamp present, and each record exactly 512 samples after the previous one (within 1.5 samples)?
        ok = bool(np.all(np.isfinite(stamps)))
        if ok:
            records_apart = np.diff(stamps) * self.fs / NS
            ok = bool(np.all(np.abs(records_apart - 1) < 1.5 / NS))

        # Missing stamps are filled by interpolating between the present ones (only for the time axis).
        present = np.isfinite(stamps)
        if present.any():
            filled = np.interp(np.arange(len(stamps)), np.where(present)[0], stamps[present])
        else:
            filled = 0
        stamps_filled = np.where(present, stamps, filled)

        # Time of every sample: its record's stamp + its position in the record / fs.
        within_record = np.arange(NS) / self.fs
        t = (stamps_filled[:, None] + within_record[None, :]).ravel()
        return x.astype(np.complex64), t, ok

    def iter_blocks(self, chunk_blocks=1400):
        """Yield (x, t, ok) for consecutive chunks of whole blocks (for streaming detectors).

        Within each chunk a straight line is fitted through the present stamps; stamps that are missing or more
        than 2 samples off that line are replaced by the line and flagged (ok = False for their samples' block).
        """
        for first in range(0, self.nb, chunk_blocks):
            last = min(first + chunk_blocks, self.nb)
            sample_blocks, stamps = self._read_records(first, last)
            samples = np.concatenate(sample_blocks).astype(np.float32)

            present = np.isfinite(stamps)
            if present.sum() < 2:
                continue

            # Per-block times; bad stamps replaced by a linear fit through the good ones in this chunk.
            block_number = np.arange(len(stamps))
            line = np.polyfit(block_number[present], stamps[present], 1)
            on_line = np.polyval(line, block_number)
            residual = np.abs(stamps - on_line)
            bad = ~present | (residual > 2.0 / self.fs)
            block_times = np.where(bad, on_line, stamps)

            x = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64) / 32768
            within_record = np.arange(NS) / self.fs
            t = (block_times[:, None] + within_record[None, :]).ravel()
            yield x, t, ~bad


def provenance(out_prefix, argv, inputs):
    """Write <out_prefix>.prov.json: argv, git hash of this tree, input file sizes/mtimes, UTC time."""
    import json
    import subprocess
    import sys

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(['git', '-C', here, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        git = result.stdout.strip()
    except Exception:
        git = ''

    input_files = {}
    for path in inputs:
        if os.path.exists(path):
            input_files[path] = dict(bytes=os.path.getsize(path), mtime=os.path.getmtime(path))

    record = dict(argv=argv, git=git, python=sys.version.split()[0], numpy=np.__version__,
                  utc=dt.datetime.now(dt.timezone.utc).isoformat(), inputs=input_files)
    with open(out_prefix + '.prov.json', 'w') as handle:
        json.dump(record, handle, indent=1)
