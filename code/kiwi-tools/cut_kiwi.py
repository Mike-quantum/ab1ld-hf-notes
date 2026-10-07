#!/usr/bin/env python3
"""
Cut a short piece out of a KiwiSDR kiwirecorder IQ file WITHOUT changing its format: the original header and the
original 512-sample blocks, each with its own GPS time stamp, copied byte for byte. Only the RIFF size field is
updated. The cut starts at the last block stamped at or before START and ends after SECONDS.

Written by Claude (Anthropic AI) for Mike Burns, AB1LD, 2026-10-07. File layout as written by kiwirecorder
(kiwiclient, John Seamons, jks-prv) and read by kiwiwav.py (ours): 36-byte RIFF + fmt header, then records of
2074 bytes = 'kiwi' chunk (8 + 10 bytes: last, dummy, GPS seconds of week, GPS nanoseconds) + 'data' chunk
(8 + 2048 bytes = 512 complex int16 samples).
Usage: cut_kiwi.py KIWI.wav START_UTC(YYYY-mm-ddTHH:MM:SS) SECONDS OUT_DIR
The output name is YYYYMMDDTHHMMSSZ_<rest of the input name after its own time stamp>, because kiwiwav.py takes the
GPS week from the time stamp in the file name.
"""
import os
import re
import struct
import sys
import datetime as dt

import kiwiwav

HEADER_BYTES = 36
RECORD_BYTES = 2074
SAMPLES_PER_RECORD = 512


def main():
    source_path, start_text, seconds_text, out_dir = sys.argv[1:5]
    start = dt.datetime.strptime(start_text, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=dt.timezone.utc)
    seconds = float(seconds_text)
    recording = kiwiwav.KiwiWav(source_path)
    first_record = recording.find_block(start.timestamp())
    record_count = int(seconds * recording.fs / SAMPLES_PER_RECORD) + 1
    name = os.path.basename(source_path)
    rest = re.sub(r'^\d{8}T\d{6}Z', '', name)
    out_path = os.path.join(out_dir, start.strftime('%Y%m%dT%H%M%SZ') + rest)
    with open(source_path, 'rb') as source:
        header = bytearray(source.read(HEADER_BYTES))
        source.seek(HEADER_BYTES + first_record * RECORD_BYTES)
        records = source.read(record_count * RECORD_BYTES)
    if len(records) != record_count * RECORD_BYTES:
        sys.exit('the recording ends before the requested interval')
    total_size = HEADER_BYTES + len(records)
    header[4:8] = struct.pack('<I', total_size - 8)
    with open(out_path, 'wb') as out:
        out.write(header)
        out.write(records)
    first_stamp = recording.stamp(first_record)
    print(f'wrote {out_path}: {record_count} records ({record_count * SAMPLES_PER_RECORD} samples, '
          f'{record_count * SAMPLES_PER_RECORD / recording.fs:.1f} s), first record GPS-stamped '
          f'{dt.datetime.fromtimestamp(first_stamp, dt.timezone.utc).isoformat()}')


if __name__ == '__main__':
    main()
