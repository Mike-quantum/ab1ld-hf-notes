"""Shared reader for KiwiSDR IQ wav files (kiwirecorder -w, GPS-stamped). [2026-10-01, after Codex audit]
Layout: RIFF(12) + fmt chunk(24) + N x [ 'kiwi' chunk (8+10: last, dummy, gps_sec_of_week u32, gps_ns u32)
                                         + 'data' chunk (8+2048 = 512 complex int16) ]  -> 2074-byte records from offset 36.
Fixes vs the per-script copies:
  * GPS week derived from the recording start in the FILENAME (YYYYMMDDTHHMMSSZ), not hard-coded; GPS-UTC = 18 s
    (valid since 2017-01-01; checked against the filename start, which must agree within 120 s).
  * Every sample time comes from ITS OWN block stamp (t_block + j/fs), never extrapolated from block 0.
  * Non-monotonic / corrupt / gapped stamps are detected; windows that touch one are flagged bad.
  * Seek uses a sparse monotonic index (every STRIDE blocks) built once and cached next to the wav (.idx.npz).
"""
import os, re, struct, datetime as dt
import numpy as np
BLK, NS, OFF0, GPS_UTC = 2074, 512, 36, 18.0
GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
FS_NOMINAL = 11998.9554          # measured 2026-09-30/10-01 (header says 11999)

class KiwiWav:
    def __init__(self, path, fs=FS_NOMINAL, stride=256):
        self.path, self.fs = path, fs
        self.f = open(path, 'rb'); self.f.seek(0, 2); self.nb = (self.f.tell() - OFF0) // BLK
        m = re.search(r'(\d{8}T\d{6})Z', os.path.basename(path))
        if not m: raise ValueError('filename lacks YYYYMMDDTHHMMSSZ start stamp: ' + path)
        self.t_name = dt.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S').replace(tzinfo=dt.timezone.utc).timestamp()
        # GPS week containing the filename start (in GPS time)
        gps_now = self.t_name - GPS_EPOCH.timestamp() + GPS_UTC
        self.week0_unix = GPS_EPOCH.timestamp() + (gps_now // 604800) * 604800 - GPS_UTC
        self._index(stride)

    def _raw_stamp(self, k):
        self.f.seek(OFF0 + k * BLK); c = self.f.read(18)
        if c[:4] != b'kiwi': raise ValueError(f'layout break at block {k}')
        s, ns = struct.unpack('<II', c[10:18]); return None if s == 0 else (s, ns)

    def stamp(self, k):
        r = self._raw_stamp(k)
        if r is None: return np.nan
        t = self.week0_unix + r[0] + r[1] * 1e-9
        if t < self.t_name - 3600: t += 604800                  # recording crossed a GPS-week boundary
        return t

    def _index(self, stride):
        cache = self.path + f'.idx{stride}.npz'
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(self.path):
            z = np.load(cache); self.ik, self.it = z['k'], z['t']
        else:
            k = np.arange(1, self.nb, stride); t = np.array([self.stamp(int(x)) for x in k])
            self.ik, self.it = k, t
            try: np.savez(cache, k=k, t=t)
            except OSError: pass
        first = self.it[np.isfinite(self.it)][0]
        if abs(first - self.t_name) > 120:
            raise ValueError(f'GPS stamps ({first:.0f}) disagree with filename start ({self.t_name:.0f}) by >120 s')
        ok = np.isfinite(self.it) & (np.abs(self.it - (first + (self.ik - self.ik[0]) * NS / self.fs)) < 3600)
        self.ik, self.it = self.ik[ok], self.it[ok]               # drop corrupt index entries (e.g. the +-68,613 s pair)

    def find_block(self, t):
        """Last block whose stamp <= t (sparse index + local scan; robust to isolated bad stamps)."""
        j = int(np.searchsorted(self.it, t)) - 1
        k = int(self.ik[max(j, 0)]); best = k
        for kk in range(k, min(k + 2 * 256 + 2, self.nb)):
            s = self.stamp(kk)
            if np.isfinite(s) and s <= t: best = kk
            elif np.isfinite(s) and s > t: break
        return best

    def read(self, t0, dur):
        """IQ (complex64, full scale 1) and per-sample unix times for [t0, t0+dur]; ok=False if the window touches a
        missing/non-monotonic stamp or a gap (> 1.5 samples between blocks)."""
        k0 = self.find_block(t0); n = int(np.ceil(dur * self.fs / NS)) + 2
        xs, ts = [], []
        for k in range(k0, min(k0 + n, self.nb)):
            self.f.seek(OFF0 + k * BLK); c = self.f.read(BLK)
            s, ns = struct.unpack('<II', c[10:18]); tb = np.nan if s == 0 else self.week0_unix + s + ns * 1e-9
            if np.isfinite(tb) and tb < self.t_name - 3600: tb += 604800
            d = np.frombuffer(c[26:26 + 4 * NS], '<i2').reshape(-1, 2)
            xs.append(d); ts.append(tb)
        ts = np.array(ts); d = np.concatenate(xs).astype(np.float32); x = (d[:, 0] + 1j * d[:, 1]) / 32768
        ok = bool(np.all(np.isfinite(ts)))
        if ok:
            step = np.diff(ts) * self.fs / NS
            ok = bool(np.all(np.abs(step - 1) < 1.5 / NS))      # each block exactly 512 samples later (+-1.5 samples)
        tsf = np.where(np.isfinite(ts), ts, np.interp(np.arange(len(ts)), np.where(np.isfinite(ts))[0], ts[np.isfinite(ts)]) if np.isfinite(ts).any() else 0)
        t = (tsf[:, None] + np.arange(NS)[None, :] / self.fs).ravel()
        return x.astype(np.complex64), t, ok

    def iter_blocks(self, chunk_blocks=1400):
        """Yield (x, t, ok) for consecutive chunks of whole blocks (for streaming detectors)."""
        for k0 in range(0, self.nb, chunk_blocks):
            xs, ts = [], []
            for k in range(k0, min(k0 + chunk_blocks, self.nb)):
                self.f.seek(OFF0 + k * BLK); c = self.f.read(BLK); s, ns = struct.unpack('<II', c[10:18])
                tb = np.nan if s == 0 else self.week0_unix + s + ns * 1e-9
                if np.isfinite(tb) and tb < self.t_name - 3600: tb += 604800
                xs.append(np.frombuffer(c[26:26 + 4 * NS], '<i2').reshape(-1, 2)); ts.append(tb)
            ts = np.array(ts); d = np.concatenate(xs).astype(np.float32)
            good = np.isfinite(ts)
            if good.sum() < 2: continue
            # per-block times; bad stamps replaced by a linear fit through the good ones in this chunk
            kk = np.arange(len(ts)); A = np.polyfit(kk[good], ts[good], 1)
            resid = np.abs(ts - np.polyval(A, kk)); bad = ~good | (resid > 2.0 / self.fs)
            tb = np.where(bad, np.polyval(A, kk), ts)
            yield (d[:, 0] + 1j * d[:, 1]).astype(np.complex64) / 32768, (tb[:, None] + np.arange(NS)[None, :] / self.fs).ravel(), ~bad

def provenance(out_prefix, argv, inputs):
    """Write <out_prefix>.prov.json: argv, git hash of this tree, input file sizes/mtimes, UTC time."""
    import json, subprocess, sys
    try: git = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: git = ''
    rec = dict(argv=argv, git=git, python=sys.version.split()[0], numpy=np.__version__,
               utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               inputs={p: dict(bytes=os.path.getsize(p), mtime=os.path.getmtime(p)) for p in inputs if os.path.exists(p)})
    json.dump(rec, open(out_prefix + '.prov.json', 'w'), indent=1)
