"""
Test signals for s4285-livepn.

1. demo_audio(): DK8OK's own demo source from LIVE PN V1.4 (his function demoAudio), ported so that his built-in
   known answer can be checked: a PN80 every 256 symbols, deterministic +/-1 data in between, root-raised-cosine
   shaping at 8 samples per symbol, linear interpolation to the audio rate, carrier offset +7 Hz, channel phase
   0.8 + 0.4 sin(2 pi t / 15) rad, amplitude 0.35, Gaussian noise 0.035 rms.
   (c) 2026 Nils Schiffhauer, DK8OK -- his design; Python port by Claude.

2. stanag4285_audio(): our own, independent STANAG 4285 generator (Claude for Mike Burns, AB1LD, 2026-10-12).
   It shares nothing with the analyser except the published sync sequence: the pulse is the analytic
   root-raised-cosine evaluated directly at the audio sample times (no upsample/resample/FIR chain), the frame
   is the full STANAG 4285 structure (80 BPSK sync, then 3 x (32 data + 16 probe) + 32 data; data and probe
   symbols 8-PSK), and the channel is any number of discrete paths, each with delay, complex gain and Doppler
   shift, plus a common carrier offset, optional slow phase wander and white noise at a stated SNR in 3 kHz.
   It returns the truth needed to score the analyser (PN-start times, carrier offset, channel phase).
"""
import numpy as np

from s4285_reference import FB, SPS, BAUD, AUDIO_CENTRE, PN31_BITS, pn80_symbols, rrc_sqrt, rrc_pulse


# ---------------------------------------------------------------------------------------------- DK8OK demo
def demo_audio(n0, n1, fs=48000, centre=AUDIO_CENTRE, rng=None, pn=None):
    """DK8OK demoAudio(n0, n1): samples n0 .. n1-1 of his demo signal (absolute sample numbers).

    The signal is defined for all time, so consecutive calls give one continuous stream.
    """
    if rng is None:
        rng = np.random.default_rng()
    if pn is None:
        pn = pn80_symbols()
    h = rrc_sqrt()

    sample_number = np.arange(n0, n1)
    tt = sample_number / fs                                  # sample times (s)

    # The symbols that influence these samples, with 20 symbols to spare at each end for the filter.
    first = int(np.floor((n0 / fs) * 2400)) - 20
    last = int(np.ceil((n1 / fs) * 2400)) + 20
    symbol_number = np.arange(first, last + 1)

    # Data symbols: a fixed +/-1 pattern from the symbol number (his formula).  Frame positions 0..79 are
    # overwritten with the PN80 sync.
    position_in_frame = np.mod(symbol_number, 256)
    data_bits = np.mod(symbol_number * 17 + np.floor(symbol_number / 7), 2)
    symbols = 2.0 * data_bits - 1
    is_sync = position_in_frame < 80
    symbols[is_sync] = pn[position_in_frame[is_sync]]

    # Pulse shaping at 8 samples per symbol (FB): impulses every 8 samples, then the RRC filter.
    impulses = np.zeros(symbol_number.size * SPS)
    impulses[::SPS] = symbols
    shaped = np.convolve(impulses, h, 'same') * np.sqrt(SPS)
    shaped_times = (first * 8 + np.arange(impulses.size)) / FB

    # To the audio sample times by linear interpolation (MATLAB interp1(..., 'linear', 0)).
    baseband = np.interp(tt, shaped_times, shaped, left=0.0, right=0.0)

    # Up to the audio centre + 7 Hz, with the slowly wandering channel phase; add noise.
    channel_phase = .8 + .4 * np.sin(2 * np.pi * tt / 15)
    carrier = np.exp(1j * (2 * np.pi * (centre + 7) * tt + channel_phase))
    noise = .035 * rng.standard_normal(tt.size)
    return .35 * np.real(baseband * carrier) + noise


def demo_truth(times, phase_offset=0.0):
    """Known answer for demo_audio at PN-start times: carrier offset +7 Hz, and the channel phase
    2 pi 7 t + 0.8 + 0.4 sin(2 pi t / 15), wrapped to (-pi, pi].  Returns (offset in Hz, phases)."""
    times = np.asarray(times)
    phase = 2 * np.pi * 7 * times + .8 + .4 * np.sin(2 * np.pi * times / 15) + phase_offset
    wrapped_phase = np.angle(np.exp(1j * phase))
    return 7.0, wrapped_phase


def demo_pn_starts(duration):
    """PN-start times in the demo: symbol 256 m, at t = 256 m / 2400 s, up to `duration` seconds."""
    frame_number = np.arange(int(duration * BAUD / 256) + 1)
    return frame_number * 256 / BAUD


# ---------------------------------------------------------------------------------------------- our generator
def stanag4285_frame_symbols(nframes, rng):
    """Complex symbols of `nframes` STANAG 4285 frames: 80 BPSK sync + 3 x (32 data + 16 probe) + 32 data.

    Sync: PN31 + PN31 + first 18 (STANAG 4285 Annex A), bit 1 -> +1 as in DK8OK's reference.  Data: random
    8-PSK.  Probe: one fixed 16-symbol 8-PSK sequence (a stand-in for the standard's scrambled probe).
    """
    pn31 = np.array([int(character) for character in PN31_BITS])
    sync_bits = pn31[np.arange(80) % 31]
    sync = 2.0 * sync_bits - 1

    def random_8psk(count, generator):
        """`count` 8-PSK symbols exp(j pi k / 4) with random k = 0..7."""
        return np.exp(1j * np.pi / 4 * generator.integers(0, 8, count))

    probe = random_8psk(16, np.random.default_rng(4285))

    frames = []
    for frame in range(nframes):
        parts = [sync.astype(complex)]
        for block in range(3):
            parts.append(random_8psk(32, rng))
            parts.append(probe)
        parts.append(random_8psk(32, rng))
        frames.append(np.concatenate(parts))

    symbols = np.concatenate(frames)
    assert symbols.size == 256 * nframes
    return symbols


def stanag4285_audio(duration, fs=48000, centre=AUDIO_CENTRE, cfo=0.0, paths=None, snr_db=None,
                     start=0.0123, wander=None, amplitude=0.3, seed=1):
    """Real USB audio of a STANAG 4285 transmission through a discrete multipath channel.

    duration: seconds of audio.
    paths: list of dicts {delay_ms, gain (complex), doppler_hz}; default one path with gain 1.
    cfo: common carrier offset (Hz) relative to the audio centre.
    wander: None, or (amplitude_rad, period_s) for a common phase  amplitude * sin(2 pi t / period).
    snr_db: signal power / noise power in 3 kHz (None = no noise).
    start: time (s) of the first symbol of the first frame on the first path.
    amplitude: RMS of the signal before noise is added.
    Returns (audio, truth) where truth holds the PN-start times, carrier offset and channel phase of path 1.
    """
    rng = np.random.default_rng(seed)
    if not paths:
        paths = [dict(delay_ms=0.0, gain=1.0, doppler_hz=0.0)]
    symbol_period = 1.0 / BAUD
    sample_number = np.arange(int(round(duration * fs)))
    t = sample_number / fs

    nframes = int(np.ceil((duration - start) / (256 * symbol_period))) + 2
    symbols = stanag4285_frame_symbols(nframes, rng)

    # Each path: sum of symbol * pulse(time since that symbol), times the path's gain and Doppler.
    baseband = np.zeros(t.size, complex)
    for path in paths:
        first_symbol_time = start + path['delay_ms'] * 1e-3
        time_in_symbols = (t - first_symbol_time) / symbol_period
        latest_symbol = np.floor(time_in_symbols).astype(int)

        path_signal = np.zeros(t.size, complex)
        for j in range(-6, 8):             # symbols within the pulse span (+/-5 needed, a little spare)
            k = latest_symbol + j
            valid = (k >= 0) & (k < symbols.size)
            pulse = rrc_pulse(time_in_symbols[valid] - k[valid])
            path_signal[valid] += symbols[k[valid]] * pulse

        doppler = np.exp(2j * np.pi * path['doppler_hz'] * t)
        baseband += complex(path['gain']) * doppler * path_signal

    # Up to the audio centre plus the carrier offset (and optional phase wander); real part = USB audio.
    phase = 2 * np.pi * (centre + cfo) * t
    if wander is not None:
        phase = phase + wander[0] * np.sin(2 * np.pi * t / wander[1])
    audio = np.real(baseband * np.exp(1j * phase))
    audio *= amplitude / np.sqrt(np.mean(audio ** 2))

    # White noise over 0 .. fs/2, scaled so that the noise in 3 kHz gives the requested SNR.
    noise_rms = 0.0
    if snr_db is not None:
        signal_power = np.mean(audio ** 2)
        noise_variance = signal_power / 10 ** (snr_db / 10) * (fs / 2) / 3000.0
        noise_rms = np.sqrt(noise_variance)
        audio = audio + noise_rms * rng.standard_normal(t.size)

    # Truth for the first path: PN-start times of complete sync sequences, offset and channel phase there.
    frame_times = start + np.arange(nframes) * 256 * symbol_period
    frame_times = frame_times[frame_times + 80 * symbol_period < duration]
    first_path = paths[0]
    delay = first_path['delay_ms'] * 1e-3
    true_offset = cfo + first_path['doppler_hz']
    channel_phase = np.angle(complex(first_path['gain'])) + 2 * np.pi * true_offset * (frame_times + delay)
    if wander is not None:
        channel_phase = channel_phase + wander[0] * np.sin(2 * np.pi * (frame_times + delay) / wander[1])

    truth = dict(pn_start=frame_times + delay,
                 cfo=true_offset,
                 phase=np.angle(np.exp(1j * channel_phase)),
                 paths=paths,
                 snr_db=snr_db,
                 noise_rms=noise_rms,
                 fs=fs,
                 centre=centre)
    return audio, truth
