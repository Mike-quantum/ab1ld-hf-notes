"""
STANAG 4285 sync reference (PN31 / PN80) and the root-raised-cosine pulse.

Part of s4285-livepn, a Python version of DK8OK's MATLAB program DK8OK_STANAG4285_LIVE_PN_V1_4.m
((c) 2026 Nils Schiffhauer, DK8OK; posted to the Dopplergram group 2026-10-11).  The constants and the PN
generator below are DK8OK's; DK8OK credits Con Wassilieff, ZL2AFP (the STANAG 4285 receiver the project grew
from) and Peter Martinez, G3PLX (suggestions).  Python version: Claude (AI assistant) for Mike Burns, AB1LD,
2026-10.

Background, for a reader new to STANAG 4285:
    The waveform runs at 2400 symbols per second in frames of 256 symbols (106.67 ms).  Every frame starts with
    the same 80-symbol BPSK synchronisation sequence ("PN80"): a 31-bit maximal-length sequence (PN31) sent
    twice, followed by its first 18 bits.  Because this sequence is known, a receiver can find the frame start,
    measure the carrier offset and channel phase, and count bit errors on it -- which is what this program does.
"""
import numpy as np

# ---------------------------------------------------------------------------------------------- constants
# DK8OK's constants (first lines of LIVE PN V1.4).
FS_AUDIO = 48000          # default sound-card sample rate (Hz)
FB = 19200                # working sample rate after resampling (Hz) = 8 samples per symbol
SPS = 8                   # samples per symbol at FB
BAUD = 2400               # STANAG 4285 symbol rate (symbols per second)
FRAME_SYMBOLS = 256       # symbols per STANAG 4285 frame
FRAME = FRAME_SYMBOLS * SPS   # one frame = 2048 samples at FB
RRC_BETA = 0.2            # roll-off of the root-raised-cosine pulse
RRC_SPAN = 10             # length of the pulse-shaping filter, in symbols
AUDIO_CENTRE = 1800.0     # usual centre of a STANAG 4285 signal in USB receiver audio (Hz)

# The PN31 bit string as DK8OK's generator produces it (his V1.1c; STANAG 4285 Annex A, Figure A-4).
PN31_BITS = '0101100111110001101110101000010'


# ---------------------------------------------------------------------------------------------- PN sequences
def pn31_bits():
    """The 31 sync bits from DK8OK's shift-register generator, checked against PN31_BITS.

    Generator polynomial x^5 + x^2 + 1.  The register starts as 1 1 0 1 0.  Each step outputs stage 5, then
    shifts the register one place towards stage 5 and puts (stage 5 XOR stage 3) into stage 1.

    DK8OK's MATLAB:  r=[1 1 0 1 0]; for k=1:31, bits(k)=r(5); r=[xor(r(5),r(3)) r(1:4)]; end
    (Python lists count from 0, so MATLAB's r(5) is register[4] and r(3) is register[2].)
    """
    register = [1, 1, 0, 1, 0]
    bits = []
    for step in range(31):
        output_bit = register[4]
        bits.append(output_bit)
        feedback_bit = register[4] ^ register[2]
        register = [feedback_bit] + register[:4]
    bits = np.array(bits, dtype=int)

    bit_string = ''.join(str(bit) for bit in bits)
    if bit_string != PN31_BITS:
        raise AssertionError('PN31 reference generation failed.')
    return bits


def pn80_symbols(bits=None):
    """The 80 sync symbols as +1/-1: PN31, PN31 again, then the first 18 bits of PN31.

    Bit 1 becomes +1 and bit 0 becomes -1, as in DK8OK's code:  pn = 2*bits(mod(0:79,31)+1)-1
    """
    if bits is None:
        bits = pn31_bits()
    bits = np.asarray(bits, dtype=int)

    # Symbol k (k = 0..79) uses bit number k modulo 31: 0..30, 0..30, 0..17.
    bit_index = np.arange(80) % 31
    sync_bits = bits[bit_index]
    return 2.0 * sync_bits - 1.0


# ---------------------------------------------------------------------------------------------- pulse shape
def rrc_sqrt(beta=RRC_BETA, span=RRC_SPAN, sps=SPS):
    """Root-raised-cosine filter taps, as MATLAB rcosdesign(beta, span, sps, 'sqrt').

    span * sps + 1 taps (81 for the defaults), symmetric about the middle tap, scaled to unit energy.
    The same function as rcosdesign_sqrt in our port of DK8OK's ESSENTIAL program (s4285_dk8ok.py, 2026-10-06).
    """
    # Tap times in symbol periods: -span/2 .. +span/2 in steps of 1/sps.
    half_length = span * sps / 2
    tap_times = np.arange(-half_length, half_length + 1) / sps

    taps = rrc_pulse(tap_times, beta)
    energy = np.sum(rrc_pulse(tap_times, beta) ** 2)
    return taps / np.sqrt(energy)


def rrc_pulse(t, beta=RRC_BETA):
    """The continuous root-raised-cosine pulse p(t), with t in symbol periods.

    General formula:
        p(t) = [sin(pi t (1 - beta)) + 4 beta t cos(pi t (1 + beta))] / [pi t (1 - (4 beta t)^2)]
    It is 0/0 at two kinds of point, where the limits are used instead:
        t = 0:                 p = 1 - beta + 4 beta / pi
        t = +/- 1 / (4 beta):  p = beta/sqrt(2) [(1 + 2/pi) sin(pi/(4 beta)) + (1 - 2/pi) cos(pi/(4 beta))]
    """
    t = np.asarray(t, dtype=float)
    pulse = np.empty_like(t)

    # Sort the times into the two special cases and the general case.
    at_zero = np.abs(t) < 1e-12
    at_singularity = np.abs(np.abs(4 * beta * t) - 1) < 1e-9
    general = ~(at_zero | at_singularity)

    pulse[at_zero] = 1 - beta + 4 * beta / np.pi

    sine_part = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
    cosine_part = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
    pulse[at_singularity] = beta / np.sqrt(2) * (sine_part + cosine_part)

    tt = t[general]
    numerator = np.sin(np.pi * tt * (1 - beta)) + 4 * beta * tt * np.cos(np.pi * tt * (1 + beta))
    denominator = np.pi * tt * (1 - (4 * beta * tt) ** 2)
    pulse[general] = numerator / denominator
    return pulse
