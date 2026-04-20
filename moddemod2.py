import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# -----------------------------
# PARAMETERS
# -----------------------------
N = 100                  # number of bits
bit_rate = 1000          # 1 kbps
fc = 10000               # carrier frequency (10 kHz)
fs = 100000              # sampling frequency (100 kHz)
Tb = 1 / bit_rate
t = np.arange(0, N*Tb, 1/fs)

# -----------------------------
# MESSAGE SIGNAL
# -----------------------------
bits = np.random.randint(0, 2, N)
message = np.repeat(bits, int(fs*Tb))

# -----------------------------
# BPSK MODULATION
# -----------------------------
carrier = np.cos(2 * np.pi * fc * t)
bpsk = (2 * message - 1) * carrier

# -----------------------------
# ADD NOISE (AWGN)
# -----------------------------
noise = np.random.normal(0, 0.5, len(bpsk))
bpsk_noisy = bpsk + noise

# -----------------------------
# M-FSK MODULATION (M = 4)
# -----------------------------
M = 4
freqs = [fc + i*1000 for i in range(M)]
symbols = np.random.randint(0, M, N)

fsk = np.zeros_like(t)

for i, sym in enumerate(symbols):
    start = int(i * Tb * fs)
    end = start + int(Tb * fs)
    fsk[start:end] = np.cos(2 * np.pi * freqs[sym] * t[start:end])

# Add noise
fsk_noisy = fsk + np.random.normal(0, 0.5, len(fsk))

# -----------------------------
# PSD
# -----------------------------
f_psd, psd = welch(bpsk, fs)

# -----------------------------
# DEMODULATION (BPSK)
# -----------------------------
demod = bpsk_noisy * carrier
recovered_bits = (demod > 0).astype(int)

# -----------------------------
# PLOTS (SEPARATE FIGURES)
# -----------------------------

# 1. Message Signal
plt.figure()
plt.plot(message[:1000])
plt.title("Message Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# 2. BPSK Modulated Signal
plt.figure()
plt.plot(bpsk[:1000])
plt.title("BPSK Modulated Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# 3. BPSK + Noise
plt.figure()
plt.plot(bpsk_noisy[:1000])
plt.title("BPSK Signal with Noise")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# 4. PSD of BPSK
plt.figure()
plt.semilogy(f_psd, psd)
plt.title("Power Spectral Density (BPSK)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid()

# 5. Constellation Diagram (BPSK)
plt.figure()
plt.scatter(bpsk_noisy[:2000], np.zeros(2000))
plt.title("Constellation Diagram (BPSK)")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid()

# 6. M-FSK Signal
plt.figure()
plt.plot(fsk[:1000])
plt.title("M-FSK Signal (M=4)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# 7. M-FSK + Noise
plt.figure()
plt.plot(fsk_noisy[:1000])
plt.title("M-FSK Signal with Noise")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

plt.show()