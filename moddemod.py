import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.gridspec as gridspec

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
bit_rate = 100  # bits per second
samples_per_bit = 100  # samples per bit
sampling_rate = bit_rate * samples_per_bit  # sampling frequency
carrier_freq = 500  # carrier frequency in Hz
duration = 0.5  # duration of signal in seconds
num_bits = int(bit_rate * duration)
t = np.arange(0, duration, 1/sampling_rate)

print(f"Parameters:")
print(f"Bit rate: {bit_rate} bps")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Number of bits: {num_bits}")
print(f"Carrier frequency: {carrier_freq} Hz")

# Step 1: Generate random binary message signal
message_bits = np.random.randint(0, 2, num_bits)
message_signal = np.repeat(message_bits, samples_per_bit)

# Step 4: PSK Modulation (BPSK)
def psk_modulate(bits, carrier_freq, sampling_rate, samples_per_bit):
    t_mod = np.arange(len(bits)) / sampling_rate
    carrier = np.cos(2 * np.pi * carrier_freq * t_mod)
    # BPSK: 0 -> -1, 1 -> +1
    psk_symbols = 2 * bits - 1
    modulated = psk_symbols * carrier
    return modulated, carrier

# Step 4: M-ary FSK Modulation (M=4)
def mfsk_modulate(bits, carrier_freq, sampling_rate, samples_per_bit, M=4):
    # Convert bits to symbols (2 bits per symbol for M=4)
    symbols = []
    for i in range(0, len(bits), int(np.log2(M))):
        if i + int(np.log2(M)) <= len(bits):
            symbol = 0
            for j in range(int(np.log2(M))):
                symbol += bits[i+j] * (2 ** (int(np.log2(M))-1-j))
            symbols.append(symbol)
    
    # Generate FSK signal
    t_mod = np.arange(len(bits)) / sampling_rate
    modulated = np.zeros(len(bits))
    freq_sep = carrier_freq / 4  # Frequency separation
    
    for i, symbol in enumerate(symbols):
        start_idx = i * samples_per_bit * int(np.log2(M))
        end_idx = min((i+1) * samples_per_bit * int(np.log2(M)), len(bits))
        if start_idx < len(bits):
            t_seg = t_mod[start_idx:end_idx]
            freq = carrier_freq + (symbol - M/2) * freq_sep
            modulated[start_idx:end_idx] = np.cos(2 * np.pi * freq * t_seg[:len(t_seg)])
    
    return modulated, symbols

# Step 3 & 4: Modulate signals
psk_modulated, psk_carrier = psk_modulate(message_signal, carrier_freq, sampling_rate, samples_per_bit)
mfsk_modulated, mfsk_symbols = mfsk_modulate(message_signal, carrier_freq, sampling_rate, samples_per_bit, M=4)

# Step 5: Add noise
def add_noise(signal_power, snr_db, modulated_signal):
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(modulated_signal))
    noisy_signal = modulated_signal + noise
    return noisy_signal, noise

# Calculate signal power
psk_power = np.mean(psk_modulated**2)
mfsk_power = np.mean(mfsk_modulated**2)

# Add noise with different SNR values
snr_db = 10  # Signal-to-Noise Ratio in dB
psk_noisy, psk_noise = add_noise(psk_power, snr_db, psk_modulated)
mfsk_noisy, mfsk_noise = add_noise(mfsk_power, snr_db, mfsk_modulated)

# Step 6: Demodulation
def psk_demodulate(noisy_signal, carrier_freq, sampling_rate, samples_per_bit):
    # Coherent detection
    t = np.arange(len(noisy_signal)) / sampling_rate
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    mixed = noisy_signal * carrier
    
    # Low-pass filter
    nyquist = sampling_rate / 2
    cutoff = bit_rate  # Cutoff frequency
    normalized_cutoff = cutoff / nyquist
    b, a = butter(4, normalized_cutoff, btype='low')
    filtered = lfilter(b, a, mixed)
    
    # Sample at bit centers
    demodulated_bits = []
    for i in range(0, len(filtered), samples_per_bit):
        sample_idx = i + samples_per_bit // 2
        if sample_idx < len(filtered):
            decision = 1 if filtered[sample_idx] > 0 else 0
            demodulated_bits.append(decision)
    
    return np.array(demodulated_bits)

def mfsk_demodulate(noisy_signal, carrier_freq, sampling_rate, samples_per_bit, M=4):
    t = np.arange(len(noisy_signal)) / sampling_rate
    freq_sep = carrier_freq / 4
    freqs = [carrier_freq + (i - M/2) * freq_sep for i in range(M)]
    
    demodulated_bits = []
    samples_per_symbol = samples_per_bit * int(np.log2(M))
    
    for i in range(0, len(noisy_signal), samples_per_symbol):
        if i + samples_per_symbol <= len(noisy_signal):
            segment = noisy_signal[i:i+samples_per_symbol]
            t_seg = t[i:i+samples_per_symbol]
            
            # Correlate with each carrier frequency
            correlations = []
            for freq in freqs:
                reference = np.cos(2 * np.pi * freq * t_seg)
                correlation = np.sum(segment * reference)
                correlations.append(correlation)
            
            # Find frequency with maximum correlation
            detected_symbol = np.argmax(correlations)
            
            # Convert symbol to bits
            for j in range(int(np.log2(M))-1, -1, -1):
                demodulated_bits.append((detected_symbol >> j) & 1)
    
    # Pad to match original length if needed
    while len(demodulated_bits) < len(message_bits):
        demodulated_bits.append(0)
    
    return np.array(demodulated_bits[:len(message_bits)])

# Demodulate
psk_demodulated = psk_demodulate(psk_noisy, carrier_freq, sampling_rate, samples_per_bit)
mfsk_demodulated = mfsk_demodulate(mfsk_noisy, carrier_freq, sampling_rate, samples_per_bit, M=4)

# Calculate Bit Error Rate (BER)
psk_ber = np.sum(message_bits != psk_demodulated) / len(message_bits)
mfsk_ber = np.sum(message_bits != mfsk_demodulated) / len(message_bits)

print(f"\nPerformance:")
print(f"PSK BER: {psk_ber:.4f} ({np.sum(message_bits != psk_demodulated)} errors out of {len(message_bits)})")
print(f"M-ary FSK (M=4) BER: {mfsk_ber:.4f} ({np.sum(message_bits != mfsk_demodulated)} errors out of {len(message_bits)})")

# Step: Calculate Power Spectral Density (PSD)
def calculate_psd(signal, sampling_rate):
    f, Pxx = signal.periodogram(signal, fs=sampling_rate, nperseg=256)
    return f, 10*np.log10(Pxx + 1e-10)  # Convert to dB

# Create plots
fig = plt.figure(figsize=(15, 20))
gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3)

# Plot 1: Message Signal (original bits)
ax1 = fig.add_subplot(gs[0, :])
time_bits = np.arange(len(message_bits)) / bit_rate
ax1.step(time_bits, message_bits, where='post', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Message Signal (Original Binary Data)')
ax1.set_ylim(-0.2, 1.2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, duration)

# Plot 2: Modulated Signals
ax2 = fig.add_subplot(gs[1, 0])
time_full = t[:len(psk_modulated)]
ax2.plot(time_full, psk_modulated, linewidth=1, color='blue', alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.set_title('PSK Modulated Signal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, min(0.1, duration))

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_full, mfsk_modulated, linewidth=1, color='red', alpha=0.7)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Amplitude')
ax3.set_title('M-ary FSK Modulated Signal (M=4)')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, min(0.1, duration))

# Plot 3: Noisy Signals
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(time_full, psk_noisy, linewidth=1, color='green', alpha=0.7)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude')
ax4.set_title(f'PSK + Noise (SNR = {snr_db} dB)')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, min(0.1, duration))

ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(time_full, mfsk_noisy, linewidth=1, color='orange', alpha=0.7)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Amplitude')
ax5.set_title(f'M-ary FSK + Noise (SNR = {snr_db} dB)')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, min(0.1, duration))

# Plot 4: Power Spectral Density
ax6 = fig.add_subplot(gs[3, 0])
f_psk, psd_psk = calculate_psd(psk_modulated, sampling_rate)
ax6.semilogy(f_psk, 10**(psd_psk/10))
ax6.set_xlabel('Frequency (Hz)')
ax6.set_ylabel('Power Spectral Density')
ax6.set_title('PSD of PSK Modulated Signal')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, sampling_rate/2)

ax7 = fig.add_subplot(gs[3, 1])
f_mfsk, psd_mfsk = calculate_psd(mfsk_modulated, sampling_rate)
ax7.semilogy(f_mfsk, 10**(psd_mfsk/10))
ax7.set_xlabel('Frequency (Hz)')
ax7.set_ylabel('Power Spectral Density')
ax7.set_title('PSD of M-ary FSK Modulated Signal')
ax7.grid(True, alpha=0.3)
ax7.set_xlim(0, sampling_rate/2)

# Plot 5: Constellation Diagrams
# For PSK: extract I component
t_const = np.arange(len(psk_modulated)) / sampling_rate
carrier_const = np.cos(2 * np.pi * carrier_freq * t_const)
i_component = psk_modulated * carrier_const

# Low-pass filter for constellation
b, a = butter(4, normalized_cutoff, btype='low')
i_filtered = lfilter(b, a, i_component)
q_filtered = np.zeros_like(i_filtered)

# Sample at bit centers
constellation_psk_tx = []
constellation_psk_rx = []

for i in range(0, len(i_filtered), samples_per_bit):
    sample_idx = i + samples_per_bit // 2
    if sample_idx < len(i_filtered):
        constellation_psk_tx.append([i_filtered[sample_idx], 0])
        
        # RX constellation from noisy signal
        i_rx = psk_noisy * carrier_const
        i_rx_filtered = lfilter(b, a, i_rx)
        constellation_psk_rx.append([i_rx_filtered[sample_idx], 0])

constellation_psk_tx = np.array(constellation_psk_tx)
constellation_psk_rx = np.array(constellation_psk_rx)

ax8 = fig.add_subplot(gs[4, 0])
ax8.scatter(constellation_psk_tx[:, 0], constellation_psk_tx[:, 1], 
           c='blue', alpha=0.6, label='Transmitted', s=30)
ax8.scatter(constellation_psk_rx[:, 0], constellation_psk_rx[:, 1], 
           c='red', alpha=0.6, label='Received (with noise)', s=30)
ax8.set_xlabel('In-Phase Component')
ax8.set_ylabel('Quadrature Component')
ax8.set_title('PSK Constellation Diagram')
ax8.grid(True, alpha=0.3)
ax8.legend()
ax8.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax8.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax8.set_aspect('equal')

# FSK Constellation (frequency vs time representation)
ax9 = fig.add_subplot(gs[4, 1])
# For FSK, plot instantaneous frequency vs time
inst_freq_psk = np.diff(np.unwrap(np.angle(signal.hilbert(psk_modulated)))) * sampling_rate / (2*np.pi)
inst_freq_mfsk = np.diff(np.unwrap(np.angle(signal.hilbert(mfsk_modulated)))) * sampling_rate / (2*np.pi)

time_freq = t[:-1]
ax9.plot(time_freq, inst_freq_mfsk, linewidth=0.5, alpha=0.7)
ax9.set_xlabel('Time (s)')
ax9.set_ylabel('Instantaneous Frequency (Hz)')
ax9.set_title('M-ary FSK - Frequency Variation')
ax9.grid(True, alpha=0.3)
ax9.set_xlim(0, min(0.1, duration))

# Plot 6: Demodulation Performance
ax10 = fig.add_subplot(gs[5, 0])
time_bits_short = np.arange(len(message_bits[:100])) / bit_rate
ax10.step(time_bits_short, message_bits[:100], where='post', 
         linewidth=2, label='Original', alpha=0.7)
ax10.step(time_bits_short, psk_demodulated[:100], where='post', 
         linewidth=1.5, label='Demodulated', linestyle='--', alpha=0.7)
ax10.set_xlabel('Time (s)')
ax10.set_ylabel('Bits')
ax10.set_title(f'PSK Demodulation Performance (BER = {psk_ber:.4f})')
ax10.legend()
ax10.grid(True, alpha=0.3)
ax10.set_ylim(-0.2, 1.2)

ax11 = fig.add_subplot(gs[5, 1])
ax11.step(time_bits_short, message_bits[:100], where='post', 
         linewidth=2, label='Original', alpha=0.7)
ax11.step(time_bits_short, mfsk_demodulated[:100], where='post', 
         linewidth=1.5, label='Demodulated', linestyle='--', alpha=0.7)
ax11.set_xlabel('Time (s)')
ax11.set_ylabel('Bits')
ax11.set_title(f'M-ary FSK Demodulation Performance (BER = {mfsk_ber:.4f})')
ax11.legend()
ax11.grid(True, alpha=0.3)
ax11.set_ylim(-0.2, 1.2)

plt.suptitle('Digital Modulation/Demodulation Analysis: PSK vs M-ary FSK', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Additional analysis: BER vs SNR
print("\nAnalyzing BER vs SNR...")
snr_values = np.arange(0, 21, 2)
psk_ber_vs_snr = []
mfsk_ber_vs_snr = []

for snr in snr_values:
    # Add noise at different SNR levels
    psk_noisy_var, _ = add_noise(psk_power, snr, psk_modulated)
    mfsk_noisy_var, _ = add_noise(mfsk_power, snr, mfsk_modulated)
    
    # Demodulate
    psk_demod_var = psk_demodulate(psk_noisy_var, carrier_freq, sampling_rate, samples_per_bit)
    mfsk_demod_var = mfsk_demodulate(mfsk_noisy_var, carrier_freq, sampling_rate, samples_per_bit, M=4)
    
    # Calculate BER
    psk_ber_var = np.sum(message_bits != psk_demod_var) / len(message_bits)
    mfsk_ber_var = np.sum(message_bits != mfsk_demod_var) / len(message_bits)
    
    psk_ber_vs_snr.append(psk_ber_var)
    mfsk_ber_vs_snr.append(mfsk_ber_var)

# Plot BER vs SNR
plt.figure(figsize=(10, 6))
plt.semilogy(snr_values, psk_ber_vs_snr, 'b-o', label='PSK', linewidth=2, markersize=8)
plt.semilogy(snr_values, mfsk_ber_vs_snr, 'r-s', label='M-ary FSK (M=4)', linewidth=2, markersize=8)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Bit Error Rate (BER)', fontsize=12)
plt.title('BER Performance Comparison: PSK vs M-ary FSK', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.ylim([1e-3, 1])
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"\nSystem Parameters:")
print(f"  • Modulation schemes: PSK and {4}-ary FSK")
print(f"  • Bit rate: {bit_rate} bps")
print(f"  • Carrier frequency: {carrier_freq} Hz")
print(f"  • Sampling rate: {sampling_rate} Hz")
print(f"  • Total bits transmitted: {num_bits}")
print(f"  • Signal duration: {duration} seconds")
print(f"  • SNR for main analysis: {snr_db} dB")

print(f"\nPerformance Results:")
print(f"  • PSK - Bit Error Rate: {psk_ber:.6f}")
print(f"  • M-ary FSK - Bit Error Rate: {mfsk_ber:.6f}")

print(f"\nObservations:")
if psk_ber < mfsk_ber:
    print("  • PSK performs better than M-ary FSK at the given SNR")
else:
    print("  • M-ary FSK performs better than PSK at the given SNR")
print("  • PSK has constant envelope but requires phase synchronization")
print("  • M-ary FSK is more robust to amplitude variations")
print("  • Higher order modulations (M>2) provide better bandwidth efficiency")
print("  • Constellation diagrams show the effect of noise on symbol decisions")
print("\nNote: BER decreases as SNR increases (as shown in the BER vs SNR plot)")
print("="*60)