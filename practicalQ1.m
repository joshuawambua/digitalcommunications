clc;
clear;
close all;

%% -----------------------------
% PARAMETERS
%% -----------------------------
N = 100;              % Number of bits
bit_rate = 1000;      % 1 kbps
fc = 10000;           % Carrier frequency (10 kHz)
fs = 100000;          % Sampling frequency (100 kHz)
Tb = 1/bit_rate;      

t = 0:1/fs:N*Tb - 1/fs;

%% -----------------------------
% MESSAGE SIGNAL
%% -----------------------------
bits = randi([0 1], 1, N);
message = repelem(bits, fs*Tb);

%% -----------------------------
% BPSK MODULATION
%% -----------------------------
carrier = cos(2*pi*fc*t);
bpsk = (2*message - 1) .* carrier;

%% -----------------------------
% ADD NOISE (AWGN)
%% -----------------------------
noise = 0.5 * randn(size(bpsk));
bpsk_noisy = bpsk + noise;

%% -----------------------------
% M-FSK MODULATION (M = 4)
%% -----------------------------
M = 4;
freqs = fc + (0:M-1)*1000;
symbols = randi([0 M-1], 1, N);

fsk = zeros(size(t));

for i = 1:N
    idx_start = (i-1)*fs*Tb + 1;
    idx_end = i*fs*Tb;
    fsk(idx_start:idx_end) = cos(2*pi*freqs(symbols(i)+1) .* t(idx_start:idx_end));
end

% Add noise
fsk_noisy = fsk + 0.5*randn(size(fsk));

%% -----------------------------
% PSD (BPSK)
%% -----------------------------
[psd, f_psd] = pwelch(bpsk, [], [], [], fs);

%% -----------------------------
% DEMODULATION (BPSK)
%% -----------------------------
demod = bpsk_noisy .* carrier;
recovered_bits = demod > 0;

%% -----------------------------
% PLOTS (SEPARATE FIGURES)
%% -----------------------------

% 1. Message Signal
figure;
plot(message(1:1000));
title('Message Signal');
xlabel('Samples');
ylabel('Amplitude');
grid on;

% 2. BPSK Modulated Signal
figure;
plot(bpsk(1:1000));
title('BPSK Modulated Signal');
xlabel('Samples');
ylabel('Amplitude');
grid on;

% 3. BPSK + Noise
figure;
plot(bpsk_noisy(1:1000));
title('BPSK Signal with Noise');
xlabel('Samples');
ylabel('Amplitude');
grid on;

% 4. PSD of BPSK
figure;
plot(f_psd, 10*log10(psd));
title('Power Spectral Density (BPSK)');
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
grid on;

% 5. Constellation Diagram (BPSK)
figure;
scatter(bpsk_noisy(1:2000), zeros(1,2000));
title('Constellation Diagram (BPSK)');
xlabel('In-phase');
ylabel('Quadrature');
grid on;

% 6. M-FSK Signal
figure;
plot(fsk(1:1000));
title('M-FSK Signal (M = 4)');
xlabel('Samples');
ylabel('Amplitude');
grid on;

% 7. M-FSK + Noise
figure;
plot(fsk_noisy(1:1000));
title('M-FSK Signal with Noise');
xlabel('Samples');
ylabel('Amplitude');
grid on;