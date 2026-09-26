import numpy as np
import matplotlib.pyplot as plt


def main():
    sample_rate = 44100
    duration = 1
    frequency = 1000
    time = np.arange(0, duration, 1 / sample_rate)
    signal = np.sin(2 * np.pi * frequency * time)


    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    amplitude = np.abs(2 * np.abs(spectrum) / len(signal))


    plt.figure(figsize=(12, 4))
    plt.plot(frequencies,amplitude)


    plt.xlabel("Frequency (Hz)")

    plt.ylabel("Amplitude")

    plt.title("FFT Spectrum")

    plt.xlim(-3000, 3000)
    plt.grid()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()