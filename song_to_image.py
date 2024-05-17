import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


# Variables
sampling_rate = 22050

# Load the audio file (first 7 seconds)
y, sr = librosa.load("input/song.wav", duration=7, sr=sampling_rate)

time = np.linspace(0, len(y) / sr, num=len(y))

# Fourrier Transform
frequencies = np.fft.rfftfreq(len(y), d=1 / sampling_rate)
spectrum = np.abs(np.fft.rfft(y))

# Plot the frequency spectrum
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(frequencies, spectrum)
ax.set_title("Frequency Spectrum")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid(True)
ax.set_xlim(0, 1000)  # Limit x-axis for better visibility of peaks
plt.show()
