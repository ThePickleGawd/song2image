import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import os
from PIL import Image

os.chdir(os.path.dirname(__file__))

# Variables
sampling_rate = 22050 / 2

# Load the audio file (first 7 seconds)
arr, _ = librosa.load("input/song_long.mp3", duration=8, sr=sampling_rate)

time = np.linspace(0, len(arr) / sampling_rate, num=len(arr))

# Fourrier Transform
frequencies = np.fft.rfftfreq(len(arr), d=1 / sampling_rate)
spectrum = np.abs(np.fft.rfft(arr))

import json
with open('output.txt', 'w') as filehandle:
    json.dump(spectrum.tolist(), filehandle)


# most = 0
# for x in spectrum:
#     most = max(most, x)
# print(most)

# print(frequencies[-1])

# Step 2: Normalize the spectrum data to the 8-bit range (0-255)
# spectrum_normalized = spectrum / np.max(spectrum)  # Normalize to range 0-1
# spectrum_normalized = (spectrum_normalized * 255).astype(np.uint8)  # Scale to range 0-255

# num_data_points = len(spectrum_normalized)
# side_length = int(np.ceil(np.sqrt(num_data_points)))  # Calculate the side length of the square
# image_data = spectrum_normalized.reshape((side_length, side_length))

# # Step 5: Create and save the image
# plt.imshow(image_data, cmap='viridis')
# plt.colorbar()
# plt.title('Frequency and Spectrum Data')
# plt.axis('off')  # Hide axes for better visualization
# plt.show()

# # Save the image
# plt.imsave(f'{os.path.dirname(__file__)}/frequency_spectrum_square_image.png', image_data, cmap='viridis')