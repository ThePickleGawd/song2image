from pydub import AudioSegment
import numpy as np
from PIL import Image
import json

# Load audio file
audio = AudioSegment.from_file("input/song.mp3")

# Convert to raw data
raw_data = np.array(audio.get_array_of_samples()).astype(np.float64)

# Get metadata
sample_rate = audio.frame_rate
channels = audio.channels

# Normalize audio data to range 0-255
normalized_data = (
    (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min()) * 255
).astype(np.uint8)

# Calculate the side length for the square image
side_len = int(np.ceil(np.sqrt(len(normalized_data))))

# Pad the data with zeros to make the length a perfect square
padded_data = np.concatenate(
    (normalized_data, np.zeros(side_len**2 - len(normalized_data), dtype=np.uint8))
)

# Reshape the padded data into a square image
image_data = padded_data.reshape((side_len, side_len))

# Save as image
image = Image.fromarray(image_data, mode="L")
image.save("output/image.png")

# Save metadata
metadata = {
    "sample_rate": sample_rate,
    "channels": channels,
    "original_shape": raw_data.shape,  # Save the original shape of the raw data
    "padded_shape": padded_data.shape,  # Save the shape after padding
    "max": raw_data.max(),
    "min": raw_data.min(),
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f)
