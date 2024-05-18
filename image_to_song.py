from pydub import AudioSegment
import numpy as np
from PIL import Image
import sounddevice as sd
import json

"""
metadata = {
    'sample_rate': sample_rate,
    'channels': channels,
    'original_shape': raw_data.shape,  # Save the original shape of the raw data
    'padded_shape': padded_data.shape,  # Save the shape after padding
    'max': raw_data.max(),
    'min': raw_data.min()
}
"""
# Load metadata
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# Load image
image = Image.open("output/image.png")
image_data = np.array(image).astype(np.float32)

# Denormalize data
denormalized_data = (image_data / 255 * (metadata['max'] - metadata['min']) + metadata['min'])

# Reshape data to original padded shape
denormalized_padded_data = denormalized_data.reshape(metadata['padded_shape'])

# Remove padding
decoded_data = denormalized_padded_data[:metadata['original_shape'][0]]

# Create audio segment from raw data
decoded_audio = AudioSegment(
    decoded_data.astype(np.int16).tobytes(),
    frame_rate=metadata['sample_rate'],
    sample_width=2,  # int16 implies 2 bytes per sample
    channels=metadata['channels']
)

# Export decoded audio to file
decoded_audio.export("output/image2song.mp3", format="mp3")