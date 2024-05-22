from pydub import AudioSegment
import numpy as np
from PIL import Image
import sounddevice as sd
from playsound import playsound
import json
import argparse
import os


def convert_image_to_audio(image_path: str, metadata_path: str):
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load image
    image = Image.open(image_path)
    image_data = np.array(image).astype(np.float32)

    # Denormalize data
    denormalized_data = (
        image_data / 255 * (metadata["max"] - metadata["min"]) + metadata["min"]
    )

    # Reshape data to original padded shape
    denormalized_padded_data = denormalized_data.reshape(metadata["padded_shape"])

    # Remove padding
    decoded_data = denormalized_padded_data[: metadata["original_shape"][0]]

    # Create audio segment from raw data
    decoded_audio = AudioSegment(
        decoded_data.astype(np.int16).tobytes(),
        frame_rate=metadata["sample_rate"],
        sample_width=2,  # int16 implies 2 bytes per sample
        channels=metadata["channels"],
    )

    # Export decoded audio to file
    output_audio_path = os.path.splitext(image_path)[0] + ".mp3"
    decoded_audio.export(output_audio_path, format="mp3")
    print(f"Audio saved to {output_audio_path}")

    playsound(output_audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play audio from a png file")
    parser.add_argument("--file", type=str, required=True, help="Path to the .png file")
    args = parser.parse_args()

    # Convert image back to audio using saved metadata
    convert_image_to_audio(image_path=args.file, metadata_path="metadata.json")
