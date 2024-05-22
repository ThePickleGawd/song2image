from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import numpy as np
from PIL import Image
import io
import base64
import json

app = FastAPI()

origins = ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_audio_to_image(file: UploadFile):
    # Load audio file from bytes
    audio = AudioSegment.from_file(io.BytesIO(file.file.read()), format="mp3")

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
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Metadata
    metadata = {
        "sample_rate": sample_rate,
        "channels": channels,
        "original_shape": raw_data.shape,  # Save the original shape of the raw data
        "padded_shape": padded_data.shape,  # Save the shape after padding
        "max": raw_data.max(),
        "min": raw_data.min(),
    }

    return img_str, metadata


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    image, metadata = convert_audio_to_image(file)
    response = {"image": image, "metadata": metadata}
    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
