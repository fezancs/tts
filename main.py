from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 

from fastapi.responses import FileResponse

import torch

from transformers import VitsModel, AutoTokenizer

from scipy.io.wavfile import write

from pydantic import BaseModel  # Import BaseModel from pydantic

import os
import uvicorn 


app = FastAPI()



origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 



# Initialize the model and tokenizer once to avoid reloading them on each request

model = VitsModel.from_pretrained("facebook/mms-tts-eng")

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")



# Define a Pydantic model to parse the request body

class SynthesizeRequest(BaseModel):

    text: str



@app.post("/synthesize/")

def synthesize_text(request_body: SynthesizeRequest):  # Use the Pydantic model for parsing

    text = request_body.text  # Access the text field from the request body

    if not text:

        raise HTTPException(status_code=400, detail="No text provided for synthesis")



    sample_input = tokenizer(text, return_tensors="pt")

    with torch.no_grad():

        output = model(**sample_input).waveform

    # Assuming a sample rate of 22050, which is common for TTS models. 

    # You might need to adjust this based on the actual model you are using.

    sample_rate = 22050

    file_path = "output.wav"

    write(file_path, sample_rate, output.cpu().numpy()[0])

    return FileResponse(file_path, media_type="audio/wav", filename="output.wav")



if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
