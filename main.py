import socketio
from aiohttp import web
from pydantic import BaseModel
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset
import io
import base64

# Create a Socket.IO server
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

# Assuming the model and processor are loaded globally to avoid reloading them on each request
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# Here we directly use a specific xvector; for a real application, this should be dynamic or configurable.
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Define event handlers
@sio.event
async def connect(sid, environ):
    print('Client connected:', sid)

@sio.event
async def disconnect(sid):
    print('Client disconnected:', sid)

@sio.event
async def message(sid, data):
    print('Message from', sid, ':', data)
    try:
        text = data
        print(text)
        inputs = processor(text=text, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        # Save the speech to an in-memory file
        buffer = io.BytesIO()
        sf.write(buffer, speech.numpy(), samplerate=16000, format='wav')
        buffer.seek(0)
        # Convert speech to base64 string
        encoded_audio = base64.b64encode(buffer.read()).decode('utf-8')
        await sio.emit('audio_data', {'audio': encoded_audio, 'text': text}, room=sid)
    except Exception as e:
        # Handle exceptions
        await sio.emit('error', {'message': str(e)}, room=sid)

# Run the Socket.IO server
if __name__ == '__main__':
    port = 5000  # Choose a port number
    print(f'Server running on port {port}')
    web.run_app(app, port=port)
