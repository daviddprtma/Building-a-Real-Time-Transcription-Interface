import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Flask route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Variables
data_queue = Queue()
# TODO#1 - Initialize Variables for Transcription and Recorder
phrase_time = None
transcription = ['']
recorder = sr.Recognizer()
recorder.energy_threshold = 1000
recorder.dynamic_energy_threshold = False
# TODO#2 - Set Up and Initialize the Microphone
if 'linux' in platform:
    mic_name = 'pulse'  # Default microphone for Linux systems
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        if mic_name in name:
            source = sr.Microphone(sample_rate=16000, device_index=index)
            break
else:
    source = sr.Microphone(sample_rate=16000)
# TODO#3 - Load the Whisper Model for Audio Transcription
model = whisper.load_model("medium.en")
# TODO#4 - Set Recording and Phrase Timeout Parameters
record_timeout = 2  # Set default timeout for recording
phrase_timeout = 3   # Set default timeout for detecting new 
#phrases

def record_callback(_, audio: sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

# Start the transcription process in a separate thread
def start_transcription():
    # TODO#5 - Adjust Microphone Settings and Begin Listening
    global phrase_time
    with source:
        recorder.adjust_for_ambient_noise(source)
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                # TODO#6 - Check and Process the Audio Data in Real-Time
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                # TODO#7 - Convert the Audio Data for Transcription
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                # TODO#8 - Emit Transcription Results to the Frontend Using SocketIO
                # Emit the transcription to the frontend
                socketio.emit('transcription', {'data': text})

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

# TODO#9 - Start the Transcription in a Separate Thread
threading.Thread(target=start_transcription).start()


if __name__ == "__main__":
    socketio.run(app, host='', port=8080)