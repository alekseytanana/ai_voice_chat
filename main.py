from datetime import datetime
import openvino_genai as gen

from openvino_genai.py_generate_pipeline import GenerationConfig

import soundfile as sf
import torch
from playsound import playsound

from transformers import pipeline
from datasets import load_dataset

from contextlib import contextmanager

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

import speech_recognition as sr


# CHAT INIT
pipe = gen.LLMPipeline(r'C:\Users\atanana\PycharmProjects\openvino.genai\TinyLlama-1.1B-Chat-v1.0')
config = GenerationConfig(max_new_tokens=20, num_beam_groups=3, num_beams=15, diversity_penalty=1.5)
pipe.set_generation_config(config)
pipe.start_chat()

# TTS INIT
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.



@contextmanager
def duration():
    try:
        start = datetime.now()
        yield
    finally:
        finish = datetime.now()
        took = finish - start
        print(f'    (took {took.seconds} seconds)')


while True:
    # INPUT VOICE
    # Parameters
    DURATION = 10  # Duration of recording in seconds
    RATE = 44100  # Sample rate (44.1 kHz)
    OUTPUT_FILENAME = "output.wav"  # Output file name

    print("Recording...")

    # Record audio
    audio_data = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished

    print("Finished recording.")

    # Save the recorded data as a WAV file
    wav.write(OUTPUT_FILENAME, RATE, audio_data)


    # RECOGNIZE TEXT
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    audio_file = sr.AudioFile('output.wav')

    with audio_file as source:
        # Adjust for ambient noise and record the audio
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

    # Recognize speech using Google Web Speech API
    try:
        prompt = recognizer.recognize_google(audio_data)
        print("Extracted Text: ", prompt)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    # prompt = input('[Me]: ')
    # if prompt == 'quit':
    #     break
    with duration():
        # start = datetime.now()
        chat_answer = pipe(prompt)
        print(f'[Bot]: {chat_answer}')
        # finish = datetime.now()
        # took = finish - start
        # print(f'    (took {took.seconds} seconds)')

    with duration():
        speech = synthesiser(chat_answer[:50], forward_params={"speaker_embeddings": speaker_embedding})
        sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    playsound('speech.wav')

pipe.finish_chat()
