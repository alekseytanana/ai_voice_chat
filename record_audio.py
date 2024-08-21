import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

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
