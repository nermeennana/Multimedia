import pyaudio
import wave

import scipy.io.wavfile as wave2
import numpy as np

from pydub import AudioSegment
from pydub.playback import play

# __________________________________________________-
# Set audio parameters
FORMAT = pyaudio.paInt16  # 16-bit PCM format
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate (e.g., 44.1 kHz) (standard audio CD quality)
CHUNK = 1024              # Buffer size (number of audio frames per buffer)

# Initialize PyAudio (Creates an instance of the PyAudio class)
audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,    # Defines the sample rate.
                    input=True,   # (Indicates that this stream is for audio input (recording).)
                    frames_per_buffer=CHUNK)   # determines the buffer size

print("Recording... (Press Ctrl+C to stop)")

frames = []
try:
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    print("Recording stopped.")

# Stop recording
stream.stop_stream()     # Stops the audio stream.
stream.close()           # Closes the stream.
audio.terminate()        # Releases the PyAudio resources.
 
# Save recorded audio to a WAV file
with wave.open('recorded.wav', 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Saved recorded audio to 'recorded.wav'")

# ___________________________________________________

samplerate, data = wave2.read('recorded.wav')
# ___________________________________________________

# 1)get data size "number of samples in the file"
data_size_samples = len(data)
# print(data_size_samples)

# ___________________________________________________

# 2)get data size in bits
bits_per_sample = 16
data_size_bits = data_size_samples * bits_per_sample
# print(data_size_bits)

# ___________________________________________________

# 3)calcute the bits per samples (sampling rate)
# Given bit rate (bits per second)
bit_rate = 128000  # Example value (adjust as needed)

# Calculate bits per sample
bits_per_sample = bit_rate / samplerate

print(bits_per_sample)

# ___________________________________________________

# 5)determine the type of recording
# Determine the type of recording based on the number of channels
recording_type = "Mono" if data.ndim == 1 else "Stereo"
# ___________________________________________________

# 6)reverse the audio
reversed_data = data[::-1]

# # Set WAV file parameters (adjust as needed)
# sample_width = 2  # 2 bytes per sample (16-bit)
# channels = 1       # Mono audio
# framerate = samplerate  # Use the same sample rate as the original

# Create a new WAV file for writing
with wave.open('output_audio.wav', 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    # Write audio data (replace with your actual data)
    wf.writeframes(reversed_data.tobytes())


# ________________________________________________

# print the results:
print("1- Data size :", data_size_samples ,"sample")
print("2- Data size in bits:", data_size_bits , "bits")
print("3- Bit rate:", bit_rate ,"bps")
print("4- Type of recording:", recording_type )
print("5- Sampling rate:", samplerate , "samples per second")
# print("6- Length of sound file:", length_seconds, "seconds")
