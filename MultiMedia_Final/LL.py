import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def read_audio(file_path):
    samplerate, data = wav.read(file_path)
    return samplerate, data

def frame_signal(data, frame_size, overlap_size, window_type):
    frame_step = frame_size - overlap_size
    num_frames = int(np.ceil(float(np.abs(len(data) - overlap_size)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + overlap_size
    z = np.zeros((pad_signal_length - len(data)))
    pad_signal = np.append(data, z)
    
    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    if window_type == 'hamming':
        window = np.hamming(frame_size)
    elif window_type == 'hann':
        window = np.hanning(frame_size)
    else:
        window = np.ones(frame_size)
    
    frames *= window
    return frames

def compute_energy(frames):
    return np.sum(frames ** 2, axis=1)

def compute_zero_crossings(frames):
    zero_crossings = np.diff(np.sign(frames)).astype(bool).sum(axis=1)
    return zero_crossings

def plot_results(original_signal, frames, energy, zero_crossings, samplerate):
    time_original = np.arange(0, len(original_signal)) / samplerate
    time_frames = np.arange(0, len(frames.flatten())) / samplerate
    frame_indices = np.arange(len(energy))

    plt.figure(figsize=(15, 12))

    plt.subplot(4, 1, 1)
    plt.plot(time_original, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 2)
    plt.plot(time_frames, frames.flatten())
    plt.title('Framed Signal (Flattened)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(4, 1, 3)
    plt.plot(frame_indices, energy)
    plt.title('Energy of Each Frame')
    plt.xlabel('Frame Index')
    plt.ylabel('Energy')

    plt.subplot(4, 1, 4)
    plt.plot(frame_indices, zero_crossings)
    plt.title('Zero Crossing Rate of Each Frame')
    plt.xlabel('Frame Index')
    plt.ylabel('Zero Crossings')

    plt.tight_layout()
    plt.show()

def main(audio_file_path, frame_size_sec, overlap_size_sec, window_type):
    samplerate, data = read_audio(audio_file_path)
    
    frame_size = int(frame_size_sec * samplerate)
    overlap_size = int(overlap_size_sec * samplerate)
    
    frames = frame_signal(data, frame_size, overlap_size, window_type)
    
    energy = compute_energy(frames)
    zero_crossings = compute_zero_crossings(frames)
    
    plot_results(data, frames, energy, zero_crossings, samplerate)
    
    # Outputs
    num_frames = frames.shape[0]
    print("Number of Frames:", num_frames)
    print("Energy Vector:", energy)
    print("Zero Crossing Vector:", zero_crossings)

# Example usage
audio_file_path = r"preamble.wav"
frame_size_sec = 0.09  # Frame size in seconds
overlap_size_sec = 0.0125  # Overlap size in seconds
window_type = 'hamming'  # Options: 'rectangular', 'hamming', 'hann'

main(audio_file_path, frame_size_sec, overlap_size_sec, window_type)
