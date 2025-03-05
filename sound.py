import numpy as np
import wave
import struct
import pygame  # Import pygame for playing sound


# Function to generate alert sound
def generate_alert_wav(filename="alert.wav", duration=0.6, freq=800):
    sample_rate = 44100  # Hz
    amplitude = 32767  # Max amplitude for 16-bit audio
    num_samples = int(sample_rate * duration)

    # Create a sine wave for the alert sound
    waveform = (amplitude * np.sin(2.0 * np.pi * freq * np.arange(num_samples) / sample_rate)).astype(np.int16)

    with wave.open(filename, 'w') as file:
        file.setnchannels(1)  # Mono sound
        file.setsampwidth(2)  # 16-bit audio
        file.setframerate(sample_rate)
        file.writeframes(struct.pack('<' + ('h' * len(waveform)), *waveform))


# Function to play the generated sound
def play_alert_sound(filename="alert.wav"):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(filename)  # Load the generated alert sound
        pygame.mixer.music.play()  # Play the sound
    except Exception as e:
        print(f"Error playing the sound: {e}")


# Generate the alert sound and play it
generate_alert_wav()  # Generate the sound file
play_alert_sound()  # Play the sound
