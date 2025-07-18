import numpy as np
from PIL import Image
from scipy.signal import butter, lfilter, convolve

# TODO make this into a whole class, that chains as Imagefun, that processes the image as audio.
# it must take a pillow image and return a pillow image in the process part, so it can be
# used standalone as well as a manipulation in imagefun (and the code is clean!)


# === Image ↔ Audio functions ===

def image_to_audio_rgb(image_path):
    img = Image.open(image_path).convert("RGB")
    data = np.array(img)
    shape = data.shape

    audio_data = []
    for i in range(3):
        channel = data[:, :, i].astype(np.float32)
        normalized = 2 * (channel / 255.0) - 1
        audio_data.append(normalized.flatten())
    
    return audio_data, shape

def audio_to_image_rgb(audio_data, shape):
    channels = []
    for i in range(3):
        audio = audio_data[i]
        img_channel = ((audio + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        channels.append(img_channel.reshape(shape[0], shape[1]))
    
    img_array = np.stack(channels, axis=2)
    return Image.fromarray(img_array)

# === Audio Effects ===

def lowpass_filter(data, cutoff=3000, fs=44100, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return lfilter(b, a, data)

def highpass_filter(data, cutoff=500, fs=44100, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high')
    return lfilter(b, a, data)

def add_echo(data, delay=1000, decay=0.5):
    """Simple feedback echo."""
    echoed = np.copy(data)
    for i in range(delay, len(data)):
        echoed[i] += decay * data[i - delay]
    return echoed

def add_reverb(data, size=500):
    """Very simple reverb using convolution with exponential decay."""
    impulse = np.exp(-np.linspace(0, 3, size))  # Decaying impulse
    return convolve(data, impulse, mode='same')

# === Example Usage ===

audio_channels, img_shape = image_to_audio_rgb("your_image.png")

# Apply effects to each channel
processed = []
for i, channel in enumerate(audio_channels):
    # Example chain of effects
    ch = lowpass_filter(channel, cutoff=2000)
    ch = add_echo(ch, delay=500, decay=0.4)
    ch = add_reverb(ch, size=300)
    processed.append(ch)

# Convert back to image
reconstructed_img = audio_to_image_rgb(processed, img_shape)
reconstructed_img.save("reconstructed_with_effects.png")
