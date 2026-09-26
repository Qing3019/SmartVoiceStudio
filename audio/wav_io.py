import soundfile as sf

def load_audio(file_path):

    audio, sample_rate = sf.read(file_path, dtype='float32')

    return audio, sample_rate

def save_audio(file_path, audio, sample_rate):
    sf.write(file_path, audio, sample_rate)