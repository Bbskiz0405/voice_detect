import os

import torch
from pydub import AudioSegment


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_accuracy(output, label):
    prediction = torch.argmax(output.data, 1)
    correct = (prediction == label).sum().item()
    accuracy = correct / len(label)
    return accuracy


# convert .flac files to .wav files in the voice data folder
def flac2wav(input_path: str, output_path: str) -> None:
    flac_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.endswith('.flac')]

    for file in flac_files:
        print('Converting ' + file)
        gen_wav = AudioSegment.from_file(os.path.join(input_path, file))
        wav_filename = os.path.splitext(file)[0] + '.wav'
        gen_wav.export(os.path.join(output_path, wav_filename), format='wav')
    print('Done converting \n')
