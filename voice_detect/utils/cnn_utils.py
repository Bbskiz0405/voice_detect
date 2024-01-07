import os
import random

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from voice_detect.utils.gen_utils import create_dir

matplotlib.use("Agg")


# plot the mel-spectrogram for the single wav file input
def get_melss(wav_file: str, new_name: str) -> None:
    # get sample rate
    x, sr = librosa.load(wav_file, sr=None, res_type='kaiser_fast')

    # get headless figure
    fig = plt.figure(figsize=(1.0, 1.0))

    # remove the axes
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # get melss
    melss = librosa.feature.melspectrogram(y=x, sr=sr)
    librosa.display.specshow(librosa.power_to_db(melss, ref=np.max), y_axis='linear')

    # save plot as jpg
    plt.savefig(new_name, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


# prepare the cnn dataset of images
def prepare_cnn_dataset(src_path: str, dest_path: str) -> None:
    # src
    label_1_path = os.path.join(src_path, 'voice')
    label_0_path = os.path.join(src_path, 'not_voice')

    label_1_wavs = [file for file in os.listdir(label_1_path) if file.endswith(".wav")]
    label_0_wavs = [file for file in os.listdir(label_0_path) if file.endswith(".wav")]

    # shuffle wavs
    random.shuffle(label_1_wavs)
    random.shuffle(label_0_wavs)

    # dest
    label_1_train_path = os.path.join(dest_path, 'train', 'voice')
    label_0_train_path = os.path.join(dest_path, 'train', 'not_voice')

    create_dir(label_1_train_path)
    create_dir(label_0_train_path)

    label_1_test_path = os.path.join(dest_path, 'test', 'voice')
    label_0_test_path = os.path.join(dest_path, 'test', 'not_voice')

    create_dir(label_1_test_path)
    create_dir(label_0_test_path)

    label_1_size = round(len(label_1_wavs) * 0.8)
    for index, filename in enumerate(label_1_wavs):
        wav_filepath = os.path.join(label_1_path, filename)
        jpg_filename = filename.rsplit(".", 1)[0] + '.jpg'
        if index < label_1_size:
            jpg_filepath = os.path.join(label_1_train_path, jpg_filename)
        else:
            jpg_filepath = os.path.join(label_1_test_path, jpg_filename)
        get_melss(wav_filepath, jpg_filepath)

    label_0_size = round(len(label_0_wavs) * 0.8)
    for index, filename in enumerate(label_0_wavs):
        wav_filepath = os.path.join(label_0_path, filename)
        jpg_filename = filename.rsplit(".", 1)[0] + '.jpg'
        if index < label_0_size:
            jpg_filepath = os.path.join(label_0_train_path, jpg_filename)
        else:
            jpg_filepath = os.path.join(label_0_test_path, jpg_filename)
        get_melss(wav_filepath, jpg_filepath)
