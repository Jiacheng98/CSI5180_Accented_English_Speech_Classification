from os import walk
import argparse
import shutil

import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import torch
torch.manual_seed(1)

from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from MLP import MLP
from TransformerModel import TransformerModel

from train_test import *


class AccentDataset(Dataset):
    """Accent dataset."""

    def __init__(self, data_dir_path):

        self.data_dir_path = data_dir_path
        self.x_tensor = []
        self.y_tensor = []
        self.initializeXYTensor()

    def initializeXYTensor(self):

        label_dic = {"AU":0, "UK": 1, "US": 2}

        for each_accent in accent_list:
            directory_name = f"{self.data_dir_path}/{each_accent}"
            for (dirpath, dirnames, filename_list) in walk(directory_name):
                for wav_file_name in filename_list:
                    if not wav_file_name.startswith ('.') and wav_file_name.endswith('.wav'):
                        wav_file_path = f"{dirpath}/{wav_file_name}"
                        data, sample_rate = librosa.load(wav_file_path, sr=None)
                        data = fix_audio_length(data)
                        mfccs, n_fft, hop_length = MFCC_features(data)
                        scaled_mfccs = normalized_MFCC_fetures(mfccs)
                        self.x_tensor.append(scaled_mfccs)
                        self.y_tensor.append(label_dic[each_accent])

        self.x_tensor = torch.FloatTensor(np.array(self.x_tensor))
        self.y_tensor = torch.tensor(np.array(self.y_tensor))

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]



def main():
    global dataset_path, text_path, dataset_list, accent_list

    dataset_path = "UT-Podcast/audio"
    text_path = "UT-Podcast/text"
    dataset_list = ["train", "test"]
    accent_list = ["AU", "UK", "US"]

    # show_dataset_examples()

    train_data = AccentDataset(data_dir_path=f"{dataset_path}/train")
    test_data = AccentDataset(data_dir_path=f"{dataset_path}/test")

    # Split train into train+val
    train_length= int(len(train_data) * 0.9)
    valid_length = len(train_data) - train_length
    train_data, val_data = torch.utils.data.random_split(train_data, [train_length, valid_length])

    train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False)
    # print(test_data[0][0].shape)

    num_frame, num_mfcc_feature = test_data[0][0].shape[0], test_data[0][0].shape[1]
    num_class = 3
    # In multi-class (Single label categorical) nn.CrossEntropyLoss applies the softmax for you
    # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    mlp_model = MLP(num_frame, num_mfcc_feature, num_class).to(device_mlp)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=args.lr)
    mlp_model = train(train_loader, val_loader, mlp_model, optimizer, criterion, args.epoch, args.patience, device_mlp, model_name="mlp")
    test(test_loader, mlp_model, device_mlp, model_name="mlp")


    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    transformer_model = TransformerModel(num_mfcc_feature, nhead, d_hid, nlayers, dropout, num_class).to(device_transformer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=args.lr)
    transformer_model = train(train_loader, val_loader, transformer_model, optimizer, criterion, args.epoch, args.patience, device_transformer, model_name="transformer")
    test(test_loader, transformer_model, device_transformer, model_name="transformer")



def show_dataset_examples():
    '''
    
    Plot a sample for AU, UK, and US accents, respectively.

    '''
    for each_accent in accent_list:
        directory_name = f"{dataset_path}/train/{each_accent}"
        for (dirpath, dirnames, filename_list) in walk(directory_name):
            for wav_file_name in filename_list:
                if not wav_file_name.startswith ('.') and wav_file_name.endswith('.wav'):
                    wav_file_path = f"{dirpath}/{wav_file_name}"

                    # librosa.load: amplitude is normalized between -1 to 1, and signals are converted to a mono signal by default
                    # "sr=None": preserve the native sampling rate of the file (8000hz)
                    data, sample_rate = librosa.load(wav_file_path, sr=None)
                    # data: a 2D array (the amplitude of the waveform at sample t, the number of channels in the audio)
                    print(f"Audio sampling rate: {sample_rate}")
                    print(f"Before fixed length, Data shape: {data.shape}, Audio duration: {data.shape[0]/sample_rate}")
                    
                    # write the audio (.wav)
                    sf.write(f'data_example/{wav_file_name}', data, sample_rate)

                    # write the audio(.wav) and find the ground truth text (for google speech-to-text API with accent info.)
                    # sf.write(f'accent_example/{wav_file_name}', data, sample_rate)
                    # text_directory_name = f"{text_path}/train/{each_accent}"
                    # accent_example(wav_file_name, text_directory_name)

                    # Plot the original waveform and trimmed/zero-padding waveform
                    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                    img1 = librosa.display.waveshow(data, sr=sample_rate, x_axis="s", ax = ax[0])
                    ax[0].set(title='Original Waveform')
                    ax[0].label_outer()
                    ax[0].set_ylabel('Normalized amplitude')

                    # problem: audio input has various length
                    # solver: use librosa fix_length library to trim/zero-padding audio input to a fixed size, 15s, 15 * 8000 = 120, 000 samples
                    data = fix_audio_length(data)
                    print(f"After fixed length, Data shape: {data.shape}, Audio duration: {data.shape[0]/sample_rate}")

                    img2 = librosa.display.waveshow(data, sr=sample_rate, x_axis="s", ax = ax[1])
                    ax[1].set(title='Trimmed/Zero-Padding Waveform')
                    ax[1].label_outer()
                    ax[1].set_ylabel('Normalized amplitude')
                    plt.savefig(f"data_example/{each_accent}_{wav_file_name}_waveform.pdf")
                    plt.close()


                    mfccs, n_fft, hop_length = MFCC_features(data)
                    
                    # Plot Mel spectrogram 
                    plt.figure()
                    # Convert an amplitude spectrogram to dB-scaled spectrogram.
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
                    librosa.display.specshow(D, y_axis='mel', sr=sample_rate, x_axis='s')
                    plt.title('Mel spectrogram')
                    plt.colorbar(format="%+2.f dB")
                    plt.savefig(f"data_example/{each_accent}_{wav_file_name}_mel_spectrogram.pdf")
                    plt.close()

                    # Plot MFCCs and Normalized MFCCs
                    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                    img1 = librosa.display.specshow(mfccs, sr=sample_rate, x_axis='s', ax=ax[0],
                                                    n_fft=n_fft, hop_length=hop_length)
                    ax[0].set(title='Original MFCC features')
                    ax[0].label_outer()
                    fig.colorbar(img1, ax=[ax[0]])

                    scaled_mfccs = normalized_MFCC_fetures(mfccs)

                    img2 = librosa.display.specshow(scaled_mfccs.T, sr=sample_rate, x_axis='s', ax=ax[1],
                                                    n_fft=n_fft, hop_length=hop_length)
                    ax[1].set(title='Mean normalized MFCC features')
                    ax[1].label_outer()
                    fig.colorbar(img2, ax=[ax[1]])

                    fig.savefig(f"data_example/{each_accent}_{wav_file_name}_mel_spectrogram_mfcc.pdf")
                    plt.close()
                    break


# def accent_example(wav_file_name, text_directory_name):
#     for (dirpath, dirnames, filename_list) in walk(text_directory_name):
#         for text_file_name in filename_list:
#             if wav_file_name[:-4] in text_file_name:
#                 src = f"{text_directory_name}/{text_file_name}"
#                 dst = f"accent_example/{text_file_name}"
#                 shutil.copy(src, dst)
#                 break



def fix_audio_length(data):

    data = librosa.util.fix_length(data, size=120000)

    return data



def MFCC_features(audio_data):
    '''
    
    Extract MFCC features from audios. It converts the audio into features based on the frequency and time characters.
    
    '''

    sample_rate = 8000
    n_fft = int(0.028*sample_rate)
    hop_length=int(0.01*sample_rate)
    # n_mfcc: No. of MFCC features for each frame, n_fft: frame length, hop_length: the frame stride
    # Eg. Set the frame length to 28 ms and the stride to 10 ms 
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc = args.mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # output dimension: (No. of NFCC features for each frame/window, No. of frames)
    # No. of frames = 1 + (seconds) * (sample rate) / (hop_length) = 1 + 15 * 8000 / (0.01 * 8000) = 1501
    # print(f"For each audio, computer {mfccs.shape[0]} MFCCs over {mfccs.shape[1]} frames")
    # print(f"MFCC features of the first frame: {mfccs[:, 0]}")

    return mfccs, n_fft, hop_length




def normalized_MFCC_fetures(mfccs):

    # the mean and standard deviation are computed for each audio file individually 
    # because the channel conditions are also different for each file. 
    # Thus, each file is standardized with its own mean and standard deviation. 
    # Otherwise, one silently assumes constant channel conditions among all the files?
    mfccs = mfccs.T
    mean = np.mean(mfccs, axis=0)
    std = np.std(mfccs, axis=0)

    mfccs_normalized = (mfccs - mean)/std
    return mfccs_normalized


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", help = "Batch size", default=1)
    parser.add_argument("-lr", help = "Lerning rate", default=1e-3)
    parser.add_argument("-epoch", help = "Epoch", default=100)
    parser.add_argument("-patience", help = "Patience", default=30)
    parser.add_argument("-mfcc", help = "Number of MFCC features", default=40)
    args = parser.parse_args()

    device_mlp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_transformer = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"MLP device: {device_mlp}")
    print(f"Transformer device: {device_transformer}")
    main()