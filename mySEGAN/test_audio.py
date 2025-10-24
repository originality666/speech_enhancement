import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm

from data_preprocess import slice_signal_modify, window_size, sample_rate
from model import Generator
from utils import emphasis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('--file_name', type=str, required=True, help='audio file name')
    parser.add_argument('--epoch_name', type=str, required=True, help='generator epoch name')
    opt = parser.parse_args()

    FILE_NAME = opt.file_name
    EPOCH_NAME = opt.epoch_name

    # ------------------------
    # Load model checkpoint
    # ------------------------
    generator = Generator()

    checkpoint_path = EPOCH_NAME
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join('epochs', EPOCH_NAME)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        generator.cuda()
    generator.eval()

    # ------------------------
    # Process input audio
    # ------------------------
    noisy_slices = slice_signal_modify(FILE_NAME, window_size, 1, sample_rate)
    enhanced_speech = []

    for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
        z = torch.randn(1, 1024, 8)
        noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)

        if torch.cuda.is_available():
            noisy_slice, z = noisy_slice.cuda(), z.cuda()

        noisy_slice, z = Variable(noisy_slice), Variable(z)
        generated_speech = generator(noisy_slice, z).data.cpu().numpy()
        generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
        generated_speech = generated_speech.reshape(-1)
        enhanced_speech.append(generated_speech)

    enhanced_speech = np.array(enhanced_speech).reshape(1, -1)

    # ------------------------
    # Save enhanced audio
    # ------------------------
    save_dir = os.path.join('data', 'enhanced_audio')
    os.makedirs(save_dir, exist_ok=True)

    file_name = os.path.join(
        save_dir,
        f"enhanced_{os.path.basename(FILE_NAME).split('.')[0]}.wav"
    )

    # 🔹 convert to int16 for compatibility
    enhanced_speech = enhanced_speech / np.max(np.abs(enhanced_speech))
    enhanced_speech_int16 = np.int16(enhanced_speech * 32767)

    wavfile.write(file_name, sample_rate, enhanced_speech_int16.T)
    print(f"\nEnhanced audio saved to: {file_name}")
