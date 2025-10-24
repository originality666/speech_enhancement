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


def enhance_single_audio(generator, file_path, save_dir):
    
    noisy_slices = slice_signal_modify(file_path, window_size, 1, sample_rate)
    enhanced_speech = []

    for noisy_slice in tqdm(noisy_slices, desc=f'Enhancing {os.path.basename(file_path)}', leave=False):
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
    enhanced_speech = enhanced_speech / np.max(np.abs(enhanced_speech))
    enhanced_speech_int16 = np.int16(enhanced_speech * 32767)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"enhanced_{os.path.basename(file_path)}")
    wavfile.write(save_path, sample_rate, enhanced_speech_int16.T)

    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch Audio Enhancement')
    parser.add_argument('--input_dir', type=str, default='data/noisy_testset_wav', help='directory of noisy wav files')
    parser.add_argument('--epoch_name', type=str, required=True, help='generator epoch name')
    parser.add_argument('--output_dir', type=str, default='results/enhanced_audio', help='output directory for enhanced audio')
    opt = parser.parse_args()

    INPUT_DIR = opt.input_dir
    OUTPUT_DIR = opt.output_dir
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
    # Process all wav files
    # ------------------------
    wav_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.wav')]
    if not wav_files:
        print(f"No .wav files found in {INPUT_DIR}")
        exit(0)

    print(f"Found {len(wav_files)} audio files in {INPUT_DIR}")
    print("Start enhancing...\n")

    for wav_file in tqdm(wav_files, desc='Overall Progress'):
        input_path = os.path.join(INPUT_DIR, wav_file)
        output_path = enhance_single_audio(generator, input_path, OUTPUT_DIR)
        print(f"Saved: {output_path}")

    print(f"\nAll done! Enhanced audios saved in: {OUTPUT_DIR}")
