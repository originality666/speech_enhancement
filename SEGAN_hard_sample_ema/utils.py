import os

import numpy as np
import torch
from torch.utils import data

from data_preprocess import serialized_test_folder, serialized_train_folder


def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.

    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal batch
    """
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    # --- 修改：添加 file_list 参数 ---
    def __init__(self, data_type, file_list=None):

        if data_type == 'train':
            data_path = serialized_train_folder
        else:
            data_path = serialized_test_folder
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))

        self.data_type = data_type
        
        # --- 修改：根据 file_list 加载文件 ---
        if file_list is None:
            # 正常模式：加载目录下所有文件
            self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
        else:
            # 难样本模式：只加载 file_list 中指定的文件
            print(f"Loading {len(file_list)} specified files from {data_path}")
            self.file_names = [os.path.join(data_path, basename) for basename in file_list]
            # 确保文件存在
            self.file_names = [f for f in self.file_names if os.path.exists(f)]


    def reference_batch(self, batch_size):
        """
        Randomly selects a reference batch from dataset.
        Reference batch is used for calculating statistics for virtual batch normalization operation.

        Args:
            batch_size(int): batch size

        Returns:
            ref_batch: reference batch
        """
        # --- 修改：确保有足够的文件来创建 ref batch ---
        num_available_files = len(self.file_names)
        if num_available_files == 0:
            raise ValueError("No files found in AudioDataset to create a reference batch.")
        
        # 如果可用文件少于 batch_size，则有放回地采样
        replace = num_available_files < batch_size
        
        ref_file_names = np.random.choice(self.file_names, batch_size, replace=replace)
        ref_batch = np.stack([np.load(f) for f in ref_file_names])

        ref_batch = emphasis(ref_batch, emph_coeff=0.95)
        return torch.from_numpy(ref_batch).type(torch.FloatTensor)

    def __getitem__(self, idx):
        # --- 修改：获取文件名和 basename ---
        file_path = self.file_names[idx]
        basename = os.path.basename(file_path)

        pair = np.load(file_path)
        pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        noisy = pair[1].reshape(1, -1)
        
        if self.data_type == 'train':
            clean = pair[0].reshape(1, -1)
            # --- 修改：返回 basename, pair, clean, noisy ---
            return basename, torch.from_numpy(pair).type(torch.FloatTensor), torch.from_numpy(clean).type(
                torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)
        else:
            # --- 修改：返回 basename, noisy (与原始行为一致) ---
            return basename, torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)