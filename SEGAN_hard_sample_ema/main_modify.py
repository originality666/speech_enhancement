import argparse
import os
import torch
import torch.nn as nn
import numpy as np  # --- 新增导入 ---
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from model import Generator, Discriminator, update_ema  # --- 修改：导入 update_ema ---
from utils import AudioDataset, emphasis


def run_find_hard_samples(opt, generator, data_loader, ref_batch):
    """
    Phase 1: 遍历训练集，计算L1 loss，找出难样本。
    """
    print("\n--- Phase 1: Finding Hard Samples ---")
    generator.eval()  # 设置为评估模式
    all_losses = []
    all_basenames = []

    with torch.no_grad():
        find_bar = tqdm(data_loader, desc="Calculating sample losses")
        for basenames, train_batch, train_clean, train_noisy in find_bar:
            z = torch.randn(train_batch.size(0), 1024, 8)
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            
            # 转换为 Variable (尽管在新版PyTorch中非必须，但保持原代码风格)
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # --- 只计算 G loss (L1 part) ---
            generated_outputs = generator(train_noisy, z)
            l1_dist = torch.abs(generated_outputs - train_clean)
            
            # 计算每个样本的L1 loss (在通道和时间维度上取均值)
            g_cond_loss_per_sample = 100 * torch.mean(l1_dist, dim=[1, 2])

            all_losses.extend(g_cond_loss_per_sample.cpu().numpy())
            all_basenames.extend(basenames)

    # --- 计算平均损失并筛选难样本 ---
    avg_loss = np.mean(all_losses)
    hard_sample_basenames = [basename for basename, loss in zip(all_basenames, all_losses) if loss > avg_loss]

    print(f"\nTotal samples processed: {len(all_losses)}")
    print(f"Average L1 loss: {avg_loss:.6f}")
    print(f"Found {len(hard_sample_basenames)} hard samples (loss > avg)")

    # --- 保存难样本列表 ---
    try:
        with open(opt.hard_sample_file, 'w') as f:
            for basename in hard_sample_basenames:
                f.write(f"{basename}\n")
        print(f"Hard sample list saved to: {opt.hard_sample_file}")
    except Exception as e:
        print(f"Error saving hard sample file: {e}")

    print("--- Phase 1 Complete ---")


def run_train_hard_samples(opt, generator, g_ema, discriminator, test_data_loader, ref_batch):
    """
    Phase 2: 加载难样本列表，使用EMA进行训练。
    """
    print("\n--- Phase 2: Training on Hard Samples with EMA ---")
    
    # --- 加载难样本列表 ---
    if not os.path.exists(opt.hard_sample_file):
        print(f"Error: Hard sample file not found at {opt.hard_sample_file}")
        print("Please run with --find_hard_samples first.")
        return
        
    with open(opt.hard_sample_file, 'r') as f:
        hard_sample_basenames = [line.strip() for line in f if line.strip()]
    
    if not hard_sample_basenames:
        print("No hard samples found in file. Exiting.")
        return

    print(f"Loading {len(hard_sample_basenames)} hard samples for training.")

    # --- 创建难样本的 Dataset 和 DataLoader ---
    hard_dataset = AudioDataset(data_type='train', file_list=hard_sample_basenames)
    hard_data_loader = DataLoader(dataset=hard_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    # --- 为新训练阶段创建优化器 ---
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    best_g_loss = float('inf')

    for epoch in range(opt.start_epoch, opt.num_epochs):
        generator.train()
        discriminator.train()
        g_ema.eval()  # EMA 模型只用于评估和保存

        train_bar = tqdm(hard_data_loader, desc=f"Hard Epoch {epoch+1}")
        avg_g_loss = 0.0

        for basenames, train_batch, train_clean, train_noisy in train_bar:
            z = torch.randn(train_batch.size(0), 1024, 8)
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # --- 训练 D ---
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)
            clean_loss.backward()

            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)
            noisy_loss.backward()
            d_optimizer.step()

            # --- 训练 G ---
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)
            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            l1_dist = torch.abs(generated_outputs - train_clean)
            g_cond_loss = 100 * torch.mean(l1_dist)
            g_loss = g_loss_ + g_cond_loss

            g_loss.backward()
            g_optimizer.step()

            # --- 新增：更新 EMA 模型 ---
            update_ema(g_ema, generator, momentum=opt.ema_momentum)

            avg_g_loss += g_loss.item()
            train_bar.set_postfix({
                'd_clean': f'{clean_loss.item():.4f}',
                'd_noisy': f'{noisy_loss.item():.4f}',
                'g_total': f'{g_loss.item():.4f}',
                'g_cond': f'{g_cond_loss.item():.4f}'
            })

        avg_g_loss /= len(hard_data_loader)
        print(f"\nEpoch {epoch+1} average G loss: {avg_g_loss:.6f}")

        # --- 保存最佳模型 (EMA 模型) ---
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            print(f"New best EMA model found at epoch {epoch+1} (g_loss={best_g_loss:.6f})")

            checkpoint = {
                'epoch': epoch,
                # --- 修改：保存 G_EMA 的状态 ---
                'generator_state_dict': g_ema.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'best_g_loss': best_g_loss
            }
            # --- 修改：保存到新文件 ---
            torch.save(checkpoint, 'epochs/best_model_hard_ema.pkl')

            # --- 测试和保存生成样本 (使用 G_EMA) ---
            g_ema.eval() # 确保 EMA 模型处于评估模式
            with torch.no_grad():
                test_bar = tqdm(test_data_loader, desc='Generating best EMA samples')
                for test_file_names, test_noisy in test_bar:
                    z = torch.randn(test_noisy.size(0), 1024, 8)
                    if torch.cuda.is_available():
                        test_noisy, z = test_noisy.cuda(), z.cuda()
                    
                    # --- 修改：使用 G_EMA 生成 ---
                    fake_speech = g_ema(test_noisy, z).cpu().numpy()
                    fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)

                    for idx in range(fake_speech.shape[0]):
                        generated_sample = fake_speech[idx]
                        # --- 修改：保存到新文件 ---
                        file_name = os.path.join('results', f'{test_file_names[idx].replace(".npy", "")}_best_hard_ema.wav')
                        wavfile.write(file_name, sample_rate, generated_sample.T)

    print("\nHard sample training complete. Best EMA model saved as epochs/best_model_hard_ema.pkl.")


def run_original_training(opt, generator, discriminator, train_data_loader, test_data_loader, ref_batch, start_epoch, best_g_loss, g_optimizer, d_optimizer):
    """
    原始的训练循环 (稍作修改以适应新的 Dataloader)
    """
    print("\n--- Running Original Training Loop ---")
    
    for epoch in range(start_epoch, opt.num_epochs):
        generator.train()
        discriminator.train()

        train_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}")
        avg_g_loss = 0.0

        # --- 修改：解包来自 utils.py 的新元组 ---
        for basenames, train_batch, train_clean, train_noisy in train_bar:
            z = torch.randn(train_batch.size(0), 1024, 8)
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # --- Train D ---
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)
            clean_loss.backward()

            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)
            noisy_loss.backward()
            d_optimizer.step()

            # --- Train G ---
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair, ref_batch)
            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            l1_dist = torch.abs(generated_outputs - train_clean)
            g_cond_loss = 100 * torch.mean(l1_dist)
            g_loss = g_loss_ + g_cond_loss

            g_loss.backward()
            g_optimizer.step()

            avg_g_loss += g_loss.item()
            train_bar.set_postfix({
                'd_clean': f'{clean_loss.item():.4f}',
                'd_noisy': f'{noisy_loss.item():.4f}',
                'g_total': f'{g_loss.item():.4f}',
                'g_cond': f'{g_cond_loss.item():.4f}'
            })

        avg_g_loss /= len(train_data_loader)
        print(f"\nEpoch {epoch+1} average G loss: {avg_g_loss:.6f}")

        # --- 保存最佳模型 ---
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            print(f"New best model found at epoch {epoch+1} (g_loss={best_g_loss:.6f})")

            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'best_g_loss': best_g_loss
            }
            torch.save(checkpoint, 'epochs/best_model.pkl')

            # --- 测试和保存 ---
            generator.eval()
            with torch.no_grad():
                test_bar = tqdm(test_data_loader, desc='Generating best samples')
                for test_file_names, test_noisy in test_bar:
                    z = torch.randn(test_noisy.size(0), 1024, 8)
                    if torch.cuda.is_available():
                        test_noisy, z = test_noisy.cuda(), z.cuda()
                    fake_speech = generator(test_noisy, z).cpu().numpy()
                    fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)

                    for idx in range(fake_speech.shape[0]):
                        generated_sample = fake_speech[idx]
                        file_name = os.path.join('results', f'{test_file_names[idx].replace(".npy", "")}_best.wav')
                        wavfile.write(file_name, sample_rate, generated_sample.T)

    print("\nTraining complete. Best model saved as epochs/best_model.pkl.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')
    parser.add_argument('--resume', default=None, type=str, help='path to checkpoint to resume training (REQUIRED for new modes)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

    # --- 新增参数 ---
    parser.add_argument('--find_hard_samples', action='store_true', help='Run phase 1: find hard samples and save to file.')
    parser.add_argument('--train_hard_samples', action='store_true', help='Run phase 2: train on hard samples using EMA.')
    parser.add_argument('--hard_sample_file', default='hard_samples.txt', type=str, help='File to save/load hard sample list.')
    parser.add_argument('--ema_momentum', default=0.999, type=float, help='Momentum for EMA model update.')
    
    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # --- 新增：模式互斥检查 ---
    if opt.find_hard_samples and opt.train_hard_samples:
        raise ValueError("Cannot --find_hard_samples and --train_hard_samples simultaneously.")

    # --- 新增：检查是否必须提供 checkpoint ---
    if (opt.find_hard_samples or opt.train_hard_samples) and opt.resume is None:
        raise ValueError("Must provide --resume checkpoint to find hard samples or train with EMA.")

    # --- 加载数据 ---
    print('Loading data...')
    # --- 修改：为 Phase 1 设置 shuffle=False ---
    train_dataset = AudioDataset(data_type='train')
    train_shuffle = not opt.find_hard_samples
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=train_shuffle, num_workers=4)
    
    test_dataset = AudioDataset(data_type='test')
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # --- 创建模型 ---
    discriminator = Discriminator()
    generator = Generator()
    
    # --- 新增：EMA 模型 ---
    g_ema = None
    if opt.train_hard_samples:
        g_ema = Generator()
        print("EMA Generator created.")

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        if g_ema:
            g_ema.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)

    print("# generator parameters:", sum(p.numel() for p in generator.parameters()))
    print("# discriminator parameters:", sum(p.numel() for p in discriminator.parameters()))

    # --- 优化器 ---
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    start_epoch = opt.start_epoch
    best_g_loss = float('inf')  

    # --- 修改：Checkpoint 加载逻辑 ---
    if opt.resume is not None:
        print(f"Loading checkpoint: {opt.resume}")
        checkpoint = torch.load(opt.resume)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        # --- 新增：为 EMA 模型加载权重 ---
        if g_ema:
            print("Initializing EMA generator with checkpoint weights.")
            g_ema.load_state_dict(checkpoint['generator_state_dict'])

        # --- 修改：如果开始新阶段，重置 epoch 和 optimizers ---
        if opt.find_hard_samples or opt.train_hard_samples:
            # 仅加载权重，不恢复 optimizer 状态或 epoch
            start_epoch = 0 # 新阶段从 epoch 0 开始
            best_g_loss = float('inf')
            print(f"Loaded weights from epoch {checkpoint.get('epoch', 'N/A')}. Starting new training phase.")
        else:
            # 正常恢复训练
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_g_loss = checkpoint.get('best_g_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}, best_g_loss={best_g_loss:.4f}")

    os.makedirs('results', exist_ok=True)
    os.makedirs('epochs', exist_ok=True)

    # --- 新增：主逻辑分支 ---
    if opt.find_hard_samples:
        run_find_hard_samples(opt, generator, train_data_loader, ref_batch)
    elif opt.train_hard_samples:
        run_train_hard_samples(opt, generator, g_ema, discriminator, test_data_loader, ref_batch)
    else:
        run_original_training(opt, generator, discriminator, train_data_loader, test_data_loader, ref_batch, start_epoch, best_g_loss, g_optimizer, d_optimizer)