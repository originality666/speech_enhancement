import argparse
import os
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from model import Generator, Discriminator
from utils import AudioDataset, emphasis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=50, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')
    parser.add_argument('--resume', default=None, type=str, help='path to checkpoint to resume training')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('Loading data...')
    train_dataset = AudioDataset(data_type='train')
    test_dataset = AudioDataset(data_type='test')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE)

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
        ref_batch = ref_batch.cuda()
    ref_batch = Variable(ref_batch)

    print("# generator parameters:", sum(p.numel() for p in generator.parameters()))
    print("# discriminator parameters:", sum(p.numel() for p in discriminator.parameters()))

    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)

    # resume training if specified
    start_epoch = opt.start_epoch
    best_g_loss = float('inf')  # record the best loss
    if opt.resume is not None:
        print(f"Resuming training from checkpoint: {opt.resume}")
        checkpoint = torch.load(opt.resume)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_g_loss = checkpoint['best_g_loss']
        print(f"Resumed from epoch {start_epoch}, best_g_loss={best_g_loss:.4f}")

    os.makedirs('results', exist_ok=True)
    os.makedirs('epochs', exist_ok=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        generator.train()
        discriminator.train()

        train_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}")
        avg_g_loss = 0.0

        for train_batch, train_clean, train_noisy in train_bar:
            z = torch.randn(train_batch.size(0), 1024, 8)
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)

            # -------------------------------
            # Train Discriminator
            # -------------------------------
            discriminator.zero_grad()
            outputs = discriminator(train_batch, ref_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)
            clean_loss.backward()

            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1), ref_batch)
            noisy_loss = torch.mean(outputs ** 2)
            noisy_loss.backward()
            d_optimizer.step()

            # -------------------------------
            # Train Generator
            # -------------------------------
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

        # --------------------------------
        # Save best model & test generation
        # --------------------------------
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            print(f"New best model found at epoch {epoch+1} (g_loss={best_g_loss:.6f})")

            # save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'best_g_loss': best_g_loss
            }
            torch.save(checkpoint, 'epochs/best_model.pkl')

            # test and save generated samples
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
    