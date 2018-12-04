import random
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as dsets

from utils import mkdir, save_checkpoint, load_checkpoint, cuda, reorganize, weights_init_normal, LambdaLR
from model import Generator, Discriminator

# make training behaviour deterministic
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic=True


class ItemPool():
    """
    This class represents a pool of generated images
    """
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def __call__(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.002, help='starting epoch')

epochs = 200
start_epoch = 0
batch_size = 1
lr = 0.0004
dataset_dir = '/media/external4T/a38iqbal/cycle_gan/datasets/apple2orange'

load_size = 286
crop_size = 256

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize(load_size),
     transforms.RandomCrop(crop_size),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

# transform = transforms.Compose([ transforms.Resize(int(crop_size*1.12), Image.BICUBIC),
#                 transforms.RandomCrop(crop_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

dataset_dirs = reorganize(dataset_dir)
a_train_data = dsets.ImageFolder(dataset_dirs['trainA'], transform=transform)
b_train_data = dsets.ImageFolder(dataset_dirs['trainB'], transform=transform)
a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)
a_train_loader = torch.utils.data.DataLoader(a_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
b_train_loader = torch.utils.data.DataLoader(b_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=3, shuffle=True, num_workers=4)
b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=3, shuffle=True, num_workers=4)



disc_a = Discriminator()
disc_b = Discriminator()
gen_a = Generator()
gen_b = Generator()

# weight initialization
disc_a.apply(weights_init_normal)
disc_b.apply(weights_init_normal)
gen_a.apply(weights_init_normal)
gen_b.apply(weights_init_normal)

MSE = nn.MSELoss()
L1 = nn.L1Loss()
cuda([disc_a, disc_b, gen_a, gen_b])

disc_a_optimizer = torch.optim.Adam(disc_a.parameters(), lr=lr, betas=(0.5, 0.999))
disc_b_optimizer = torch.optim.Adam(disc_b.parameters(), lr=lr, betas=(0.5, 0.999))
gen_a_optimizer = torch.optim.Adam(gen_a.parameters(), lr=lr, betas=(0.5, 0.999))
gen_b_optimizer = torch.optim.Adam(gen_b.parameters(), lr=lr, betas=(0.5, 0.999))
# gen_optimizer = torch.optim.Adam(itertools.chain(gen_b.parameters(), gen_a.parameters()), lr=lr, betas=(0.5, 0.999))

disc_a_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_a_optimizer, lr_lambda=LambdaLR(epochs, 0, 100).step)
disc_b_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_b_optimizer, lr_lambda=LambdaLR(epochs, 0, 100).step)
gen_a_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_a_optimizer, lr_lambda=LambdaLR(epochs, 0, 100).step)
gen_b_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_b_optimizer, lr_lambda=LambdaLR(epochs, 0, 100).step)
# gen_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=LambdaLR(epochs, 0, 100).step)

a_fake_pool = ItemPool()
b_fake_pool = ItemPool()


ckpt_dir = '/media/external4T/a38iqbal/cycle_gan/checkpoints/apple2orange'
mkdir(ckpt_dir)
try:
    ckpt = load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    disc_a.load_state_dict(ckpt['disc_a'])
    disc_b.load_state_dict(ckpt['disc_b'])
    gen_a.load_state_dict(ckpt['gen_a'])
    gen_b.load_state_dict(ckpt['gen_b'])
    disc_a_optimizer.load_state_dict(ckpt['disc_a_optimizer'])
    disc_b_optimizer.load_state_dict(ckpt['disc_b_optimizer'])
    gen_a_optimizer.load_state_dict(ckpt['gen_a_optimizer'])
    gen_b_optimizer.load_state_dict(ckpt['gen_b_optimizer'])
#     gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


with torch.no_grad():
    a_test_real = torch.autograd.Variable(iter(a_test_loader).next()[0])
    b_test_real = torch.autograd.Variable(iter(b_test_loader).next()[0])
a_test_real, b_test_real = cuda([a_test_real, b_test_real])\


# train
for epoch in range(start_epoch, epochs):
    for i, ((a_train_real, _), (b_train_real, _)) in enumerate(zip(a_train_loader, b_train_loader)):
        step = epoch * min(len(a_train_loader), len(b_train_loader)) + i + 1

        gen_a.train()
        gen_b.train()

        a_train_real, b_train_real = cuda([a_train_real, b_train_real])

        # generate fake images
        a_train_fake = gen_a(b_train_real)
        b_train_fake = gen_b(a_train_real)

        #         # real to real
        #         a_train_identity = gen_a(a_train_real)
        #         b_train_identity = gen_b(b_train_real)

        a_train_cycle = gen_a(b_train_fake)
        b_train_cycle = gen_b(a_train_fake)

        a_train_fake_disc = disc_a(a_train_fake)
        b_train_fake_disc = disc_b(b_train_fake)

        # generator loss
        real_label = cuda(torch.ones(a_train_fake_disc.size()))
        a_train_loss_gen = MSE(a_train_fake_disc, real_label)
        b_train_loss_gen = MSE(b_train_fake_disc, real_label)

        # identity loss
        #         a_train_loss_identity = L1(a_train_identity, a_train_real)
        #         b_train_loss_identity = L1(b_train_identity, b_train_real)

        # cyclic loss
        a_train_loss_cycle = L1(a_train_cycle, a_train_real)
        b_train_loss_cycle = L1(b_train_cycle, b_train_real)

        gen_loss = a_train_loss_gen + b_train_loss_gen
        identity_loss = 0  # 5.0 * (a_train_loss_identity + b_train_loss_identity)
        cycle_loss = 5.0 * (a_train_loss_cycle + b_train_loss_cycle)

        #         train_loss_gen = gen_loss + identity_loss + cycle_loss
        train_loss_gen = gen_loss + cycle_loss

        # generator backprop
        gen_a.zero_grad()
        gen_b.zero_grad()
        gen_a_optimizer.zero_grad()
        gen_b_optimizer.zero_grad()
        train_loss_gen.backward()
        gen_a_optimizer.step()
        gen_b_optimizer.step()
        #         gen_optimizer.step()

        a_train_fake = torch.Tensor(a_fake_pool(a_train_fake.detach().cpu()))
        b_train_fake = torch.Tensor(b_fake_pool(b_train_fake.detach().cpu()))
        #         a_train_fake = torch.Tensor(image_from_pool(a_fake_pool, a_train_fake.detach().cpu().numpy()))
        #         b_train_fake = torch.Tensor(image_from_pool(b_fake_pool, b_train_fake.detach().cpu().numpy()))
        a_train_fake, b_train_fake = cuda([a_train_fake, b_train_fake])

        # train discriminators
        a_train_real_disc = disc_a(a_train_real)
        a_train_fake_disc = disc_a(a_train_fake)
        b_train_real_disc = disc_b(b_train_real)
        b_train_fake_disc = disc_b(b_train_fake)
        real_label = cuda(torch.ones(a_train_fake_disc.size()))
        fake_label = cuda(torch.zeros(a_train_fake_disc.size()))

        # discriminator loss
        a_train_real_loss_disc = MSE(a_train_real_disc, real_label)
        a_train_fake_loss_disc = MSE(a_train_fake_disc, fake_label)
        b_train_real_loss_disc = MSE(b_train_real_disc, real_label)
        b_train_fake_loss_disc = MSE(b_train_fake_disc, fake_label)

        a_train_loss_disc = (a_train_real_loss_disc + a_train_fake_loss_disc) * 0.5
        b_train_loss_disc = (b_train_real_loss_disc + b_train_fake_loss_disc) * 0.5

        # discriminator backprop
        disc_a.zero_grad()
        disc_b.zero_grad()
        disc_a_optimizer.zero_grad()
        disc_b_optimizer.zero_grad()
        a_train_loss_disc.backward()
        b_train_loss_disc.backward()
        disc_a_optimizer.step()
        disc_b_optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch: (%3d) (%5d/%5d), Gen Loss: (%5f), ID Loss: (%5f), Cycle Loss (%5f),"
                  "Total Gen Loss (%5f), Disc A loss (%5f), Disc B Loss (%5f)"
                  % (epoch, i + 1, min(len(a_train_loader), len(b_train_loader)),
                     gen_loss, identity_loss, cycle_loss, train_loss_gen, a_train_loss_disc, b_train_loss_disc))

        if (i + 1) % 100 == 0:
            gen_a.eval()
            gen_b.eval()

            a_test_fake = gen_a(b_test_real)
            b_test_fake = gen_b(a_test_real)

            a_test_cycle = gen_a(b_test_fake)
            b_test_cycle = gen_b(a_test_fake)

            pic = torch.cat([a_test_real, b_test_fake, a_test_cycle,
                             b_test_real, a_test_fake, b_test_cycle],
                            dim=0).data / 2.0 + 0.5

            save_dir = '/media/external4T/a38iqbal/cycle_gan/sample_images/apple2orange'
            mkdir(save_dir)
            torchvision.utils.save_image(pic,
                                         '{}/Epoch_({})_({}of{}).jpg'.format(save_dir,
                                                                             epoch,
                                                                             i + 1,
                                                                             min(len(a_train_loader),
                                                                                 len(b_train_loader))),
                                         nrow=3)

            save_checkpoint({'epoch': epoch + 1,
                             'disc_a': disc_a.state_dict(),
                             'disc_b': disc_a.state_dict(),
                             'gen_a': gen_a.state_dict(),
                             'gen_b': gen_b.state_dict(),
                             'disc_a_optimizer': disc_a_optimizer.state_dict(),
                             'disc_b_optimizer': disc_b_optimizer.state_dict(),
                             'gen_a_optimizer': gen_a_optimizer.state_dict(),
                             'gen_b_optimizer': gen_b_optimizer.state_dict()},
                            '{}/Epoch_({}_iter_{}).ckpt'.format(ckpt_dir, epoch + 1, i + 1),
                            max_keep=2)

    #                 save_checkpoint_per_epoch({'epoch': epoch + 1,
    #                                  'gen_loss': gen_loss,
    #                                  'cycle_loss': cycle_loss,
    #                                  'disc_loss': a_train_loss_disc + b_train_loss_disc,
    #                                  'disc_a': disc_a.state_dict(),
    #                                  'disc_b': disc_a.state_dict(),
    #                                  'gen_a': gen_a.state_dict(),
    #                                  'gen_b': gen_b.state_dict(),
    #                                  'disc_a_optimizer': disc_a_optimizer.state_dict(),
    #                                  'disc_b_optimizer': disc_b_optimizer.state_dict(),
    #                                  'gen_a_optimizer': gen_a_optimizer.state_dict(),
    #                                  'gen_b_optimizer': gen_b_optimizer.state_dict()},
    #                                  '{}/Epoch_({}_iter_{}).ckpt'.format(ckpt_dir, epoch + 1, i + 1),
    #                                  epoch + 1)

    #         break

    # update learning rates
    disc_a_lr_scheduler.step()
    disc_b_lr_scheduler.step()
    gen_a_lr_scheduler.step()
    gen_b_lr_scheduler.step()
#     gen_lr_scheduler.step()