import torch
import argparse
from utils import reorganize, cuda, load_checkpoint, mkdir
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from model import Generator


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--load_size', type=int, default=286, help='load size of the images')
parser.add_argument('--crop_size', type=int, default=256, help='crop size of the images during transformation')
parser.add_argument('--dataset', type=str, default='apple2orange', help='dataset name')
args = parser.parse_args()

dataset_dir = './{}'.format(args.dataset)
dataset_dirs = reorganize(dataset_dir)

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize(args.load_size),
     transforms.RandomCrop(args.crop_size),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

gen_a = Generator()
gen_b = Generator()

ckpt_dir = './checkpoints/{}'.format(args.dataset)
try:
    ckpt = load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    gen_a.load_state_dict(ckpt['gen_a'])
    gen_b.load_state_dict(ckpt['gen_b'])
#     gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
except:
    print(' [*] No checkpoint!')
    raise Exception('No checkpoint found. Cannot be tested.')


a_test_real = torch.tensor(iter(a_test_loader).next()[0], requires_grad=False)
b_test_real = torch.tensor(iter(b_test_loader).next()[0], requires_grad=False)
a_test_real, b_test_real = cuda([a_test_real, b_test_real])

gen_a.eval()
gen_b.eval()

for i, (b_test_real, _) in enumerate(b_test_loader):
    a_test_fake = gen_a(b_test_real)
    b_test_cycle = gen_b(a_test_fake)

    pic = torch.cat([b_test_real, a_test_fake, b_test_cycle],
                    dim=0).data / 2.0 + 0.5

    save_dir = './test_result/{}'.format(args.dataset)
    mkdir(save_dir)
    torchvision.utils.save_image(pic,
                                 '{}/test_{}.jpg'.format(save_dir, i + 1),
                                 nrow=3)

for i, (a_test_real, _) in enumerate(a_test_loader):
    b_test_fake = gen_b(a_test_real)
    a_test_cycle = gen_b(a_test_fake)

    pic = torch.cat([a_test_real, b_test_fake, a_test_cycle],
                    dim=0).data / 2.0 + 0.5

    save_dir = './sample_images/{}'.format(args.dataset)
    mkdir(save_dir)
    torchvision.utils.save_image(pic,
                                 '{}/test_{}.jpg'.format(save_dir, i + 1),
                                 nrow=3)
