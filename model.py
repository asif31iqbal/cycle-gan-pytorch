import torch.nn as nn


def conv_general(input_dim, output_dim, kernel_size, stride, padding=0,
                 norm=nn.InstanceNorm2d, normalize=True, activate=True, relu_factor=0):
    ops = list()
    conv_layer = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=False)
    #     truncated_normal_(conv_layer.weight, std=0.02)

    ops.append(conv_layer)

    if normalize:
        ops.append(norm(output_dim))

    if activate:
        if relu_factor:
            relu = nn.LeakyReLU(relu_factor)
        else:
            relu = nn.ReLU()
        ops.append(relu)

    return nn.Sequential(*ops)


def deconv_general(input_dim, output_dim, kernel_size, stride, padding=0, output_padding=0,
                   norm=nn.InstanceNorm2d, normalize=True, activate=True, relu_factor=0):
    ops = list()
    deconv_layer = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride,
                                      padding, output_padding, bias=False)
    #     truncated_normal_(deconv_layer.weight, std=0.02)
    ops.append(deconv_layer)

    if normalize:
        ops.append(norm(output_dim))

    if activate:
        if relu_factor:
            relu = nn.LeakyReLU(relu_factor)
        else:
            relu = nn.ReLU()
        ops.append(relu)

    return nn.Sequential(*ops)


class ResidualBlock(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()

        self.refl_pad = nn.ReflectionPad2d(1)
        self.conv_general = conv_general(input_dim, output_dim, 3, 1)
        self.conv = nn.Conv2d(output_dim, output_dim, 3, 1)
        self.instance_norm = nn.InstanceNorm2d(output_dim)

    def forward(self, x):
        o = self.refl_pad(x)
        o = self.conv_general(x)
        o = self.refl_pad(x)
        o = self.conv(x)
        o = self.instance_norm(x)

        return x + o


class Generator(nn.Module):

    def __init__(self, channels=64, residual_blocks=9):
        super(Generator, self).__init__()
        # 3 input image channels, 2566 output channels, 7*7 square convolution
        # kernel
        self.residual_blocks = residual_blocks
        self.refl_pad = nn.ReflectionPad2d(3)

        self.conv_general1 = conv_general(3, channels, 7, 1)
        self.conv_general2 = conv_general(channels, channels * 2, 3, 2, 1)
        self.conv_general3 = conv_general(channels * 2, channels * 4, 3, 2, 1)

        self.res_block = ResidualBlock(channels * 4, channels * 4)

        self.deconv_general1 = deconv_general(channels * 4, channels * 2, 3, 2, 1, 1)
        self.deconv_general2 = deconv_general(channels * 2, channels, 3, 2, 1, 1)

        self.conv = nn.Conv2d(channels, 3, 7, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # encoder
        x = self.refl_pad(x)
        x = self.conv_general1(x)
        x = self.conv_general2(x)
        x = self.conv_general3(x)

        # transformer
        for i in range(self.residual_blocks):
            x = self.res_block(x)

        # decoder
        x = self.deconv_general1(x)
        x = self.deconv_general2(x)
        x = self.refl_pad(x)
        x = self.conv(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, channels=64):
        super(Discriminator, self).__init__()
        # 3 input image channels, 2566 output channels, 7*7 square convolution
        # kernel

        self.conv_general1 = conv_general(3, channels, 4, 2, 1, normalize=False, relu_factor=0.2)
        self.conv_general2 = conv_general(channels, channels * 2, 4, 2, 1, relu_factor=0.2)
        self.conv_general3 = conv_general(channels * 2, channels * 4, 4, 2, 1, relu_factor=0.2)
        self.conv_general4 = conv_general(channels * 4, channels * 8, 4, 1, 1, relu_factor=0.2)
        self.conv = nn.Conv2d(channels * 8, 1, 4, 1, 1)

    def forward(self, x):
        x = self.conv_general1(x)
        x = self.conv_general2(x)
        x = self.conv_general3(x)
        x = self.conv_general4(x)
        x = self.conv(x)

        # Average pooling and flatten
        #         return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        return x