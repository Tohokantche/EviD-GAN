#https://github.com/balansky/pytorch_gan
import math
import torch.nn.functional as F
from layers.categorical_batch_norm import CategoricalBatchNorm
from layers.spectral_norm import *

class Conv2DNormalGamma(nn.Module):
    def __init__(self, in_filters,out_filters=1):
        super(Conv2DNormalGamma, self).__init__()
        #nn.Conv2d(curr_dim, 1, 4)
        self.conv = SpectralNorm(nn.Conv2d(in_filters,out_filters*4, 4))

    def evidence(self, x):
        # return tf.exp(x)
        return torch.nn.functional.softplus(x)

    def forward(self, x):
        output = self.conv(x)
        mu, logv, logalpha, logbeta = torch.split(output, output.shape[1]//4, dim=1)  #tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)
    

class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 kernel_size=3, stride=1, padding=1, optimized=False, spectral_norm=1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.optimized = optimized
        self.hidden_channels = out_channels if not hidden_channels else hidden_channels

        self.conv1 = Conv2d(self.in_channels, self.hidden_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.conv2 = Conv2d(self.hidden_channels, self.out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, spectral_norm_pi=spectral_norm)
        self.s_conv = None
        torch.nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        if self.in_channels != self.out_channels or optimized:
            self.s_conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0,
                                 spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.s_conv.weight.data, 1.)

        self.activate = torch.nn.ReLU()

    def residual(self, input):
        x = self.conv1(input)
        x = self.activate(x)
        x = self.conv2(x)
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.optimized:
            x = torch.nn.functional.avg_pool2d(x, 2)
        if self.s_conv:
            x = self.s_conv(x)
        return x


    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x + x_r


class Gblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None, num_categories=None,
                 kernel_size=3, stride=1, padding=1, upsample=True):
        super(Gblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     upsample, spectral_norm=0)
        self.upsample = upsample
        self.num_categories = num_categories

        self.bn1 = self.batch_norm(self.in_channels)
        self.bn2 = self.batch_norm(self.hidden_channels)
        if upsample:
            # self.up = torch.nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
            self.up = lambda a: torch.nn.functional.interpolate(a, scale_factor=2)
        else:
            self.up = lambda a: None

    def batch_norm(self, num_features):
        return torch.nn.BatchNorm2d(num_features) if not self.num_categories \
            else CategoricalBatchNorm(num_features, self.num_categories)

    def residual(self, input, y=None):
        x = input
        x = self.bn1(x, y) if self.num_categories else self.bn1(x)
        x = self.activate(x)
        if self.upsample:
            x = self.up(x)
            # output_size = list(input.size())
            # output_size[-1] = output_size[-1] * 2
            # output_size[-2] = output_size[-2] * 2
            # x = self.up(x, output_size=output_size)
        x = self.conv1(x)
        x = self.bn2(x, y) if self.num_categories else self.bn2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x

    def shortcut(self, input):
        x = input
        if self.upsample:
            x = self.up(x)
        if self.s_conv:
            x = self.s_conv(x)
        return x

    def forward(self, input, y=None):
        x = self.residual(input, y)
        x_r = self.shortcut(input)
        return x + x_r


class Dblock(Block):

    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, stride=1, padding=1,
                 downsample=False, spectral_norm=1):
        super(Dblock, self).__init__(in_channels, out_channels, hidden_channels, kernel_size, stride, padding,
                                     downsample, spectral_norm)
        self.downsample = downsample

    def residual(self, input):
        x = self.activate(input)
        x = self.conv1(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def shortcut(self, input):
        x = input
        if self.s_conv:
            x = self.s_conv(x)
        if self.downsample:
            x = torch.nn.functional.avg_pool2d(x, 2)
        return x

    def forward(self, input):
        x = self.residual(input)
        x_r = self.shortcut(input)
        return x+ x_r


class BaseGenerator(torch.nn.Module):

    def __init__(self, z_dim, ch, d_ch=None, n_categories=None, bottom_width=4):
        super(BaseGenerator, self).__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.d_ch = d_ch if d_ch else ch
        self.n_categories = n_categories
        self.bottom_width = bottom_width
        self.dense = torch.nn.Linear(self.z_dim, self.bottom_width * self.bottom_width * self.d_ch)
        torch.nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        self.blocks = torch.nn.ModuleList()
        self.final = self.final_block()

    def final_block(self):
        conv = torch.nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv.weight.data, 1.)
        final_ = torch.nn.Sequential(
            torch.nn.BatchNorm2d(self.ch),
            torch.nn.ReLU(),
            conv,
            torch.nn.Tanh()
        )
        return final_


    def forward(self, input, y=None):
        x = self.dense(input)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        for block in self.blocks:
            x = block(x, y)
        x = self.final(x)
        return x


class ResnetGenerator(BaseGenerator):

    def __init__(self, ch=64, z_dim=128, n_categories=None, bottom_width=4):
        super(ResnetGenerator, self).__init__(z_dim, ch, ch*16, n_categories, bottom_width)
        self.blocks.append(Gblock(self.ch * 16, self.ch * 16, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch * 16, self.ch * 8, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch * 8, self.ch * 4, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch * 4, self.ch * 2, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch * 2, self.ch, upsample=True, num_categories=self.n_categories))


class ResnetGenerator32(BaseGenerator):

    def __init__(self, ch=256, z_dim=128, n_categories=None, bottom_width=4):
        super(ResnetGenerator32, self).__init__(z_dim, ch, ch, n_categories, bottom_width)
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch, self.ch, upsample=True, num_categories=self.n_categories))


class ResnetGenerator64(BaseGenerator):

    def __init__(self, ch=64, z_dim=128, n_categories=None, bottom_width=4):
        super(ResnetGenerator64, self).__init__(z_dim, ch, ch*16, n_categories, bottom_width)
        self.blocks.append(Gblock(self.ch*16, self.ch*8, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch*8, self.ch*4, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch*4, self.ch*2, upsample=True, num_categories=self.n_categories))
        self.blocks.append(Gblock(self.ch*2, self.ch, upsample=True, num_categories=self.n_categories))


class BaseDiscriminator(torch.nn.Module):

    def __init__(self, in_ch, out_ch=None, n_categories=0, l_bias=True, spectral_norm=1):
        super(BaseDiscriminator, self).__init__()
        self.activate = torch.nn.ReLU()
        self.ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.n_categories = n_categories
        self.blocks = torch.nn.ModuleList([Block(3, self.ch, optimized=True, spectral_norm=spectral_norm)])
        self.l = Linear(self.out_ch, 1, l_bias, spectral_norm_pi=spectral_norm)
        torch.nn.init.xavier_uniform_(self.l.weight.data, 1.)
        if n_categories > 0:
            self.l_y = Embedding(n_categories, self.out_ch, spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.l_y.weight.data, 1.)

    def forward(self, input, y=None):
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        x = torch.sum(x, (2, 3))
        output = self.l(x)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y*x, dim=1, keepdim=True)
        return output

class BaseEvidDiscriminator(torch.nn.Module):

    def __init__(self, in_ch, out_ch=None, n_categories=0, l_bias=True, spectral_norm=1):
        super(BaseEvidDiscriminator, self).__init__()
        self.activate = torch.nn.ReLU()
        self.ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.n_categories = n_categories
        self.blocks = torch.nn.ModuleList([Block(3, self.ch, optimized=True, spectral_norm=spectral_norm)])
        #self.l0 = Linear(self.out_ch, 128, l_bias, spectral_norm_pi=spectral_norm)
        self.l1 = LinearNormalGamma(self.out_ch, 4, l_bias, spectral_norm_pi=spectral_norm)
        #torch.nn.init.xavier_uniform_(self.l0.weight.data, 1.)
        torch.nn.init.xavier_uniform_(self.l1.weight.data, 1.)
        if n_categories > 0:
            self.l_y = Embedding(n_categories, self.out_ch, spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.l_y.weight.data, 1.)

    def forward(self, input, y=None):
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        feat=x
        #print(feat.shape)
        x = torch.sum(x, (2, 3))
        #x = self.l0(x)
        #x = F.leaky_relu(x, 0.1, inplace=True)
        output = self.l1(x)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y*x, dim=1, keepdim=True)
        return feat.view(-1,128*8*8),output
    
class BaseEvidDiscriminatorDiscrete(torch.nn.Module):

    def __init__(self, in_ch, out_ch=None, n_categories=0, l_bias=True, spectral_norm=1):
        super(BaseEvidDiscriminatorDiscrete, self).__init__()
        self.activate = torch.nn.ReLU()
        self.ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.n_categories = n_categories
        self.blocks = torch.nn.ModuleList([Block(3, self.ch, optimized=True, spectral_norm=spectral_norm)])
        #self.l0 = Linear(self.out_ch, 128, l_bias, spectral_norm_pi=spectral_norm)
        self.l1 = LinearBeta(self.out_ch, 2, l_bias, spectral_norm_pi=spectral_norm)
        #torch.nn.init.xavier_uniform_(self.l0.weight.data, 1.)
        torch.nn.init.xavier_uniform_(self.l1.weight.data, 1.)
        if n_categories > 0:
            self.l_y = Embedding(n_categories, self.out_ch, spectral_norm_pi=spectral_norm)
            torch.nn.init.xavier_uniform_(self.l_y.weight.data, 1.)

    def forward(self, input, y=None):
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.activate(x)
        feat=x
        #print(feat.shape)
        x = torch.sum(x, (2, 3))
        #x = self.l0(x)
        #x = F.leaky_relu(x, 0.1, inplace=True)
        output = self.l1(x)
        if y is not None:
            w_y = self.l_y(y)
            output += torch.sum(w_y*x, dim=1, keepdim=True)
        return feat.view(-1,128*8*8),output


class ResnetDiscriminator(BaseDiscriminator):

    def __init__(self, ch=64, n_categories=0, spectral_norm=1):
        super(ResnetDiscriminator, self).__init__(ch, ch*16, n_categories, spectral_norm=spectral_norm)
        self.blocks.append(Dblock(self.ch, self.ch*2, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*2, self.ch*4, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*4, self.ch*8, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*8, self.ch*16, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*16, self.ch*16, downsample=False, spectral_norm=spectral_norm))


class ResnetDiscriminator32(BaseDiscriminator):

    def __init__(self, ch=128, n_categories=0, spectral_norm=1):
        super(ResnetDiscriminator32, self).__init__(ch, ch, n_categories, l_bias=False, spectral_norm=spectral_norm)
        self.blocks.append(Dblock(self.ch, self.ch, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        
class ResnetEvidDiscriminator32(BaseEvidDiscriminator):

    def __init__(self, ch=128, n_categories=0, spectral_norm=1):
        super(ResnetEvidDiscriminator32, self).__init__(ch, ch, n_categories, l_bias=False, spectral_norm=spectral_norm)
        self.blocks.append(Dblock(self.ch, self.ch, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        
class ResnetEvidDiscriminatorDiscrete32(BaseEvidDiscriminatorDiscrete):

    def __init__(self, ch=128, n_categories=0, spectral_norm=1):
        super(ResnetEvidDiscriminatorDiscrete32, self).__init__(ch, ch, n_categories, l_bias=False, spectral_norm=spectral_norm)
        self.blocks.append(Dblock(self.ch, self.ch, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch, self.ch, downsample=False, spectral_norm=spectral_norm))


class ResnetDiscriminator64(BaseDiscriminator):

    def __init__(self, ch=64, n_categories=0, spectral_norm=1):
        super(ResnetDiscriminator64, self).__init__(ch, ch*16, n_categories, spectral_norm=spectral_norm)
        self.blocks.append(Dblock(self.ch, self.ch*2, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*2, self.ch*4, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*4, self.ch*8, downsample=True, spectral_norm=spectral_norm))
        self.blocks.append(Dblock(self.ch*8, self.ch*16, downsample=True, spectral_norm=spectral_norm))