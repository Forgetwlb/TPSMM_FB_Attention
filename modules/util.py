from torch import nn
import torch.nn.functional as F
import torch
from timm.models.layers import DropPath
import numpy as np
import cv2
import matplotlib.colors as colors
import matplotlib.pyplot as plt

class TPS:
    '''
    TPS transformation, mode 'kp' for Eq(2) in the paper, mode 'random' for equivariance loss.TPS变换，文中Eq(2)的模态为kp，等方差损失的模态为random。
    '''
    def __init__(self, mode, bs, **kwargs):
        self.bs = bs
        self.mode = mode
        if mode == 'random':
            noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
            self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                        std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        elif mode == 'kp':
            kp_1 = kwargs["kp_1"]
            kp_2 = kwargs["kp_2"]
            device = kp_1.device
            kp_type = kp_1.type()
            self.gs = kp_1.shape[1]
            n = kp_1.shape[2]
            K = torch.norm(kp_1[:,:,:, None]-kp_1[:,:, None, :], dim=4, p=2)
            K = K**2
            K = K * torch.log(K+1e-9)

            one1 = torch.ones(self.bs, kp_1.shape[1], kp_1.shape[2], 1).to(device).type(kp_type)
            kp_1p = torch.cat([kp_1,one1], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
            P = torch.cat([kp_1p, zero], 2)
            L = torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2)
            L = torch.cat([L, P], 3)

            zero = torch.zeros(self.bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
            Y = torch.cat([kp_2, zero], 2)
            one = torch.eye(L.shape[2]).expand(L.shape).to(device).type(kp_type)*0.01
            L = L + one

            param = torch.matmul(torch.inverse(L), Y)
            self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)

            self.control_points = kp_1
            self.control_params = param[:,:,:n,:]
        else:
            raise Exception("Error TPS mode")

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, frame.shape[2], frame.shape[3], 2]
        if self.mode == 'kp':
            shape.insert(1, self.gs)
        grid = self.warp_coordinates(grid).view(*shape)
        return grid

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(coordinates.device)
        control_params = self.control_params.type(coordinates.type()).to(coordinates.device)

        if self.mode == 'kp':
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]

            distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(self.bs, control_points.shape[1], -1, 1, 2)

            distances = distances ** 2
            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
            transformed = transformed.permute(0, 1, 3, 2) + result

        elif self.mode == 'random':
            theta = theta.unsqueeze(1)
            transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
            transformed = transformed.squeeze(-1)
            ances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = ances ** 2

            result = distances.sum(-1)
            result = result * torch.log(result + 1e-9)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        else:
            raise Exception("Error TPS mode")

        return transformed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation 将一个关键点转换成高斯表示
    """

    coordinate_grid = make_coordinate_grid(spatial_size, kp.type()).to(kp.device)
    number_of_leading_dimensions = len(kp.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = kp.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape 预处理kp形状
    shape = kp.shape[:number_of_leading_dimensions] + (1, 1, 2)
    kp = kp.view(*shape)

    mean_sub = (coordinate_grid - kp)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def estimate_optical_flow(source_image, target_image):
    batch_size, channels, height, width = source_image.shape

    optical_flow = torch.zeros(batch_size, 2, height, width,  dtype=torch.float32)  # Initialize optical flow tensor

    for i in range(batch_size):
        source_img = source_image[i].permute(1, 2, 0).detach().cpu().numpy()
        target_img = target_image[i].permute(1, 2, 0).detach().cpu().numpy()

        source_img_gray = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
        target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

        # Compute optical flow using Farnebäck method
        flow = cv2.calcOpticalFlowFarneback(source_img_gray, target_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
        # hsv[..., 0] = ((ang * 180 / np.pi / 2) + 100) % 180
        # hsv[..., 1] = 255
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imwrite('output_image1.png', rgb)

        # Normalize and convert flow to torch tensor
        flow_normalized = flow / (width / 2)  # Normalize flow values
        flow_normalized = torch.from_numpy(flow_normalized).permute(2, 0, 1).detach()  # Convert to torch tensor

        optical_flow[i] = flow_normalized

    return optical_flow

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.创建一个给定空间大小的网格[-1,1]x[-1,1]。
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class CV2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock2dECA(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2dECA, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.eca = ECAAttention(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.eca(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution. 简单的块，保留空间分辨率。
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        #print('encoder:' ,outs[-1].shape)
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
            #print('encoder:' ,outs[-1].shape)
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []
        self.out_channels = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)
        # self.out_filters = block_expansion + in_features

    def forward(self, x, mode = 0):
        out = x.pop()
        outs = []
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        if(mode == 0):
            return out
        else:
            return outs


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels
        # self.out_filters = self.decoder.out_filters

    def forward(self, x, mode = 0):
        return self.decoder(self.encoder(x), mode)


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out


def to_homogeneous(coordinates):
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())

    return torch.cat([coordinates, ones], dim=-1)

def from_homogeneous(coordinates):
    return coordinates[..., :2] / coordinates[..., 2:3]


class DownBlock2dNB(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2dNB, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.NonLocalBlock = NonLocalBlock(out_features)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        out = self.NonLocalBlock(out)
        return out


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class UpBlock2dECA(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2dECA, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.ECAAttention = ECAAttention(out_features)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        out = self.ECAAttention(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def flow_warp_image(source_image, optical_flow):
    batch_size, _, height, width = source_image.shape
    # 创建坐标格点
    grid = make_coordinate_grid((height, width), type=source_image.type()).to(source_image.device)
    # 扩展坐标格点到与输入图像相同的维度
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape为 [N, H, W, 2]

    # 根据光流调整坐标格点
    warped_grid = grid + optical_flow.permute(0, 2, 3, 1).to(source_image.device)  # shape为 [N, 2, H, W]

    # 执行扭曲插值
    warped_image = F.grid_sample(source_image, warped_grid, align_corners=True)

    return warped_image