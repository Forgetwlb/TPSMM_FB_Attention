import numpy as np
from matplotlib import pyplot as plt
import struct
import torch
import torch.nn.functional as F

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

from torchvision.utils import flow_to_image
import torch

def flow_to_image_torch(flow):
    flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
    flow_im = flow_to_image(flow)
    img = np.transpose(flow_im.numpy(), [1, 2, 0])
    print(img.shape)
    return img

def make_color_wheel(bins=None):
    """Build a color wheel.
    Args:
        bins(list or tuple, optional): Specify the number of bins for each
            color range, corresponding to six ranges: red -> yellow,
            yellow -> green, green -> cyan, cyan -> blue, blue -> magenta,
            magenta -> red. [15, 6, 4, 11, 13, 6] is used for default
            (see Middlebury).
    Returns:
        ndarray: Color wheel of shape (total_bins, 3).
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)
    print(RY)
    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]
    print(ry)
    num_bins = RY + YG + GC + CB + BM + MR
    print(num_bins)
    color_wheel = np.zeros((3, num_bins), dtype=np.float32)
    print(color_wheel)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        if i == 0:
            print(i, color)
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T

def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    # assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]
    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)  # 使用最大模长来放缩坐标值
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    angle = np.arctan2(-dy, -dx) / np.pi  # HxW（-1, 1]

    bin_real = (angle + 1) / 2 * (num_bins - 1)  # HxW (0, num_bins-1]
    bin_left = np.floor(bin_real).astype(int)  # HxW 0,1,...,num_bins-1
    bin_right = (bin_left + 1) % num_bins  # HxW 1,2,...,num_bins % num_bins -> 1, 2, ..., num_bins, 0
    w = (bin_real - bin_left.astype(np.float32))[..., None]  # HxWx1
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]  # 线性插值计算实际的颜色值
    small_ind = rad <= 1  # 以模长为1作为分界线来分开处理，个人理解这里主要是用来控制颜色的饱和度，而前面的处理更像是控制色调。
    # 小于1的部分拉大
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    # 大于1的部分缩小
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img

def write_flo_file(filename, flow):
    height, width, _ = flow.shape

    # .flo文件的头部信息
    header = np.array([80, 73, 69, 72], np.uint8).tobytes()
    size = np.array([width, height], np.int32).tobytes()

    # 将光流数据转换为浮点数，并将其展平
    flow_data = flow.astype(np.float32).flatten()

    # 将头部信息、尺寸信息和光流数据写入文件
    with open(filename, 'wb') as f:
        f.write(header)
        f.write(size)
        f.write(flow_data.tobytes())

def read_flo_file(filename):
    with open(filename, 'rb') as f:
        # 读取头部信息
        header = np.frombuffer(f.read(4), dtype=np.uint8)
        # 检查文件头是否为光流文件
        if header.tobytes() != b'PIEH':
            raise ValueError('Invalid .flo file.')

        # 读取尺寸信息
        width = np.frombuffer(f.read(4), dtype=np.int32)[0]
        height = np.frombuffer(f.read(4), dtype=np.int32)[0]

        # 读取光流数据
        flow_data = np.frombuffer(f.read(), dtype=np.float32)

        # 将展平的光流数据重新形状为(height, width, 2)
        flow = flow_data.reshape(height, width, 2)

    return flow

def visualize_flow(flow):
    # 计算光流向量的大小
    magnitude = np.sqrt(np.sum(flow**2, axis=2))

    # 计算光流向量的角度
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

    # 创建一个空的图像数组
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    # 将角度和大小映射到色调和亮度
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(magnitude * 15, 255)

    # 将HSV图像转换为RGB图像
    rgb = plt.get_cmap('hsv')(hsv[..., 0] / 255.0)
    rgb[..., 2] = hsv[..., 2] / 255.0

def deformed_source1(source, deformed,contribution_maps):
    colormap = plt.get_cmap('gist_rainbow')
    images = []
    full_mask = []
    for i in range(deformed.shape[1]):
        image = deformed[:, i].data.cpu()
        # import ipdb;ipdb.set_trace()
        image = F.interpolate(image, size=source.shape[1:3])
        mask = contribution_maps[:, i:(i + 1)].data.cpu().repeat(1, 3, 1, 1)
        mask = F.interpolate(mask, size=source.shape[1:3])
        image = np.transpose(image.numpy(), (0, 2, 3, 1))
        mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

        if i != 0:
            color = np.array(colormap((i - 1) / (deformed.shape[1] - 1)))[:3]
        else:
            color = np.array((0, 0, 0))

        color = color.reshape((1, 1, 1, 3))

        # images.append(image)
        if i != 0:
            images.append(mask * color)
        else:
            images.append(mask)

        full_mask.append(mask * color)

    image1 = sum(full_mask)
    return image1[0]



if __name__ == "__main__":
    filename = '../flow.flo'
    flow = read_flo_file(filename)
    visualize_flow(flow)