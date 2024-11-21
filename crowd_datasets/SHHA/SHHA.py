import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torch.nn.functional as F
import glob
import scipy.io as io
import matplotlib.pyplot as plt

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "train.list"
        self.eval_list = "test.list"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        # random crop augumentaiton
        #-----热力图初始化-----------#
        heatmaps=None
        if self.train and self.patch:
            # img, point = random_crop(img, point)
            #-------------如果是训练模式，则对输入图像resize，对应的点坐标也做变换------------#
            img, point =resize_image_and_points(img,point,(128,512))#(width:128,height:512)
            img_size = img.shape[1:]
            #-----------根据GT的点坐标生成热力图，这里的（height=256，width=64），取这个大小是因为上面的（128，512）大小的图/
            # 经过backbone后生成的特征图大小即为（64，256）保持一致-------------------#
            heatmaps = generate_heatmap_from_points(point, img_size,(256,64))

            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        else:
            #----------如果是验证模式，则只进行图片的resize----------------#
            img, point = resize_image_and_points(img, point, (128, 512))  # (width:128,height:512)
            # 获取img宽高
            # img_size = img.shape[1:]
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        # target = [{} for i in range(len(point))]
        # for i, _ in enumerate(point):
        #     target[i]['point'] = torch.Tensor(point[i])
        #     image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
        #     image_id = torch.Tensor([image_id]).long()
        #     target[i]['image_id'] = image_id
        #     target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
        # 将img可视化一下

        # return heatmaps,img, target
        #--------返回值只需要热力图和resize后的原图------------#
        return heatmaps, img


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    #为之后使用opencv数据增强不转换为Image类型
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])

    return img, np.array(points)
def resize_image_and_points(image, points, target_size):
    """
    对图片进行 resize 并调整坐标
    :param image: 输入图片 (H, W, C) 或 (H, W)，可以是 NumPy 数组
    :param points: 坐标点列表，形状为 [(x1, y1), (x2, y2), ...]
    :param target_size: 目标大小 (new_width, new_height)
    :return: resize 后的图片和新的坐标点列表
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    # 获取原始图片大小
    original_height, original_width = image_np.shape[:2]
    new_width, new_height = target_size

    # 计算缩放因子
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # 使用 OpenCV 进行 resize
    resized_image_np = cv2.resize(image_np, (new_width, new_height))

    # 调整坐标点
    resized_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]

    # 转换回 Tensor 格式并调整为 (C, new_height, new_width)
    resized_image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1)  # (C, H, W)

    return resized_image_tensor, resized_points
def generate_heatmap_from_points(points, img_size,heatmap_size,sigma=2):
    """
    根据坐标点生成热力图，并调整到指定大小。
    :param points: (N, 2) 的 numpy 数组，表示关键点的 (x, y) 坐标
    :param img_size: 原始图像大小 (H, W)，例如 (128, 128)
    :param heatmap_size: 目标热力图大小 (H', W')，例如 (64, 64)
    :param sigma: 控制高斯分布的范围
    :return: 生成的热力图，形状为 (H', W')
    """
    H, W = img_size
    heatmap = np.zeros((H, W), dtype=np.float32)  # 初始化原始大小的热力图

    # 遍历每个点，生成高斯分布
    for x, y in points:
        for i in range(H):
            for j in range(W):
                heatmap[i, j] += np.exp(-((i - y) ** 2 + (j - x) ** 2) / (2 * sigma ** 2))

    # 转换为 PyTorch Tensor 并调整大小
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # 添加 batch 和通道维度
    heatmap_resized = F.interpolate(heatmap, size=heatmap_size, mode='bilinear', align_corners=False)
    return heatmap_resized
# random crop augumentation
def random_crop(img, den, num_patch=8):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.shape[1] - half_h)
        start_w = random.randint(0, img.shape[2] - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # ------------可视化一下img---------------
        # 将张量从 (C, H, W) 转换为 (H, W, C)
        # img_np = img[:, start_h:end_h, start_w:end_w].permute(1, 2, 0).numpy()
        #
        # # 如果张量是浮点类型且值范围不是 [0, 1]，需要归一化
        # img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        #
        # # 使用 matplotlib 显示图像
        # plt.imshow(img_np)
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
        # # ---------------copy the cropped points---------------
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den