# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch
import torch.nn as nn
import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
import matplotlib.pyplot as plt
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    epoch_loss = 0.0
    loss_fn=nn.MSELoss()
    # iterate all training samples
    for samples,heatmaps in data_loader:
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        #-----------经过修改，outputs中仅有heatmaps-------------#
        outputs = model(samples)
        # 计算损失
        heatmaps = heatmaps.to('cuda')
        loss = loss_fn(outputs['heatmaps'], heatmaps)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        # -----------------以下可视化预测-----------------#
        tensor_heatmap = outputs['heatmaps'][0]
        # 移除 Batch 和通道维度
        heatmap = tensor_heatmap.squeeze(0).squeeze(0)  # 结果形状为 (640, 256)

        # 转换为 NumPy 数组
        heatmap_np = heatmap.detach().cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Intensity")
        plt.title("Heatmap Visualization")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()
        #-----------------以下可视化GT-----------------#
        # tensor_heatmap = heatmaps[0]
        # # 移除 Batch 和通道维度
        # heatmap = tensor_heatmap.squeeze(0).squeeze(0)  # 结果形状为 (640, 256)
        #
        # # 转换为 NumPy 数组
        # heatmap_np = heatmap.detach().cpu().numpy()
        #
        # # 可视化
        # plt.figure(figsize=(10, 5))
        # plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        # plt.colorbar(label="Intensity")
        # plt.title("Heatmap Visualization")
        # plt.xlabel("Width")
        # plt.ylabel("Height")
        # plt.show()
    # ----------输出损失----------------#
    print(f"Epoch [{epoch + 1}], Loss: {loss / len(data_loader):.4f}")
    return {"loss": {epoch_loss}, "loss_ce": 0}




    #     # calc the losses
    #     loss_dict = criterion(outputs, targets)
    #     weight_dict = criterion.weight_dict
    #     losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    #
    #     # reduce all losses
    #     loss_dict_reduced = utils.reduce_dict(loss_dict)
    #     loss_dict_reduced_unscaled = {f'{k}_unscaled': v
    #                                   for k, v in loss_dict_reduced.items()}
    #     loss_dict_reduced_scaled = {k: v * weight_dict[k]
    #                                 for k, v in loss_dict_reduced.items() if k in weight_dict}
    #     losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    #
    #     loss_value = losses_reduced_scaled.item()
    #
    #     if not math.isfinite(loss_value):
    #         print("Loss is {}, stopping training".format(loss_value))
    #         print(loss_dict_reduced)
    #         sys.exit(1)
    #     # backward
    #     optimizer.zero_grad()
    #     losses.backward()
    #     if max_norm > 0:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    #     optimizer.step()
    #     # update logger
    #     metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
    #     metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for samples,heatmaps in data_loader:
        samples = samples.to(device)

        outputs = model(samples)
        tensor_heatmap = outputs['heatmaps']
        # 移除 Batch 和通道维度
        heatmap = tensor_heatmap.squeeze(0).squeeze(0)  # 结果形状为 (640, 256)

        # 转换为 NumPy 数组
        heatmap_np = heatmap.detach().cpu().numpy()

        # 可视化热力图
        plt.figure(figsize=(10, 5))
        plt.imshow(heatmap_np, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Intensity")
        plt.title("Heatmap Visualization")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()
    #---------mae和mse目前都是空，只是为了能让模型顺利训练起来-------------#
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse
    #     outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    #
    #     outputs_points = outputs['pred_points'][0]
    #
    #     gt_cnt = targets[0]['point'].shape[0]
    #     # 0.5 is used by default
    #     threshold = 0.5
    #
    #     points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    #     predict_cnt = int((outputs_scores > threshold).sum())
    #     # if specified, save the visualized images
    #     if vis_dir is not None:
    #         vis(samples, targets, [points], vis_dir)
    #     # accumulate MAE, MSE
    #     mae = abs(predict_cnt - gt_cnt)
    #     mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
    #     maes.append(float(mae))
    #     mses.append(float(mse))
    # # calc MAE, MSE
