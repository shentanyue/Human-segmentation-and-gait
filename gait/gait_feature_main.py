# -*- coding: utf-8 -*-
# @Author  : shenty
# @Time    : 2018/4/8
import sys
sys.path.append('./GaitSet')
from SetNet import SetNet
from pretreatment import cut_img
import xarray as xr
import math
import os
import torch.autograd as autograd
import torch.nn as nn
import torch
import cv2
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# np.set_printoptions(threshold=np.inf)


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


class Gait_feature(object):
    def __init__(self):
        self.hidden_dim = 256
        self.model_name = 'GaitSet'
        self.lr = 1e-4

        self.encoder = SetNet(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder).cuda()
        # print(encoder)
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.encoder.load_state_dict(
            torch.load(
                '/data/zhangwd/result/GaitSet/work/checkpoint/GaitSet'
                '/GaitSet_CASIA-B_73_False_256_0.2_128_full_30'
                '-80000'
                '-encoder.ptm'))

        self.optimizer.load_state_dict(
            torch.load(
                '/data/zhangwd/result/GaitSet/work/checkpoint/GaitSet'
                '/GaitSet_CASIA-B_73_False_256_0.2_128_full_30'
                '-80000'
                '-optimizer.ptm'))
        self.encoder.eval()

    def img2xarray(self, img_list, resolution):
        frame_list = []
        for img in img_list:
            img = cut_img(img[:, :, 0])
            img = np.reshape(img, [resolution, resolution, -1])[:, :, 0]
            frame_list.append(img)
            # img = np.expand_dims(img, axis=2)
            # img = np.repeat(img, 3, axis=2)

        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def load_data(self, resolution, cut_padding, img_list):
        data = [self.img2xarray(img_list, resolution)[:, :, cut_padding:-cut_padding].astype('float32') / 255.0]
        frame_sets = [set(feature.coords['frame'].values.tolist()) for feature in data]
        frame_sets = list(set.intersection(*frame_sets))

        def select_frame(index):
            sample = data[index]
            _ = [feature.values for feature in sample]
            return _

        data = list(map(select_frame, range(len(data))))
        # gpu_num = min(torch.cuda.device_count(), batch_size)

        data = [data]
        frame_sets = [frame_sets]

        batch_size = 1
        gpu_num = 1
        feature_num = 1

        # print('gpu_num:', type(gpu_num), gpu_num)
        # print('batch_size:', type(batch_size), batch_size)
        # print('feature_num:', type(feature_num), feature_num)
        batch_per_gpu = math.ceil(batch_size / gpu_num)

        batch_frames = [[
            len(frame_sets[i])
            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
            if i < batch_size
        ] for _ in range(gpu_num)]

        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
        seqs = [[
            np.concatenate([
                data[i][j]
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]
        seqs = [np.asarray([
            np.pad(seqs[j][_],
                   ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                   'constant',
                   constant_values=0)
            for _ in range(gpu_num)])
            for j in range(feature_num)]
        batch_frames = np.asarray(batch_frames)

        return seqs, batch_frames

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def run(self, img_list):
        resolution = 64
        resolution = int(resolution)
        cut_padding = int(float(resolution) / 64 * 10)

        data, batch_frames = self.load_data(resolution, cut_padding, img_list)

        for j in range(len(data)):
            data[j] = self.np2var(data[j]).float()
        if batch_frames is not None:
            batch_frame = self.np2var(batch_frames).int()

        feature, _ = self.encoder(*data, batch_frame)
        n, num_bin, _ = feature.size()
        feature = feature.view(n, -1).data.cpu().numpy()

        return feature


if __name__ == '__main__':

    gait = Gait_feature()
    frame_list = []
    dirs = os.listdir('090')
    dirs.sort()
    for dir in dirs:
        img = cv2.imread('090/'+dir)
        frame_list.append(img)
    feature = gait.run(frame_list)
    print(feature)
    # dist = cuda_dist(feature, feature2)
    # print(dist)
