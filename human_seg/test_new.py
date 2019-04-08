import sys

sys.path.append('./GaitSet')
import os
import torch.autograd as autograd

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from gait.SetNet import SetNet
import torch.nn as nn
import torch
import cv2
import torch.optim as optim

import torch.nn.functional as F


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


hidden_dim = 256
model_name = 'GaitSet'
lr = 1e-4

encoder = SetNet(hidden_dim).float()
encoder = nn.DataParallel(encoder).cuda()
# print(encoder)
optimizer = optim.Adam([
    {'params': encoder.parameters()},
], lr=lr)

encoder.load_state_dict(
    torch.load(
        '/data/zhangwd/result/GaitSet/work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000'
        '-encoder.ptm'))

# print(encoder)
optimizer.load_state_dict(
    torch.load(
        '/data/zhangwd/result/GaitSet/work/checkpoint/GaitSet/GaitSet_CASIA-B_73_False_256_0.2_128_full_30-80000'
        '-optimizer.ptm'))

# model = model.cuda()
encoder.eval()

img = cv2.imread('002-cl-01-036-003.png')
img = autograd.Variable(torch.from_numpy(img)).cuda()

h = img.shape[0]
w = img.shape[1]
c = img.shape[2]
# n, s, c, h, w = x.size()
# torch.Size([64, 64, 3])
# torch.Size([64, 64, 1, 3])
img = img.view(-1, c, h, w)
img = img.type(torch.cuda.FloatTensor)
print(img.shape)

# =======================================================================
# =======================================================================

img2 = cv2.imread('002-cl-01-036-011.png')
img2 = autograd.Variable(torch.from_numpy(img2)).cuda()

h2 = img2.shape[0]
w2 = img2.shape[1]
c2 = img2.shape[2]

img2 = img2.view(-1, c, h, w)
img2 = img2.type(torch.cuda.FloatTensor)

# ===============================================
# =============================================
feature, _ = encoder(img)
n, num_bin, _ = feature.size()
feature = feature.view(n, -1).data.cpu().numpy()

feature2, _ = encoder(img2)
n2, num_bin2, _ = feature2.size()
feature2 = feature2.view(n2, -1).data.cpu().numpy()


dist = cuda_dist(feature, feature2)
print(dist)
# print(feature)

# n, s, c, h, w = x.size()
