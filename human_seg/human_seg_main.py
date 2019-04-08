# -*- coding: utf-8 -*-

import time
from human_seg.vgg import VGGNet
from human_seg.my_fcn import FCNs
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torch
import cv2
from PIL import Image
import visdom
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Human_seg(object):
    def __init__(self, weight_path, input_img, img_size, use_cuda):

        self.weight_path = weight_path
        self.input_img = input_img
        self.use_cuda = use_cuda
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        checkpoint = torch.load(self.weight_path)
        self.vgg_model = VGGNet(pretrained=False, requires_grad=False)
        self.fcn_model = FCNs(pretrained_net=self.vgg_model, n_class=2)
        # fcn_model = ResNetDUCHDC(pretrained=False,num_classes=2)
        self.fcn_model.load_state_dict(checkpoint['model'])
        self.fcn_model.to(self.device)

    def run(self):
        print('Start...')
        input_img = cv2.resize(self.input_img, (self.img_size, self.img_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        # input_img = input_img[np.newaxis, :, :, :]
        input_img = Image.fromarray(input_img)
        input_img = self.transform(input_img)
        input_img = input_img.unsqueeze(0)
        input_img = torch.autograd.Variable(input_img)
        if self.use_cuda:
            input_img = input_img.cuda()
        else:
            input_img = input_img
        start = time.time()
        output = self.fcn_model(input_img)
        output = nn.functional.sigmoid(output)
        end = ((time.time() - start) * 1000)
        output_np = output.cpu().data.numpy().copy()
        output_np = np.argmin(output_np, axis=1)
        output_np = output_np
        print('time:', end)

        return output_np


if __name__ == '__main__':
    # weight_path = '/data/shengty1/gait/FCN8s-weights/duc-20k/fcn_model_30.pt'
    vis = visdom.Visdom(port=8097)
    weight_path = '/data/shengty1/gait/FCN8s-weights/44k-norm/fcn_model_1111.pt'
    input_img_path = '/data/shengty1/gait/identity/0ce2875e-0e06-4dbf-98b9-f5ed1c372967/128ce23e/014.jpg'
    img_size = 160
    use_cuda = True
    input_img = cv2.imread(input_img_path)
    tag = Human_seg(weight_path, input_img, img_size, use_cuda)
    for index in range(50):
        img_output = tag.run()
        if np.mod(index, 5) == 1:
            vis.close()
            vis.images(img_output[:, None, :, :], opts=dict(title='pred'))
