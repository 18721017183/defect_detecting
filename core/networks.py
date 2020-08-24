#coding:utf-8
#!/usr/bin/env python3.7

import torch
import cv2
import numpy as np

# #
# class Model(torch.nn.Module):
#     '''四层卷积'''
#     def __init__(self):
#         super(Model,self).__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=3,
#                             out_channels=32,
#                             kernel_size=3,
#                             stride=1,
#                             padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2,
#                                stride=2,
#                                padding=0),
#             torch.nn.Conv2d(32,64,3,1,padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2,2,0),
#             torch.nn.Conv2d(64,128,3,1,padding=1),
#             torch.nn.ReLU(),
#             # torch.nn.MaxPool2d(2,2,0)
#             torch.nn.Conv2d(128, 256, 3, 1, padding=1),
#             torch.nn.ReLU(),
#         )
#         self.dense = torch.nn.Sequential(
#             torch.nn.Linear(56*56*256,512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512,4)
#         )
#
#     def forward(self,x):
#         x = self.conv(x)
#         x = x.view(-1,56*56*256)
#         x = self.dense(x)
#         # return x
#         return x


'''
VGG
'''
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            torch.nn.Conv2d(64, 128, 3, 1, padding=1),
            torch.nn.Conv2d(128, 128, 3, 1, padding=1),
            torch.nn.MaxPool2d(2,2,0),
            torch.nn.Conv2d(128, 256, 3, 1, padding=1),
            torch.nn.Conv2d(256, 256, 3, 2, padding=1),
            torch.nn.Conv2d(256, 256, 3, 2, padding=1),
            torch.nn.MaxPool2d(2, 2, 0),
            # torch.nn.Conv2d(256, 512, 3, 1, padding=1),
            # torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            # torch.nn.Conv2d(512, 512, 3, 2, padding=1),
            # torch.nn.MaxPool2d(2, 2, 0),
            # torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            # torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            # torch.nn.Conv2d(512, 512, 3, 1, padding=1),
            # torch.nn.MaxPool2d(2, 2, 0)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(7*7*256,1024),
            torch.nn.ReLU(),
            # torch.nn.Linear(2048,1024),
            # torch.nn.ReLU(),
            torch.nn.Linear(1024, 4)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,7*7*512)
        x = self.dense(x)
        return x





def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
