#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
import onnxruntime as rt
from PIL import Image
import torch
import core.config as cfg
from torch.autograd import Variable
import cv2
from pylab import *
np.set_printoptions(threshold=np.inf)
import os

# sess = rt.InferenceSession("torch_model_train_Acc0.9385_test_Acc0.928.onnx")
sess = rt.InferenceSession("torch_model_train_Acc98.036_test_Acc0.959_ct.onnx")

# image_path = r'C:\Users\pc\Desktop\img\img\image_total\expand\test\0'
image_path = r'C:\Users\pc\Desktop\img\img\image_total\expand\5'
# image_path = './data/test/3'
# image_path = './data/a'
images = os.listdir(image_path)
X = []
X_show = []
for image_name in images:
    if image_name == "desktop.ini":
        continue
    # b31. 构建图像的路径
    image_full_path = os.path.join(image_path, image_name)
    # b32. 加载图像
    print(image_full_path)
    img = Image.open(image_full_path)
    X_show.append(image_full_path)   #用于显示
    # img = np.asarray(img)
    # print(img.shape)
    # b33. 将RGB的图像转换为灰度图像
    # img = img.convert("L")
    # b34. 由于原始图像的大小是不固定的，导致构建出来的特征向量也是大小不固定的，所以将图像转换为大小一致的情况
    img = img.resize((cfg.image_height, cfg.image_width))
    # b35. 将图像转换为numpy数组的形式
    # img_arr = np.array(img).reshape(image_height, image_width, 1)
    img_arr = np.array(img).reshape(cfg.image_height, cfg.image_width, 3)
    img_arr = img_arr.transpose(2, 0, 1)
    # 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
    img_arr = img_arr / 255.0
    # b36. 将图像的特征属性数据添加到X和Y中
    X.append(img_arr)
# c. 将X和Y转换为numpy数组的形式
X = np.asarray(X).astype(np.float32)
# X = Variable(torch.Tensor((X)))
# 使用cuda加速
# if torch.cuda.is_available:
#     print('data.cuda()')
#     data = X.cuda()

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(input_name)
print(len(images))

k = 1
err_img_path = []
result = []
done = []
correct = os.path.split(image_path)[-1]
print("correct:",correct)
for i in range(len(images)):
    start_time = time.time()
    pred_onx = sess.run([label_name], {input_name: np.expand_dims(X[i],0)})[0]  #一次预测1张图片  时间：0.014044046401977539
    # pred_onx = sess.run([label_name], {input_name: X[0:6]})[0]   #一次预测6张图片   时间：0.07222199440002441
    print("pred_onnx",pred_onx)
    Max = np.argmax(pred_onx, 1)
    # print(pred_onx)
    # print('Probability:{},label is :{}'.format(pred_onx,Max))
    print("预测值：",Max[0])
    if (Max[0] != int(correct)):
        k += 1
        err_img_path.append(X_show[i])
        img_show = cv2.imread(X_show[i])
        print(X_show[i])
        cv2.imshow("{}".format(i),img_show)
        cv2.waitKey(0)
    # result.append(Max[0])
    # if k == 6:
    #     k = 0
    #     print(result)
    #     if np.max(result) > 0:
    #         done.append(i)
    #     result = []
    end_time = time.time()

    # print(end_time-start_time)   #计算运行时间
for name in err_img_path:
    print(name)
print("{}/{}".format(k,len(images)))