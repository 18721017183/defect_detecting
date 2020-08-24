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

sess = rt.InferenceSession("torch_model_train_Acc0.9385_test_Acc0.928.onnx")

# image_path = './data/test/3'
image_path = './data/a'
images = os.listdir(image_path)
X = []
for image_name in images:
    # b31. 构建图像的路径
    image_full_path = os.path.join(image_path, image_name)
    # b32. 加载图像
    img = Image.open(image_full_path)
    img = np.asarray(img)
    max = np.max(img)
    # print('max',max)
    with open('bb.txt','w') as f:
        for i in img:
            for j in i:
                for k in j:
                    f.write(str(k)+' ')
    img = cv2.imread(image_full_path)
    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # b33. 将RGB的图像转换为灰度图像
    # img = img.convert("L")
    # b34. 由于原始图像的大小是不固定的，导致构建出来的特征向量也是大小不固定的，所以将图像转换为大小一致的情况
    # img = img.resize((cfg.image_height, cfg.image_width))
    # b35. 将图像转换为numpy数组的形式
    # img_arr = np.array(img).reshape(image_height, image_width, 1)
    # img_arr = np.array(img).reshape(cfg.image_height, cfg.image_width, 3)
    img_arr = np.array(img)
    print('img_arr..',np.max(img_arr))
    img_arr = img_arr.transpose(2, 0, 1)
    # 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
    img_arr = img_arr / 255.0
    # b36. 将图像的特征属性数据添加到X和Y中
    X.append(img_arr)
    max1 = np.max(X)
    print('max1',max1)
    print(np.shape(X))
    # print('-'*100)
    # print(X)
    # print('-'*100)
    n = 0
    max__ = 0
    with open('aa.txt','w') as f:
        for i in X:
            for j in i:
                for k in j:
                    for m in k:
                        f.write(str(m))
                        n += 1
                        if max__ < m:
                            max__ = m
    print('n',n)
    print('max__',max__)
    break
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
print(label_name)


k = 1
result = []
done = []
for i in range(50):
    start_time = time.time()
    pred_onx = sess.run([label_name], {input_name: np.expand_dims(X[i],0)})[0]  #一次预测1张图片  时间：0.014044046401977539
    # pred_onx = sess.run([label_name], {input_name: X[0:6]})[0]   #一次预测6张图片   时间：0.07222199440002441
    Max = np.argmax(pred_onx, 1)
    # print(pred_onx)
    # print('Probability:{},label is :{}'.format(pred_onx,Max))
    result.append(Max[0])
    if k == 6:
        k = 0
        print(result)
        if np.max(result) > 0:
            done.append(i)
        result = []
    k += 1
    end_time = time.time()
    # print(end_time-start_time)
print(done)