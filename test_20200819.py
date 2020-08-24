import numpy as np
from time import time
import argparse
import onnxruntime as rt
import core.config as cfg
import cv2
np.set_printoptions(threshold=np.inf)
import os
from PIL import Image


# (1, 3, 112, 112) <class 'numpy.ndarray'>
# pred_onx: [[ 0.76079816 14.146033  ]]
# 预测值: 1

# pred_onx: [[ 4.7841578 -4.5109444]]
# 预测值: 0

sess = rt.InferenceSession("torch_model_train_Acc98.036_test_Acc0.959_ct.onnx")
image_path = r'C:\Users\pc\Desktop\img\img\image_total\expand\5\1.jpg'

image = cv2.imread(image_path)
img = cv2.resize(image ,(cfg.image_height, cfg.image_width))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_arr = img.transpose(2, 0, 1)

# image = Image.open(image_path)
# img = image.resize((cfg.image_height, cfg.image_width))

# img = image.resize((cfg.image_height, cfg.image_width))
# b35. 将图像转换为numpy数组的形式
# 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
img_arr = img_arr / 255.0
img_arr = np.expand_dims(img_arr ,0).astype(np.float32)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(img_arr.shape ,type(img_arr))
pred_onx = sess.run([label_name], {input_name: img_arr})[0]
print('pred_onx:' ,pred_onx)
Max = np.argmax(pred_onx, 1)
print('预测值:' ,Max[0])
cv2.imwrite("./data_ct/1.jpg" ,image)
cv2.imshow("pre:" ,image)
cv2.waitKey(0)

# sess = rt.InferenceSession("torch_model_train_Acc98.036_test_Acc0.959_ct.onnx")
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# image_path = r'C:\Users\pc\Desktop\img\img\image_total\expand\5\1.jpg'
# img = Image.open(image_path)
# # img = np.asarray(img)
# # print(img.shape)
# # b33. 将RGB的图像转换为灰度图像
# # img = img.convert("L")
# # b34. 由于原始图像的大小是不固定的，导致构建出来的特征向量也是大小不固定的，所以将图像转换为大小一致的情况
# img = img.resize((cfg.image_height, cfg.image_width))
# # b35. 将图像转换为numpy数组的形式
# # img_arr = np.array(img).reshape(image_height, image_width, 1)
# img_arr = np.array(img).reshape(cfg.image_height, cfg.image_width, 3)
# img_arr = img_arr.transpose(2, 0, 1)
# # 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
# img_arr = img_arr / 255.0
# img_arr = img_arr.astype(np.float32)
# pred_onx = sess.run([label_name], {input_name: np.expand_dims(img_arr, 0)})[0]  # 一次预测1张图片  时间：0.014044046401977539
# # pred_onx = sess.run([label_name], {input_name: X[0:6]})[0]   #一次预测6张图片   时间：0.07222199440002441
# Max = np.argmax(pred_onx, 1)
# # print(pred_onx)
# # print('Probability:{},label is :{}'.format(pred_onx,Max))
# print(Max[0])