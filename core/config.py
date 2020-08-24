#个人设计缺陷分类参数
'''
train_batch_size = 32  #训练的批次大小
val_batch_size = 16    #测试得批次大小
learning_rate = 0.001   #学习率
train_max_epoch = 10          #运行的批次
val_data_path = './data/test'  #测试数据路径
train_data_path = './data/train' #训练数据路径
# val_data_path = './data/img_same_size'  #测试数据路径
# train_data_path = './data/img_same_size' #训练数据路径
save_onnx_model = 'torch_model.onnx' #onnx模型
save_torch_model = './model/01.pth' #pth模型
save_torch_model_weight =  './model/01_weight.pth'


image_height = 224  #输入图片的高度
image_width= 224 #输入图片的宽度
image_channels = 3

import os
classes = len(os.listdir('./data/train'))  #要分类的类别
'''


'''样本为传统方法设计'''
train_batch_size = 32  #训练的批次大小
val_batch_size = 32    #测试得批次大小
learning_rate = 0.00001   #学习率
train_max_epoch = 10        #运行的批次
val_data_path = './data_ct/test'  #测试数据路径
train_data_path = './data_ct/train' #训练数据路径
save_onnx_model = 'torch_model_ct.onnx' #onnx模型
save_torch_model = './model/01_ct.pth' #pth模型
save_torch_model_weight =  './model/01_weight_ct.pth'


image_height = 112  #输入图片的高度
image_width= 112 #输入图片的宽度
image_channels = 3

import os
classes = len(os.listdir('./data_ct/train'))  #要分类的类别