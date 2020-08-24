#coding:utf-8
#!/usr/bin/env python3.7

'''ubuntu中跑,找不到cv2包'''
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')   #解决cv2没法导入的问题

import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.onnx as torch_onnx

from tensorboardX import SummaryWriter

from PIL import Image
import numpy as np

import core.config as cfg

'''
选择网络，需要更改网络中分类类别数目
'''
# from core.networks import Model
# from models_network.vgg import vgg11_bn as Model
from models_network.resnet import resnet18 as Model
# from models_network.resnet import resnet50 as Model  #acc 0.5  运行内存不足
# from models_network.densenet import densenet121 as Model  #训：acc:77.0446  测：0.7600
# from models_network.senet import seresnet18 as Model  #  torch_model_Loss1.165_Acc0.562.onnx  测试集：Loss is:0.1409,Correct is:0.5462
# from models_network.googlenet import googlenet as Model


import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

writer = SummaryWriter('./runs')   #定义日志输出


'''
模型效果评估
'''
def val(net,loss_func):
    #加载数据
    transform = transforms.Compose([
        transforms.Resize((cfg.image_height, cfg.image_width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    val_data = datasets.ImageFolder(root=cfg.val_data_path, transform=transform)
    val_loader = DataLoader(dataset=val_data, batch_size=cfg.val_batch_size, shuffle=True)

    #不做训练操作
    net.eval()

    #定义损失，用于评估
    total_loss = 0.0
    total_correct_num = 0

    # #定义数据迭代器
    # val_iter = iter(val_loader)
    # max_iter = min(max_iter, len(val_loader))
    #
    # #开始测试
    # for i in range(max_iter):
    #     #获取当前批次数据
    #     (data, label) = val_iter.next()
    #
    #     #使用cuda加速
    #     if torch.cuda.is_available:
    #         print('data.cuda(),label.cuda()')
    #         data,label = data.cuda(),label.cuda()
    '''遍历数据'''
    for idx, data in enumerate(val_loader):
        torch.cuda.empty_cache()
        # 读取数据
        x_test, y_test = data
        x_test, y_test = Variable(x_test), Variable(y_test)
        if torch.cuda.is_available:
            x_test, y_test = x_test.cuda(), y_test.cuda()
        #前向预测
        output = net(x_test)
        #计算损失
        loss = loss_func(output,y_test)
        #计算准确率
        correct_num = torch.sum(torch.argmax(output, 1) == y_test.data)
        total_correct_num += correct_num
        #累加损失
        total_loss += float(loss)
        print('loss is:{:.4f},accuracy is:{:.4f}'.format(loss,float(correct_num)/cfg.val_batch_size))
    print('Loss is:{:.4f},Correct is:{:.4f}'.format(total_loss/len(val_loader),float(total_correct_num)/len(val_data)))
    return total_loss/len(val_loader),float(total_correct_num)/len(val_data)

'''
模型训练
'''
def train(type=1):
    #加载训练数据
    transform = transforms.Compose([
        transforms.Resize((cfg.image_height,cfg.image_width)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_data = datasets.ImageFolder(root=cfg.train_data_path, transform=transform)
    data_loader = DataLoader(dataset=train_data, batch_size=cfg.train_batch_size, shuffle=True)

    #定义模型
    model = Model()
    model = model.train()
    if torch.cuda.is_available():
        print('model.cuda()')
        model = model.cuda()

    #使用交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    #定义优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)


    if type == 0 and os.path.exists(cfg.save_torch_model):
        print('进行模型恢复操作。。。')
        model = torch.load(cfg.save_torch_model)
    elif type == 1 and os.path.exists(cfg.save_torch_model_weight):
        print("加载模型参数....")
        model.load_state_dict(torch.load(cfg.save_torch_model_weight))

    #开始训练
    save_loss = 0
    save_acc = 0
    train_onnx_loss = 0
    train_onnx_acc = 0
    test_onnx_loss = 0
    test_onnx_acc = 0

    for epoch in range(cfg.train_max_epoch):
        running_loss = 0
        running_accuracy = 0
        #释放不用的gpu
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        # scheduler.step()
        for idx, data in enumerate(data_loader):
            train_onnx_loss = save_loss
            train_onnx_acc = save_acc
            #读取数据
            x_train, y_train = data
            x_train, y_train = Variable(x_train), Variable(y_train)
            if torch.cuda.is_available:
                x_train, y_train = x_train.cuda(),y_train.cuda()

            #前向计算
            y_pred = model(x_train)
            #计算损失
            loss = criterion(y_pred, y_train)
            #梯度重置
            optimizer.zero_grad()
            #反向计算
            loss.backward()
            #参数更新
            optimizer.step()

            #效果评估
            running_loss += loss.data.data
            running_accuracy += torch.sum(torch.argmax(y_pred, 1) == y_train.data)
            print('{}:{}/{}'.format(epoch, idx, len(data_loader)))
            save_loss = loss.data.data
            save_acc = float(torch.sum(torch.argmax(y_pred, 1) == y_train.data))/cfg.train_batch_size
            print('loss:{},accuracy:{}'.format(save_loss, save_acc))

            #记录日志
            writer.add_scalar('loss', loss, epoch * cfg.train_batch_size + idx)
            writer.add_scalar('running_accuracy', running_accuracy, epoch * cfg.train_batch_size + idx)
            writer.add_scalar('running_loss', running_loss, epoch * cfg.train_batch_size + idx)

            #模型保存
            if (epoch*cfg.train_batch_size+idx)//20==0:
                torch.save(model,cfg.save_torch_model)
                torch.save(model.state_dict(),cfg.save_torch_model_weight)

        #当前批次的模型效果
        save_acc = 100 * float(running_accuracy) / len(train_data)
        print('Loss is:{:.4f},accuracy is:{:.4f}'.format(running_loss / len(train_data),save_acc))
        # torch.save(model, './model/model_loss{}_acc{}.pkl'.format(running_loss / len(train_data),
        #                                                           100 * float(running_accuracy) / len(train_data)))
        # torch.save(model.state_dict(), './model/model_loss{}_acc{}.pkl'.format(running_loss / len(train_data),
        #                                                                        100 * float(running_accuracy) / len(train_data)))
        torch.save(model,cfg.save_torch_model)
        print('模型保存成功')

        #测试集上效果评估
        test_onnx_loss,test_onnx_acc = val(model,criterion)
    # writer.export_scalars_to_json("C:/Users/pc/Desktop/tensorboardX/all_scalars.json")
    writer.close()

    #模型转换为onnx
    input_shape = (cfg.image_channels, cfg.image_height, cfg.image_width)
    model_onnx_path = "torch_model_train_Acc{:.3f}_test_Acc{:.3f}.onnx".format(train_onnx_acc,test_onnx_acc)
    model.train(False)

    # print(list(model.parameters))
    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(1, *input_shape))
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    output = torch_onnx.export(model,
                               dummy_input,
                               model_onnx_path,
                               verbose=False)
    print('onnx转出成功')


'''图片样本预测'''
def predict(input_image_path = './data/img_same_size/0'):
    model = torch.load(r'./model/01.pth')

    image_path = input_image_path
    images = os.listdir(image_path)
    X = []
    #开始测试
    for image_name in images:
        # b31. 构建图像的路径
        image_full_path = os.path.join(image_path, image_name)
        # b32. 加载图像
        img = Image.open(image_full_path)
        # b33. 将RGB的图像转换为灰度图像
        # img = img.convert("L")
        # b34. 由于原始图像的大小是不固定的，导致构建出来的特征向量也是大小不固定的，所以将图像转换为大小一致的情况
        img = img.resize((cfg.image_height, cfg.image_width))
        # b35. 将图像转换为numpy数组的形式
        # img_arr = np.array(img).reshape(image_height, image_width, 1)
        img_arr = np.array(img).reshape(cfg.image_height, cfg.image_width, 3)
        img_arr = img_arr.transpose(2,0,1)
        # 防止图像的数据像素值太大，导致模型过拟合，将像素值从0~255缩减到0~1之间（这个计算规则就是MinMaxScaler）
        img_arr = img_arr / 255.0
        # b36. 将图像的特征属性数据添加到X和Y中
        X.append(img_arr)
    # c. 将X和Y转换为numpy数组的形式
    X = np.asarray(X).astype(np.float64)
    X = Variable(torch.Tensor((X)))
    #使用cuda加速
    if torch.cuda.is_available:
        print('data.cuda()')
        data = X.cuda()

    #前向预测
    batch_size = 8
    total_num = len(data)
    total_batch = total_num // batch_size
    for batch in range(total_batch):
        start_idx = batch_size * batch
        end_idx = start_idx + batch_size
        output = model(data[start_idx:end_idx])
        #计算损失
        pred = torch.argmax(output, 1)
        print(pred)

if __name__ == '__main__':
    '''外部输入学习率，方便训练运行'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=cfg.learning_rate, help='initial learning rate')
    args = parser.parse_args()

    #模型训练
    train()
    #样本预测
    # predict('./data/test/0')
