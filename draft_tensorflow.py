"""日常的草稿"""


import numpy as np
import csv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from tensorflow.keras import Sequential, layers, losses, optimizers, models
import os

# 1. 调试early stopping
# a = [0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.77, 0.77, 0.77, 0.77, 0.77, 0.999]
# wait = np.zeros(len(a))
# temp = 0
# i = 0
#
# for acc in a:
#     if acc >= 0.999:
#         print("The realistic training epochs is: %d" % (i + 1))  # 设置early stopping，打印实际训练次数
#         break
#     if acc - temp <= 0.001:
#         wait[i] = wait[i] + 1
#         if (i >= 4) & (np.sum(wait[(i - 4):i]) >= 3):
#             print("The realistic training epochs is: %d" % (i + 1))
#             break
#
#     i += 1
#     temp = acc
# 记录SNR=-10dB时，每一次训练的精度，写入csv
# accuracy2 = {'model_1': [0, 0.214000, 0.214500, 0.215000, 0.484000, 0.619500, 0.367000, 0.741500, 0.830000, 0.620000, 0.803000,
#                          0.807500, 0.813500, 0.881500, 0.866000, 0.889000, 0.871000, 0.876500, 0.887500, 0.870500, 0.886500,
#                          0.888500, 0.865500, 0.866000, 0.887000, 0.879500, 0.869500],
#              'model_2': [0, 0.214500, 0.256500, 0.409500, 0.427000, 0.613500, 0.615500, 0.449500, 0.548000, 0.421000, 0.643500, 0.595500],
#              'model_3': [0, 0.321500, 0.214500, 0.219500, 0.346000, 0.407000, 0.610500, 0.615500, 0.712500, 0.690000, 0.712000,
#                         0.737500, 0.650000, 0.718500, 0.734500, 0.640000, 0.672500, 0.702000, 0.746000, 0.719500],
#              'model_4': [0, 0.214500, 0.214500, 0.214500, 0.218000, 0.596500, 0.641500, 0.650000, 0.636000, 0.617500, 0.626500,
#                         0.652500, 0.682500, 0.741500, 0.797000, 0.823000, 0.831500, 0.816500, 0.824500, 0.796000, 0.837500,
#                         0.797000, 0.791000, 0.812000],
#              'model_5': [0, 0.383500, 0.464500, 0.395000, 0.492500, 0.614000, 0.621500, 0.619000, 0.635500, 0.629500, 0.615000,
#                         0.631500, 0.632500, 0.639500, 0.627500, 0.652000, 0.634500, 0.651000, 0.613000, 0.618500, 0.625500]}
#
# with open('top5_model_accuracy.csv', mode='a+', newline='') as f:
#     data = []
#     a = []
#     for index in accuracy2:
#         data = accuracy2[index]
#         a.append([data, [index]])
#
#     writer = csv.writer(f)
#     writer.writerow(a)

# 2. 从csv文件中画图
lines1 = []
files = ['optimized_model1.csv']
for file in files:
    with open(file, 'r') as f:
        for line in f:
            temp = line.split(',')
            temp[-1] = temp[-1][:-1]  # remove \n
            temp[0:50] = [float(i) for i in temp[0:50]]  # convert accuracy to float
            lines1.append(temp)

lines2 = []
files = ['optimized_model2.csv']
for file in files:
    with open(file, 'r') as f:
        for line in f:
            temp = line.split(',')
            temp[-1] = temp[-1][:-1]  # remove \n
            temp[0:50] = [float(i) for i in temp[0:50]]  # convert accuracy to float
            lines2.append(temp)

high_acc1 = []
for i in range(8):
    temp = max(lines1[i][0:50])
    high_acc1.append(temp)

high_acc2 = []
for i in range(8):
    temp = max(lines2[i][0:50])
    high_acc2.append(temp)

high_acc1.insert(0, np.nan)
high_acc2.insert(0, np.nan)

plt.plot(high_acc1, '--k', label='standard NAS')
plt.plot(high_acc2, 'r', label='balanced NAS')
plt.legend()
plt.ylim((0, 1))
plt.xlabel('B')
plt.ylabel('Accuracy')
# my_x_ticks = np.arange(1, 9, 1)
# plt.xticks(my_x_ticks)
plt.show()
#
# plt.plot(lines[0][0:50], label='model_1')
# plt.plot(lines[1][0:50], label='model_2')
# plt.plot(lines[2][0:50], label='model_3')
# plt.plot(lines[3][0:50], label='model_4')
# plt.plot(lines[4][0:50], label='model_5')
# plt.legend()
# plt.show()


# 3. 计算模型参数量
# 先用model summary
# from model import ModelGenerator


# action = [0, '1x7-7x1 conv', 0, '1x7-7x1 conv', 0, '3x3 maxpool', 0, '3x3 avgpool', 0, '1x7-7x1 conv', 0, '7x7 dconv']
#
#
# def counts(actions, input_channel, filter_number, classes, cell):
#     """ 在最后还有一个全局池化
#         池化层默认padding为same，所以输出尺寸为input_size/2
#         卷积层后面要加偏置，含有BN，所以要加3*output_feature_map
#         前B层的filter个数为32，后B层为64 """
#
#     # 先从actions中取出卷积核及池化核，取出奇数位
#     operator = actions[1::2]
#     conv_size = []
#     pool_size = []
#     numbers = 0
#
#     for index in range(len(operator)):
#         name = operator[index].split(' ', 1)
#         if name[1] == 'conv':
#             temp = [i.split('x', 1) for i in name[0].split('-', 1)]  # 分离出数值
#             content1 = [[float(temp[0][0]), float(temp[0][1])], [float(temp[1][0]), float(temp[1][1])]]  # 将str转换为float,[[1,7],[7,1]]
#             conv_size.append(content1)
#         elif name[1] == 'dconv':
#             temp = name[0].split('x', 1)
#             content1 = [float(temp[0]), float(temp[1])]
#             conv_size.append(content1)
#
#         if any([name[1] == 'maxpool', name[1] == 'avgpool']):
#             temp = name[0].split('x', 1)
#             content2 = [float(temp[0]), float(temp[1])]
#             pool_size.append(content2)
#
#     # 取出卷积层的数值
#     conv_size2 = []
#     for i in conv_size:
#         if isinstance(i, list):
#             for j in i:
#                 if isinstance(j, list):
#                     for t in j:
#                         conv_size2.append(t)  # 只取出了嵌套的列表
#                 else:
#                     conv_size2.append(j)
#
#     conv_size2 = [conv_size2[i:i+2] for i in range(0, len(conv_size2), 2)]
#
#     # 综合其他参数开始计算网络中的训练参数总量
#     if cell == 1:
#         feed1 = input_channel
#         for i in range(len(conv_size2)):  # 卷积层
#             numbers1 = feed1 * conv_size2[i][0] * conv_size2[i][1] * filter_number[0] + 3 * filter_number[0]  # cell_1
#             feed1 = filter_number[0]
#             numbers += numbers1
#
#         numbers += (classes * feed1 + classes)  # Dense层
#
#     elif cell == 2:
#         feed1 = input_channel
#         for i in range(len(conv_size2)):  # 卷积层
#             numbers1 = feed1 * conv_size2[i][0] * conv_size2[i][1] * filter_number[0] + 3 * filter_number[0]  # cell_1
#             feed1 = filter_number[0]
#             numbers += numbers1
#         feed2 = feed1
#         for i in range(len(conv_size2)):
#             numbers2 = feed2 * conv_size2[i][0] * conv_size2[i][1] * filter_number[1] + 3 * filter_number[1]  # cell_2
#             feed2 = filter_number[1]
#             numbers += numbers2
#
#         numbers += (classes * feed2 + classes)  # Dense层
#
#     else:
#         print('The numbers of cell is undefined!')
#
#     return numbers
#
#
# total_count = counts(action, input_channel=3, filter_number=[32, 64], classes=5, cell=1)
num_acc = []
for i in range(8):
    action = lines1[i][50:(50+(i+1)*4)]
    temp = counts(action, input_channel=3, input_size=[32, 32, 3], stride=2, filter_number=[32, 64],
                  classes=5, B=i+1, cell=2)
    num_acc.append(temp)

num_rew = []
for i in range(8):
    action = lines2[i][50:(50+(i+1)*4)]
    temp = counts(action, input_channel=3, input_size=[32, 32, 3], stride=2, filter_number=[32, 64],
                  classes=5, B=i+1, cell=2)
    num_rew.append(temp)

num_acc.insert(0, np.nan)
num_rew.insert(0, np.nan)
plt.plot(num_acc, '--k', label='standard NAS')
plt.plot(num_rew, 'r', label='balanced NAS')
plt.legend()
plt.xlabel('B')
plt.ylabel('Computation amount')
plt.show()

# network = Sequential()
# network.add(layers.Conv2D(32, kernel_size=(1, 7), padding='SAME', strides=2, activation='relu', input_shape=(512, 512, 3)))
# network.add(layers.Conv2D(32, kernel_size=(7, 1), padding='SAME', strides=2, activation='relu'))
#
# network.add(layers.Conv2D(32, kernel_size=(1, 7), padding='SAME', strides=2, activation='relu'))
# network.add(layers.Conv2D(32, kernel_size=(7, 1), padding='SAME', strides=2, activation='relu'))
#
# network.add(layers.MaxPooling2D(pool_size=3, strides=2))  # 第1个池化层，高宽各减半的池化层
#
# network.add(layers.MaxPooling2D(pool_size=3, strides=2))  # 第1个池化层，高宽各减半的池化层
#
# network.add(layers.Conv2D(32, kernel_size=(1, 7), padding='SAME', strides=2, activation='relu'))
# network.add(layers.Conv2D(32, kernel_size=(7, 1), padding='SAME', strides=2, activation='relu'))
#
# network.add(layers.Conv2D(32, kernel_size=(7, 7), padding='SAME', strides=2, activation='relu'))

# network.add(layers.Conv2D(64, kernel_size=(1, 7), padding='SAME', strides=2, activation='relu'))
# network.add(layers.Conv2D(64, kernel_size=(7, 1), padding='SAME', strides=2, activation='relu'))
#
# network.add(layers.Conv2D(64, kernel_size=(1, 7), padding='SAME', strides=2, activation='relu'))
# network.add(layers.Conv2D(64, kernel_size=(7, 1), padding='SAME', strides=2, activation='relu'))
#
# network.add(layers.MaxPooling2D(pool_size=3, strides=2))  # 第1个池化层，高宽各减半的池化层
#
# network.add(layers.MaxPooling2D(pool_size=3, strides=2))  # 第1个池化层，高宽各减半的池化层
#
# network.add(layers.Conv2D(64, kernel_size=(1, 7), padding='SAME', strides=2, activation='relu'))
# network.add(layers.Conv2D(64, kernel_size=(7, 1), padding='SAME', strides=2, activation='relu'))
#
# network.add(layers.Conv2D(64, kernel_size=(7, 7), padding='SAME', strides=2, activation='relu'))

# network.add(layers.Dense(units=5, activation='softmax'))
# network.summary()

# 4. 调试skip connection，需要创造支路，同时需要drop path
# action = [0, '1x7-7x1 conv', 0, '1x7-7x1 conv', 0, '3x3 maxpool', 0, '3x3 avgpool', 0, '1x7-7x1 conv', 0, '7x7 dconv']
# network = ModelGenerator(action)  # 建立网络，调用时将图像数据输入network

# 5. 调试平衡函数
'''
思路：精度为增函数，运算量函数为减函数；
函数选取幂函数，线性函数，指数函数，sigmoid函数
第一种：令精度的次幂小于运算量，仿真结果发现，当信噪比较低时，高精度的模型得分较小
第二种：令精度的次幂大于运算量
'''
# （一）第一种
# (1) 平衡函数与精度之间的关系
# x = np.linspace(0, 1, 1000)
# F1 = 1/(1+np.exp(-10*(x-0.5)))  # sigmoid函数
# plt.plot(x, F1, label='$F1(x)$', color='green', linewidth=0.5)
# plt.legend()
# plt.show()

# f2 = a**(1/2)  # 幂函数
# plt.plot(a, f2, label='$f2(a)$', color='green', linewidth=0.5)
# plt.show()

# (2) 平衡函数与运算量之间的关系
# x = np.linspace(0, 1, 1000)
# fe_1 = 1/(1+np.exp(10*(x-0.5)))  # sigmoid函数
# plt.plot(x, fe_1, label='$fe_1(x)$', color='green', linewidth=0.5)
# plt.show()

# (3) 画出不同边缘函数组合的图像(二维图)
# a = np.linspace(0, 1, 1000)
# fa_1 = a**(1/2)
# fa_1 = 1/(1+np.exp(-10*(a-0.5)))
# fe_1 = -a+1
# f1 = (1/2) * (fe_1+fa_1)
# plt.plot(a, f1, color='green', linewidth=0.5)
# plt.show()

# F_2 = -x*x
# f2 = (1/2) * (fe_2+fa_1)
# plt.plot(x, F_2, label='$F2(y)$', color='green', linewidth=0.5)
# plt.legend()
# plt.show()

# fe_3 = 1/(a+0.01)
# f3 = (1/2) * (fe_3+fa_1)
# plt.plot(a, f3, color='green', linewidth=0.5)
# plt.show()

# fe_4 = 1/(1+np.exp(10*(a-0.5)))
# f4 = (1/2) * (fe_4+fa_1)
# plt.plot(a, f4, color='green', linewidth=0.5)
# plt.show()

# 三维图
# from mpl_toolkits.mplot3d import Axes3D
# x = np.arange(0, 1, 0.01)  # 精度
# y = np.arange(0, 1, 0.01)  # 运算量
# fig = plt.figure()
# ax = Axes3D(fig)
# X, Y = np.meshgrid(x, y)  # 网格的创建
# # fa_1 = X**(1/2)
# fa_1 = 1/(1+np.exp(-10*(X-0.5)))
# # fe_1 = -Y + 1
# fe_1 = -Y*Y
# Z = 1/2 * (fa_1 + fe_1)
# plt.xlabel('x')
# plt.ylabel('y')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# plt.show()

# （二）第二种
# 此法效果较好
# fa_1 = 1/(1+np.exp(-10*(X-0.5)))
# fe_1 = np.exp(-Y)
# Z = 0.5*(fa_1+fe_1)
# plt.xlabel('x')
# plt.ylabel('y')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# plt.show()


# 利用训练后保存的模型对其他数据集进行分类
os.chdir('E:/project/models/SIGNAL-5/LeNet/noise_free')
root = 'D:/project/CNN_signal/dataset/test'  # 其他数据集所处位置

# 1. 加载模型
new_model = models.load_model('my_model.h5')

# 2. 准备数据集
# 创建CSV
def load_csv(root, filename, name2label):
    # 从csv 文件返回images,labels 列表
    # root:数据集根目录，filename:csv 文件名， name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        # 如果csv文件不存在，则创建
        images = []
        for name in name2label.keys():
            images += glob(os.path.join(root, name, '*.png'))  # 读取文件夹中的png文件
        print(len(images), images)
        random.shuffle(images)  # 随机打乱顺序
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                writer.writerow([img, label])
            print('write into csv file:', filename)

        # 此时已经有csv文件，从csv中读取样本路径和标签
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
         reader = csv.reader(f)
         for row in reader:
             img, label = row
             label = int(label)
             images.append(img)
             labels.append(label)
        # 返回图片路径list和标签list
    return images, labels


def load_image(root):
    # 创建数字编码表
    name2label = {}
    # 遍历根目录下的子文件夹，并排序，保证映射关系固定（所以每一个类要放到一个文件夹中）
    for name in sorted(os.listdir(os.path.join(root))):
        # 跳过非文件夹
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())
        # 读取label信息
    images, labels = load_csv(root, 'images.csv', name2label)
    return images, labels, name2label

# 图片预处理
def preprocess(x, y):
    x = tf.io.read_file(x)  # x为图片的路径list
    x = tf.image.decode_png(x, channels=1)  # 图像解码，灰度为单通道
    x = tf.image.resize(x, [64, 64])  # 图片缩放
    # 转换成张量
    # x: 从[0, 255]转化到[0, 1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y)  # 转换成张量 y为图片的数字编码
    return x, y

images2, labels2, table2 = load_image(root)
test_images = tf.data.Dataset.from_tensor_slices((images2, labels2))  # 所得的为tuple对象
test_images = test_images.shuffle(np.size(images2)).map(preprocess).batch(batchsize)  # 将样本顺序打乱后进行预处理

#  2. 测试
loss, acc = new_model.evaluate(test_images, verbose=2)
























