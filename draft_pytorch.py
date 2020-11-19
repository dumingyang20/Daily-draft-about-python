"""日常的草稿"""

import numpy as np
import csv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import os
import torch
import random
from matplotlib.pyplot import MultipleLocator


# 1. 调试新的balance function

# np.random.seed(26)
# x = np.random.rand(200)  # accuracy
# y = np.random.rand(200)  # regularized computation amounts
#
# # a = 20 * np.random.rand(50)  # weight_1
# # b = 20 * np.random.rand(50)  # weight_2
#
# m = []
# n = []
# metric = []
# data = [[], [], [], [], []]
# lamd = 0.2  # threshold for acc
# mu = 0.2  # threshold for computation amount
# # 每次从x,y中各取出两个值进行比较
# for p in range(len(x)-1):
#     fx1 = 1 / (1 + np.exp(-10 * (x[p] - 0.5)))
#     fx2 = 1 / (1 + np.exp(-10 * (x[p+1] - 0.5)))
#
#     for q in range(len(y)-1):
#         fy1 = np.exp(-y[q])
#         fy2 = np.exp(-y[q+1])
#
#         # judge procedure
#         # save all qualified x,y,fx+fy to fit a Gaussian process
#
#         if x[p]-x[p+1] > lamd and mu < y[q]-y[q+1] < 2*mu:
#             if fx1+fy1 > fx2+fy2:
#                 metric.append(1)
#                 data[1].append([[x[p], y[q], fx1+fy1], [x[p+1], y[q+1], fx2+fy2]])
#
#             else:
#                 metric.append(0)
#
#         elif x[p]-x[p+1] > lamd and y[q]-y[q+1] > 2*mu:
#             if fx1+fy1 < fx2+fy2:
#                 metric.append(2)
#                 data[2].append([[x[p], y[q], fx1 + fy1], [x[p + 1], y[q + 1], fx2 + fy2]])
#             else:
#                 metric.append(0)
#
#         elif x[p]-x[p + 1] > -lamd and y[q]-y[q + 1] < -2*mu:
#             if fx1+fy1 > fx2+fy2:
#                 metric.append(3)
#                 data[3].append([[x[p], y[q], fx1 + fy1], [x[p + 1], y[q + 1], fx2 + fy2]])
#             else:
#                 metric.append(0)
#
#         elif x[p]-x[p + 1] < -2*lamd:
#             if fx1+fy1 < fx2+fy2:
#                 metric.append(4)
#                 data[4].append([[x[p], y[q], fx1 + fy1], [x[p + 1], y[q + 1], fx2 + fy2]])
#             else:
#                 metric.append(0)
#
#
# # 画出metric的柱状图，以观察条件满足情况
# data_dict = {}
# for key in metric:
#     data_dict[key] = data_dict.get(key, 0) + 1
#
# # print("data_dict:", data_dict)
# num = [index[1] for index in sorted(data_dict.items())]  # 统计符合各约束条件的数量
# # labels = ['No', 'Constraint 1', 'Constraint 2', 'Constraint 3', 'Constraint 4']
# # plt.bar(range(len(num)), num, tick_label=labels)
# # plt.show()
#
#
# # 统计满足条件最少的点，作为GP的训练数据
# train_data_index = num.index(min(num))
# train_data = np.array(data[train_data_index])
# train_data = train_data.reshape(-1, 3)


# 2. plot according to the experimental results
# compare their accuracy and computation cost mainly
# define a function to traverse folder and extract target data among specific files


# def get_data(dir_name, file_name, key):
#     assert os.path.isdir(dir_name)  # 确定为文件夹
#     data = []
#     for s in os.listdir(dir_name):
#         if not os.path.isfile(s):  # traverse folder only
#             target_dir = os.path.join(os.path.join(dir_name, s), file_name)
#             assert os.path.exists(target_dir)  # 确定指定文件存在
#             # load csv
#             sub_data = load_csv(target_dir, key)
#             data.append(sub_data)
#
#     return data
#
#
# def load_csv(file_name, key):  # output the whole column according to the 'key'
#
#     with open(file_name) as f:
#         reader = csv.reader(f)
#         result = list(reader)
#
#     index = result[0].index(key)
#     result = np.array(result)  # transfer list to array
#     column = result[:, index][1:]  # get values in target column
#     column = list(filter(lambda s: s and s.strip(), column))  # delete empty string
#     data = []
#     for element in column:
#         data.append(float(element))  # transfer string to float
#
#     return data
#
#
# root = '/Users/mydu/project/Gaussian-process-for-the-NAS/results/block/lambda=0.3,mu=0.1'
#
# acc_only = get_data(root, 'AAC.csv', 'accuracy')  # accuracy-only model
# comp_only = get_data(root, 'AAC.csv', 'computation')
#
# acc_GP = get_data(root, 'SAC_GP.csv', 'accuracy')  # GP-based model
# comp_GP = get_data(root, 'SAC_GP.csv', 'computation')
#
# acc_ini = get_data(root, 'SAC_ini.csv', 'accuracy')  # initialization model
# comp_ini = get_data(root, 'SAC_ini.csv', 'computation')
#
#
# # plot here
# # (1) curve line figure
# plt.figure(1)
# plt.plot([comp_only[:][i][2] for i in range(len(comp_only))], label='accuracy_only model')
# plt.plot([comp_GP[:][i][2] for i in range(len(comp_only))], '-*', label='GP-based model')
# plt.plot([comp_ini[:][i][2] for i in range(len(comp_only))], '-+', label='initialization model')
# plt.legend()
# plt.xlabel('Block')
# plt.ylabel('Computation cost')
# plt.grid()
# # plt.show()
# # plt.savefig('Computation cost.pdf')
#
# plt.figure(2)
# plt.plot([acc_only[:][i][2] for i in range(len(acc_only))], label='accuracy_only model')
# plt.plot([acc_GP[:][i][2] for i in range(len(acc_GP))], '-*', label='GP-based model')
# plt.plot([acc_ini[:][i][2] for i in range(len(acc_ini))], '-+', label='initialization model')
# plt.ylim((0, 1))
# plt.legend()
# plt.xlabel('Block')
# plt.ylabel('Validation accuracy')
# plt.grid()
# # plt.show()
# # plt.savefig('Validation accuracy.pdf')
#
#
# plt.figure(3)
# plt.plot(acc_GP[:][:], 'r-')
# plt.plot(acc_only[:][:], 'b-+')
# plt.plot(acc_ini[:][:], 'k--')
# plt.ylim((0, 1))
# plt.xlabel('Block')
# plt.ylabel('Validation accuracy')
# plt.grid()
# # plt.show()
# # plt.savefig('Validation accuracy_1.pdf')
#
# plt.figure(4)
# plt.plot(comp_GP[:][:], 'r-')
# plt.plot(comp_only[:][:], 'b-+')
# plt.plot(comp_ini[:][:], 'k--')
# plt.xlabel('Block')
# plt.ylabel('Computation cost')
# plt.grid()
# # plt.show()
# # plt.savefig('Computation cost_2.pdf')
#
#
# # (2) scatter figure
# model_index = np.arange(1, 11)
# plt.figure(5)
# for i in range(len(acc_GP)):
#     plt.scatter(model_index, acc_GP[:][i], marker='o', c='r')
# for i in range(len(acc_ini)):
#     plt.scatter(model_index, acc_ini[:][i], marker='x', c='k')
# for i in range(len(acc_only)):
#     plt.scatter(model_index, acc_only[:][i], marker='v', c='b')
# plt.xlabel('Model Index')
# plt.ylabel('Accuracy')
#
# x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
# ax = plt.gca()  # ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# plt.show()


# 3. 分割csv文件为多个文件
def separate(root, files, B, K):
    # load csv file
    os.chdir(root)
    with open(files) as f:
        reader = csv.reader(f)
        result = list(reader)

    assert B == (len(result)-36)/K + 1
    for i in range(B):
        if i == 0:
            # write the results of this trial into a file
            with open('train_history_1.csv', mode='a+', newline='') as f:
                data = [result[0]]
                data.extend(result[j+1] for j in range(35))
                writer = csv.writer(f)
                writer.writerows(data)  # 多行写入

        else:
            file_name = 'train_history_'+str(i+1)+'.csv'
            with open(file_name, mode='a+', newline='') as f:
                data = result[36:(36+K*i)]
                writer = csv.writer(f)
                writer.writerows(data)


root = '/Users/mydu/project/Gaussian-process-for-the-NAS/results/block/lambda=0.3,mu=0.1/B=8'
separate(root, 'train_history(accuracy).csv', B=8, K=10)

