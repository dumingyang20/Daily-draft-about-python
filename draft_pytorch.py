"""日常的草稿"""


import numpy as np
import csv
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import os
import torch

# 1. 调试新的balance function

x = np.random.rand(100)  # accuracy
y = np.random.rand(100)  # regularized computation amounts

# a = 20 * np.random.rand(50)  # weight_1
# b = 20 * np.random.rand(50)  # weight_2

m = []
n = []
metric = []
train_data = []

lamd = 0.2  # threshold for acc
mu = 0.2  # threshold for computation amount
# 每次从x,y中各取出两个值进行比较
for p in range(len(x)-1):
    fx1 = 1 / (1 + np.exp(-10 * (x[p] - 0.5)))
    fx2 = 1 / (1 + np.exp(-10 * (x[p+1] - 0.5)))

    for q in range(len(y)-1):
        fy1 = np.exp(-y[q])
        fy2 = np.exp(-y[q+1])

        # judge procedure
        # save all qualified x,y,fx+fy to fit a Gaussian process

        if x[p]-x[p+1] > lamd and mu < y[q]-y[q+1] < 2*mu:
            if fx1+fy1 > fx2+fy2:
                metric.append(1)
                train_data.append([])
            else:
                metric.append(0)

        elif x[p]-x[p+1] > lamd and y[q]-y[q+1] > 2*mu:
            if fx1+fy1 < fx2+fy2:
                metric.append(2)
            else:
                metric.append(0)

        elif x[p]-x[p + 1] > -lamd and y[q]-y[q + 1] < -2*mu:
            if fx1+fy1 > fx2+fy2:
                metric.append(3)
            else:
                metric.append(0)

        elif x[p]-x[p + 1] < -2*lamd:
            if fx1+fy1 < fx2+fy2:
                metric.append(4)
            else:
                metric.append(0)


# 画出metric的柱状图，以观察条件满足情况
data_dict = {}
for key in metric:
    data_dict[key] = data_dict.get(key, 0) + 1

print("data_dict:", data_dict)
num = [index[1] for index in sorted(data_dict.items())]  # 统计符合各约束条件的数量
labels = ['No', 'Constraint 1', 'Constraint 2', 'Constraint 3', 'Constraint 4']
plt.bar(range(len(num)), num, tick_label=labels)
plt.show()


















