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

a = 20 * np.random.rand(50)  # weight_1
b = 20 * np.random.rand(50)  # weight_2

m = []
n = []

lamd = 0.1  # threshold for acc
mu = 0.2  # threshold for computation amount
for i in range(len(a)):
    for j in range(len(b)):
        print("a = %f, b = %f" % (a[i], b[j]))
        # 每次从x,y中各取出两个值进行比较
        for p in range(len(x)-1):
            if x[p] > x[p+1]:
                fx1 = 1 / (1 + np.exp(-a[i] * (x[p] - 0.5)))
                fx2 = 1 / (1 + np.exp(-a[i] * (x[p + 1] - 0.5)))
            else:  # 调整，使得前一项大于后一项
                t = x[p]
                x[p] = x[p+1]
                x[p+1] = t

                fx1 = 1 / (1 + np.exp(-a[i] * (x[p] - 0.5)))
                fx2 = 1 / (1 + np.exp(-a[i] * (x[p+1] - 0.5)))

            for q in range(len(y)-1):
                if y[q] > y[q + 1]:
                    fy1 = np.exp(-b[j] * y[q])
                    fy2 = np.exp(-b[j] * y[q+1])
                else:
                    t = y[q]
                    y[q] = y[q + 1]
                    y[q + 1] = t

                    fy1 = np.exp(-b[j] * y[q])
                    fy2 = np.exp(-b[j] * y[q+1])

                if x[p]-x[p+1] < lamd and y[q]-y[q+1] > mu and fx1+fy1 < fx2+fy2:
                    if x[p]-x[p+1] > lamd and y[q]-y[q+1] < mu and fx1+fy1 > fx2+fy2:
                        # if x[p] - x[p + 1] < lamd and y[q] - y[q + 1] > mu and fx1 + fy1 < fx2 + fy2:
                            if x[p] - x[p + 1] < lamd and y[q] - y[q + 1] < mu and fx1 + fy1 > fx2 + fy2:
                                print("a,b is qualified")
                                m.append(a[i])
                                n.append(b[j])













