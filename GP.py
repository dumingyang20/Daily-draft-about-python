import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve
from numpy.linalg import cholesky
from itertools import cycle
from GP_train import train_data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class SimpleGP():
    """ One dimensional Gaussian Process class.
    Uses squared exponential covariance form.
    parameters
    ----------
    width_scale : float, positive
        Same as sigma in (4) of post
    length_scale : float, positive
        Same as l in (4) of post
    noise : float
        Added to diagonal of covariance, useful for improving convergence
    """

    def __init__(self, width_scale, length_scale, noise=10 ** (-6)):
        self.width_scale = width_scale  # 核函数中的超参数
        self.length_scale = length_scale
        self.noise = noise

    def _exponential_cov(self, x1, x2):
        """
        Return covariance matrix for two arrays, 计算两个阵列的协方差，平方指数核函数
        with i-j element = cov(x_1i, x_2j).
        parameters
        ----------
        x1, x2: np.array
            arrays containing x locations
        """
        dis = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        # return (self.width_scale ** 2) * np.exp(-np.subtract.outer(x1, x2) ** 2 / (2 * self.length_scale ** 2))
        return (self.width_scale ** 2) * np.exp(-dis / (2 * self.length_scale ** 2))

    def fit(self, sample_x, sample_y):
        """
        Save for later use the Cholesky matrix
        associated with the inverse that appears
        in (5) of post. Also evaluate the weighted
        y vector that appears in that equation.
        parameters
        ----------
        sample_x : np.array
            locations where we have sampled
        sample_y : np.array
            y values observed at each sample location
        sample_s : np.array
            array of stds for each sample
        """

        self.sample_x = np.array(sample_x)

        S = self._exponential_cov(sample_x, sample_x)  # 协方差矩阵K(X,X)
        # d = np.diag(np.array(sample_s) ** 2 + self.noise)  # 构造对角矩阵 方差+噪声

        # self.lower_cholesky = cholesky(S + d)  # cholesky分解，得到下三角矩阵，条件：对称正定阵
        self.lower_cholesky = cholesky(S + 1e-8 * np.eye(len(S)))
        self.weighted_sample_y = cho_solve((self.lower_cholesky, True), sample_y)
        # 上述两步为求解：(K(X,X)+std^2*I) * weighted_sample_y = sample_y
        # 为后续的预测做准备

    def interval(self, test_x):
        """
        Obtain the one-sigam confidence interval  置信区间
        for a set of test points parameters
        ----------
        test_x : np.array
            locations where we want to test
        """
        # test_x = np.array([test_x]).flatten()
        test_x = np.array(test_x)
        # means, stds = [], []
        S0 = self._exponential_cov(self.sample_x, test_x)  # K(X,X')
        v = cho_solve((self.lower_cholesky, True), S0)
        means = np.dot(S0.T, self.weighted_sample_y)  # 均值与sample相同
        cov = self._exponential_cov(test_x, test_x) - np.dot(S0.T, v)
        # for row in test_x:
        #     S0 = self._exponential_cov(row, self.sample_x)  # K(X,X')
        #     v = cho_solve((self.lower_cholesky, True), S0)
        #     means.append(np.dot(S0, self.weighted_sample_y))  # 均值与sample相同
        #     stds.append(np.sqrt(self.width_scale ** 2 - np.dot(S0, v)))
        return means, cov


# Insert data here.
sample_x = train_data[:, 0:2]  # 训练点集
sample_y = train_data[:, 2]
# sample_s = [0.01, 0.05, 0.125]

WIDTH_SCALE = 1
LENGTH_SCALE = 1
SAMPLES = 8
model = SimpleGP(WIDTH_SCALE, LENGTH_SCALE)
model.fit(sample_x, sample_y)

# 测试
test_d1 = np.arange(0, 1, .02)  # 50个x坐标
test_d2 = np.arange(0, 1, .02)  # 50个y坐标
test_d1, test_d2 = np.meshgrid(test_d1, test_d2)  # 生成网格
test_x = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]  # 共2500个测试点
means, cov = model.interval(test_x)
z = means.reshape(test_d1.shape)  # 预测值
# samples = model.sample(test_x, SAMPLES)


# plots here.
fig = plt.figure(figsize=(7, 5))
ax = Axes3D(fig)
ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)  # 绘制表面
ax.scatter(np.asarray(sample_x)[:, 0], np.asarray(sample_x)[:, 1], sample_y, c=sample_y, cmap=cm.coolwarm)  # 在表面绘制散点
ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)  # 绘制投影
# plt.savefig('GP.pdf')  # 保存成为PDF格式
plt.show()

