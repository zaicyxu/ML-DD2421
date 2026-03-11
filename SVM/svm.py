import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Define kernal
def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, p=3):
    return (np.dot(x, y) + 1) ** p


def rbf_kernel(x, y, sigma=0.5):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


# 全局变量初始化
P = None
targets = None
inputs = None
C = None
kernel = linear_kernel  # 默认使用线性核


# 目标函数
def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)


# 等式约束函数
def zerofun(alpha):
    return np.dot(alpha, targets)


# 计算偏置b
def compute_b(alpha, inputs, targets, kernel):
    sv_indices = (alpha > 1e-5) & (alpha < C - 1e-5)  # 支持向量（0 < alpha_i < C）
    sv = inputs[sv_indices]
    sv_t = targets[sv_indices]
    sv_alpha = alpha[sv_indices]

    if len(sv_alpha) == 0:
        return 0  # 如果没有满足条件的支持向量，默认b=0
    b_sum = 0
    for i in range(len(sv_alpha)):
        b_sum += sv_t[i] - np.sum(alpha * targets * np.array([kernel(inputs[j], sv[i]) for j in range(len(alpha))]))
    return b_sum / len(sv_alpha)


# 生成测试数据
def generate_data():
    np.random.seed(100)
    classA = np.concatenate(
        (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]
    return inputs, targets


# 主程序
def main():
    global P, targets, inputs, C, kernel
    inputs, targets = generate_data()
    N = inputs.shape[0]

    # 设置参数
    C = 1.0  # 松弛变量参数
    kernel = rbf_kernel  # 切换核函数：linear/polynomial/rbf

    # 预计算P矩阵
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])

    # 初始猜测和约束
    start = np.zeros(N)
    bounds = [(0, C) for _ in range(N)]
    constraints = {'type': 'eq', 'fun': zerofun}

    # 调用优化器
    ret = minimize(objective, start, bounds=bounds, constraints=constraints)
    alpha = ret['x']

    # 提取支持向量
    sv_indices = alpha > 1e-5
    sv_alpha = alpha[sv_indices]
    sv = inputs[sv_indices]
    sv_t = targets[sv_indices]

    # 计算b
    b = compute_b(alpha, inputs, targets, kernel)

    # 定义指示函数
    def indicator(x, y):
        s = np.array([x, y])
        result = 0
        for i in range(len(sv_alpha)):
            result += sv_alpha[i] * sv_t[i] * kernel(sv[i], s)
        return result - b

    # 绘制数据点和支持向量
    plt.plot([p[0] for p in inputs[targets == 1]], [p[1] for p in inputs[targets == 1]], 'b.')
    plt.plot([p[0] for p in inputs[targets == -1]], [p[1] for p in inputs[targets == -1]], 'r.')
    plt.plot(sv[:, 0], sv[:, 1], 'go', markersize=10, fillstyle='none')  # 支持向量用绿色圆圈标记

    # 绘制决策边界和边际
    xgrid = np.linspace(-5, 5, 100)
    ygrid = np.linspace(-4, 4, 100)
    grid = np.array([[indicator(x, y) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()