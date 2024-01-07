# import numpy as np
#
# x = np.arange(0, 12).reshape((2,6))
# print(x, '\n')
# b = np.mean(x, axis=0, keepdims=True)
# # b = np.mean(x)
# # b = np.mean(x, axis=1, keepdims=True)
# print(b)
#
# print(x[-1])
# print('================')
# for layer in reversed(x[-1, :]):
#     print(layer)
#
# print(help(reversed))

import numpy as np
import matplotlib.pyplot as plt

# 导入数据集
data = np.loadtxt('xor_dataset.csv', delimiter=',')
print('数据集大小：', len(data))
print(data[:5])

# 划分训练集与测试集
ratio = 0.8
split = int(ratio * len(data))
np.random.seed(0)
data = np.random.permutation(data)
# y的维度调整为(len(data), 1)，与后续模型匹配
x_train, y_train = data[:split, :2], data[:split, -1].reshape(-1, 1)
x_test, y_test = data[split:, :2], data[split:, -1].reshape(-1, 1)