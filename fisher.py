import random
import numpy as np

def creatdata(total):
    sample = []
    label = []
    for i in range(total):
        a = round(random.uniform(-2,2),2)
        b = round(random.uniform(-2,2),2)
        sample.append([a, b])
        if a + b >= 0:
            label.append(int(1))
        else:
            label.append(int(-1))
    return np.array(sample), label


def fisher(sample, label, total):
    miu1 = [0, 0]   #+1类均值
    miu2 = [0, 0]   #-1类均值
    miu1_num = 0    #+1类数量
    miu2_num = 0    #-1类数量
    sum1 = np.array([[0, 0],
                     [0, 0]])   #+1类协方差
    sum2 = np.array([[0, 0],
                     [0, 0]])   #-1类协方差
    for i in range(total):
        if label[i] == 1:
            miu1 = miu1 + sample[i]
            miu1_num = miu1_num + 1
        if label[i] == -1:
            miu2 = miu2 + sample[i]
            miu2_num = miu2_num + 1
    miu1 = miu1 / miu1_num
    miu2 = miu2 / miu2_num

    for i in range(total):
        if label[i] == 1:
            a = sample[i] - miu1
            a = np.mat(a)
            sum1 = sum1 + a.T @ a
        if label[i] == -1:
            b = sample[i] - miu2
            b = np.mat(b)
            sum2 = sum2 + b.T @ b

    sw = sum1 + sum2  #类内总离差阵
    sw_1 = sw.I
    a = np.mat(miu1 - miu2)
    w = sw_1 @ a.T
    b = np.mat(miu1 + miu2)
    y0 = w.T @ b.T

    return w, y0/2

def fisher_accuracy(sample, label, w, s, total):
    error = 0
    for i in range(total):
        if w.T @ sample[i] > s and label[i] != 1:
            error = error + 1
        if w.T @ sample[i] < s and label[i] != -1:
            error = error + 1

    accuracy = 1 - error / total
    return accuracy


total = 10
data, y = creatdata(total)
# data = np.array([[5, 37],
#                  [7, 30],
#                  [10, 35],
#                  [11.5, 40],
#                  [14, 38],
#                  [12, 31],
#                  [35, 21.5],
#                  [39, 21.7],
#                  [34, 16],
#                  [37, 17]])
# y = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1]
w,s = fisher(data, y, total)
accuracy = fisher_accuracy(data, y, w, s, total)
# print(w)
# print(s)
# print(accuracy)

