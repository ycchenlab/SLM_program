# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 22:23:10 2023

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    result = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        for n in range(N):
            result[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return result

# 生成一个示例信号
fs = 1000  # 采样率
t = np.arange(0, 1, 1/fs)  # 时间
f1 = 50  # 第一个频率成分
f2 = 120  # 第二个频率成分
x = 0.7*np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)  # 合成信号

# 执行DFT
X = dft(x)

# 绘制原始信号和频谱
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("original sig")
plt.xlabel("time")

plt.subplot(2, 1, 2)
freqs = np.fft.fftfreq(len(x), d=1/fs)
plt.stem(freqs, np.abs(X), use_line_collection=True)
plt.title("spec")
plt.xlabel("freq (Hz)")
plt.tight_layout()

plt.show()