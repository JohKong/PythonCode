import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.colors import Normalize
from matplotlib import cm

# 生成模拟数据：正弦波 + 噪声
np.random.seed(0)
years = np.linspace(1900, 2020, 500)  # 从 1900 到 2020 共 500 个点
signal = np.sin(2 * np.pi * years / 8) + 0.5 * np.random.randn(len(years))  # 周期为 8 年的正弦波，加上噪声

# 小波变换参数
scales = np.arange(1, 128)  # 根据信号周期调整尺度范围
wavelet = 'cmor1.5-1.0'     # 使用 Morlet 小波

# 小波变换
coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=(years[1] - years[0]))
power = (abs(coefficients)) ** 2  # 计算功率谱

# 设置显著性水平（95%）的阈值
significance_threshold = np.percentile(power, 95)
significant_region = power >= significance_threshold

# 绘制小波功率谱图
plt.figure(figsize=(12, 6))
norm = Normalize(vmin=power.min(), vmax=power.max())
cmap = cm.jet
plt.contourf(years, scales, power, levels=20, cmap=cmap, norm=norm)
plt.colorbar(label='Power', extend='both')

# 添加显著性区域打点
plt.contourf(years, scales, significant_region, levels=[0.5, 1], colors='none', hatches=['....'])

# 设置影响锥区域（即锥形边界，假设影响锥为最大周期的一半）
cone_of_influence = scales[-1] / 2
plt.fill_between(years, cone_of_influence, scales[-1], color='black', alpha=0.3)

# 设置坐标轴和标题
plt.yscale('log')  # 对数尺度显示周期
plt.ylabel('Period (year)')
plt.xlabel('Year')
plt.title('Wavelet Power Spectrum with Significance and Cone of Influence')

# 添加 95% 信度水平区域的等高线
plt.contour(years, scales, power, levels=[significance_threshold], colors='black', linewidths=0.5)

plt.show()
