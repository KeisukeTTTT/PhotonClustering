import numpy as np
import matplotlib.pyplot as plt
import tes_analysis_tools as tat
import sklearn
from sklearn.decomposition import PCA

# -----------------------
# データ読み込み
# -----------------------

dir = "../WF/exp1/"
savedir = "./exp1_result/"
pulse = np.load(dir + "pulse.npy")
noise = np.load(dir + "noise.npy")
time = np.load(dir + "time.npy")

n = pulse.shape[0]
dp = pulse.shape[1]
m = noise.shape[0]
dt = time[1] - time[0]

print("n  = ", n, " (num of pulse data) ")
print("m  = ", m, " (num of noise data) ")
print("dp = ", dp)
print("dt = ", dt)


# baseline 補正, 先頭からdpbl点の平均をoffsetとして引き去る
dpbl = 200
pulse = tat.correct_baseline(pulse, dpbl)

# 正負反転
pulse = -1.0 * pulse


# 波形確認
for i in range(0, 5):
    plt.plot(pulse[i, :])
    plt.show()


# ------------------------------------
# (A) 従来型の波形処理
# ------------------------------------

# 波形整形
pulse = tat.shaping(pulse, dt, 1.0e-6, 1.0e-6, True)

# 平均波形
avg = tat.make_average_pulse(pulse, 0.0, 1.0, 200, 500, True, True, True)

# 単純波高値
tat.simple_ph_spectrum(pulse, 200, 500, True, True)

# 最適フィルタ
ph, hist = tat.optimal_filter_freq(pulse, avg, noise, dt, 1.0e6, True, True)


# ------------------------------------
# (B) クラスタリング手法
# ------------------------------------

# 主成分分析
print(pulse.shape)
pca = PCA(n_components=2)
pca.fit(pulse)
X_pca = pca.transform(pulse)
print(X_pca.shape)

# 第一主成分のヒストグラム
plt.hist(X_pca[:, 0], bins=256)
plt.show()

# 第一, 第二主成分の散布図 <- これをクラスタリングしたい
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=2)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
