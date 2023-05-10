import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tes_analysis_tools as tat
import sklearn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.distribution import MultiprocessingDistributor

sns.set()

dir = "../WF/exp1/"
savedir = "./exp1_result/"
pulse = np.load(dir + "pulse.npy")
noise = np.load(dir + "noise.npy")
time = np.load(dir + "time.npy")

n = pulse.shape[0]
dp = pulse.shape[1]
m = noise.shape[0]
dt = time[1] - time[0]

# baseline 補正, 先頭からdpbl点の平均をoffsetとして引き去る
dpbl = 200
pulse = tat.correct_baseline(pulse, dpbl)

# 正負反転
pulse = -1.0 * pulse


# データ補正 <- 18次回帰曲線
def regression(data):
    fit = np.polyfit(time, data, 18)
    fit_fn = np.poly1d(fit)
    return fit_fn(time)


def remove_constant_columns(df):
    df_filtered = df.loc[:, df.nunique() != 1]
    return df_filtered


def remove_binary_columns(df):
    columns_to_select = df.apply(lambda col: not set(col.unique()).issubset({0, 1}))
    df_filtered = df.loc[:, columns_to_select]
    return df_filtered


def remove_constant_or_binary_columns(df):
    columns_to_select = df.apply(lambda col: (col.nunique() != 1) and not set(col.unique()).issubset({0, 1}))
    df_filtered = df.loc[:, columns_to_select]
    return df_filtered


def standardize(df):
    scaler = StandardScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_std


def remove_corr_cols(df, threshold=0.7):
    corr = df.corr().abs()
    high_corr = np.where(corr > threshold)
    to_drop = []
    for idx, col in zip(high_corr[0], high_corr[1]):
        f1, f2 = corr.index[idx], corr.columns[col]
        if f1 != f2 and f1 not in to_drop and f2 not in to_drop:
            to_drop.append(f1)
    df_filtered = df.drop(to_drop, axis=1)
    return df_filtered


def preprocess(df, threshold=0.7):
    df_filtered = remove_constant_or_binary_columns(df)
    df_filtered = standardize(df_filtered)
    df_filtered = remove_corr_cols(df_filtered, threshold)
    return df_filtered


class Clustering:
    def __init__(self, pulse, time, tsfresh_create):
        self.time = time
        self.df = pd.DataFrame(pulse)
        self.df.index.name = "id"
        self.df_corrected = self.df.apply(regression, axis=1, result_type="expand")
        self.pulse = self.df_corrected.values
        self.X_pca = self.pca(self.pulse)
        self.tsfresh_features = self.tsfresh(self.df_corrected, tsfresh_create)
        self.X_pca_tsfresh = self.pca(self.tsfresh_features)

    def pca(self, data):
        pca_model = PCA(n_components=2)
        X_pca = pca_model.fit_transform(data)
        return X_pca

    def kmeans(self, data, n_clusters):
        model = KMeans(n_clusters=n_clusters)
        y_pred = model.fit_predict(data)
        return y_pred

    def dbscan(self, data, eps, min_samples):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = model.fit(data).labels_
        return y_pred

    def tsfresh(self, df, create):
        if create:
            # tsfreshに対応した形に変換
            df_tsfresh = df.stack().reset_index()
            df_tsfresh.columns = ["id", "time", "voltage"]
            df_tsfresh["id"] = df_tsfresh["id"].astype("object")
            df_tsfresh["time"] = df_tsfresh["time"].astype("object")

            # tsfreshによる特徴量作成
            Distributor = MultiprocessingDistributor(n_workers=os.cpu_count(), disable_progressbar=False, progressbar_title="Feature Extraction")
            extracted_features = extract_features(df_tsfresh, column_id="id", column_sort="time", distributor=Distributor)
            impute(extracted_features)
            extracted_features = preprocess(extracted_features)

            # csvファイルに保存
            extracted_features.to_csv("../data/tsfresh_features.csv", index=False)
        else:
            extracted_features = pd.read_csv("../data/tsfresh_features.csv")

        return extracted_features

    def plot_pca_result(self):
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], s=2)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig("../images/pca_result.png")
        plt.close()

    def plot_clusters(self, X, cluster_ids, cluster_id_range, fig_file):
        for class_value in cluster_id_range:
            row_ix = np.where(cluster_ids == class_value)
            plt.scatter(X[row_ix, 0], X[row_ix, 1], s=1, label=class_value)
        plt.legend()
        # plt.show()
        plt.savefig(fig_file)
        plt.close()

    def pca_kmeans(self):
        y_pred = self.kmeans(self.X_pca, n_clusters=6)
        self.plot_clusters(self.X_pca, y_pred, list(set(y_pred)), "../images/pca_kmeans.png")

    def pca_dbscan(self):
        y_pred = self.dbscan(self.X_pca, eps=0.014, min_samples=100)
        self.plot_clusters(self.X_pca, y_pred, list(set(y_pred)), "../images/pca_dbscan.png")

    def tsfresh_pca_kmeans(self):
        y_pred = self.kmeans(self.X_pca_tsfresh, n_clusters=6)
        self.plot_clusters(self.X_pca_tsfresh, y_pred, list(set(y_pred)), "../images/tsfresh_pca_kmeans.png")

    def tsfresh_pca_dbscan(self):
        y_pred = self.dbscan(self.X_pca_tsfresh, eps=6, min_samples=50)
        self.plot_clusters(self.X_pca_tsfresh, y_pred, list(set(y_pred)), "../images/tsfresh_pca_dbscan.png")

    def cluster_hist(self, labels):
        plt.hist(labels)
        plt.show()
        plt.close()


if __name__ == "__main__":
    """
    1. 主成分分析
        i. K-Means
        ii. DBSCAN
    2. tsfresh -> 主成分分析
        i. K-Means
        ii. DBSCAN
    """

    my_model = Clustering(pulse, time, tsfresh_create=False)

    my_model.plot_pca_result()
    # 1-i: 主成分分析 -> K-Means
    my_model.pca_kmeans()
    # 1-ii: 主成分分析 -> DBSCAN
    my_model.pca_dbscan()
    # 2-i: tsfresh -> 主成分分析 -> K-Means
    my_model.tsfresh_pca_kmeans()
    # 2-ii: tsfresh -> 主成分分析 -> DBSCAN
    my_model.tsfresh_pca_dbscan()
