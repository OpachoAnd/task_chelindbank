from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


class Clusterization:
    def __init__(self):
        pass
    
    def elbow_method(self, df: pd.DataFrame):
        """
        Метод локтя для определения числа кластеров
        Метод строит график зависимости инертности (суммы квадратов расстояний до центров кластеров) 
        от k (число кластеров)
        """
        X = df.values # Извлекаем значения без индекса

        # Стандартизация данных (приведение к среднему 0 и дисперсии 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertia = []  # Инерция - сумма квадратов расстояний точек до центров своих кластеров
        k_range = range(1, 20) # Проверяем от 1 до 10 кластеров

        for k in k_range:
            # n_init='auto' для версий sklearn >= 2.4, или n_init=10 для более старых
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertia, marker='o', linestyle='--')
        plt.title('Метод локтя: поиск оптимального числа кластеров')
        plt.xlabel('Число кластеров (k)')
        plt.ylabel('Инертность (Inertia)')
        plt.xticks(k_range)
        plt.grid(True)
        plt.savefig('elbow_method_graph.png')


    def search_cluster(self, df: pd.DataFrame, k: int):
        df_ = df.copy(deep=True)
        X = df_.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Приводим каждый признак к среднему значению 0 и стандартному отклонению 1
        k_means = KMeans(n_clusters=k, 
                         init='k-means++',
                         n_init='auto',
                         max_iter=300,
                         random_state=42)
        cluster_labels = k_means.fit_predict(X_scaled)
        df_['cluster'] = cluster_labels
        print('df_\n', df_)

        segment_profiles = df_.groupby('cluster').mean().iloc[:, :-1]
        print(segment_profiles)

        # centroids_original_scale = scaler.inverse_transform(k_means.cluster_centers_)
        
        # Названия кластеров для каждой строки
        # labels = k_means.labels_
        # df['Кластер'] = labels

        # print("\nДанные с меткой кластера:")
        # print(df)

        # pca = PCA(n_components=2)
        # reduced_data = pca.fit_transform(X_scaled)
        # print('reduced_data\n', reduced_data)


