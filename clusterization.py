from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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

    def silhouette_method(self, df: pd.DataFrame):
        """
        Средний силуэтный коэффициент (Silhouette Score) для определения числа кластеров
        """
        k_1 = 2  # Силуэт коэфф. рассчитывается с k_1 кластеров
        k_2 = 11  # Силуэт коэфф. рассчитывается до k_2 кластеров
        X = df.values  # Извлекаем значения без индекса

        # Стандартизация данных (приведение к среднему 0 и дисперсии 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        silhouette_avg_scores = []
        k_range_sil = range(k_1, k_2) # Силуэт рассчитывается начиная с 2 кластеров

        for k in k_range_sil:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            score = silhouette_score(X_scaled, cluster_labels)
            silhouette_avg_scores.append(score)
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_range_sil, silhouette_avg_scores, marker='s', color='red', linestyle='--')
        plt.title("Средний силуэтный коэфф.")
        plt.xlabel("Число кластеров (k)")
        plt.ylabel("Средний силуэтный коэфф.")
        plt.xticks(k_range_sil)
        plt.grid(True)
        plt.savefig('silhouette_method_graph.png')


    def search_cluster(self, df: pd.DataFrame, k: int):
        """
        Метод для поиска кластеров

        return Клиенты с кластерами, Доля расходов кластеров в соответствующей макрокатегории
        """
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

        # Клиенты с кластерами
        df_['cluster'] = cluster_labels

        # Доля расходов кластеров в соответствующей макрокатегории
        share_expenses = df_.groupby('cluster').mean()

        return df_, share_expenses
