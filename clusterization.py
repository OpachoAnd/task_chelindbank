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

        inertia = []
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

        # plt.show()


    def search_cluster(self, df: pd.DataFrame):
        print(df)
        scaler = StandardScaler()
        # Приводим каждый признак к среднему значению 0 и стандартному отклонению 1
        X_scaled = scaler.fit_transform(df)
        print('X_scaled\n', X_scaled)

        k_means = KMeans(n_clusters=8 , 
                         init='k-means++',
                         n_init='auto',
                         max_iter=300,
                         random_state=42)
        clusters = k_means.fit_predict(X_scaled)

        centroids_original_scale = scaler.inverse_transform(k_means.cluster_centers_)
        print("Центроиды кластеров:")
        print(pd.DataFrame(centroids_original_scale, columns=df.columns))
        
        # Названия кластеров для каждой строки
        labels = k_means.labels_
        df['Кластер'] = labels

        print("\nДанные с меткой кластера:")
        print(df)

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X_scaled)
        print('reduced_data\n', reduced_data)
        # plt.figure(figsize=(10, 7))
        # sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette='viridis', s=100)
        # plt.title('Кластеризация K-Means (4 признака спроецированы в 2D через PCA)')
        # plt.xlabel('Первая главная компонента')
        # plt.ylabel('Вторая главная компонента')
        # plt.legend(title='Кластер')
        # plt.show()

        # Явно создаем фигуру и оси
        # fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=100, c='blue', marker='o', alpha=0.7)
        # Добавляем подписи и заголовок
        plt.xlabel('Ось X')
        plt.ylabel('Ось Y')
        plt.title('Диаграмма рассеяния')
        plt.savefig('sine_wave.png')
        # ax.plot(reduced_data[:, 0], reduced_data[:, 1], label='Кластер')
        # ax.set_title('Кластер')
        # ax.set_xlabel('Ось X')
        # ax.set_ylabel('Ось Y')
        # ax.grid(True)
        # ax.legend()

        # fig.savefig('sine_wave.png')
        # plt.close(fig) # Освобождаем память

