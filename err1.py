

def cov_matrix():

    import numpy as np
# Создание массива
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    data1 = np.array([[1, 2, 3]])

    # Вычисление ковариационной матрицы с rowvar=True
    cov_matrix_rowvar = np.cov(data, data1, rowvar=False)



    print("Ковариационная матрица с rowvar=True:")
    print(cov_matrix_rowvar)


def sco():
    import numpy as np

    scores = [80, 85, 90, 75, 95]

    # Вычисляем среднее значение
    mean = np.mean(scores)
    print(mean)

    # Вычисляем разности между значениями и средним значением
    diffs = [score - mean for score in scores]

    # Возведение разностей в квадрат
    squared_diffs = [diff ** 2 for diff in diffs]

    # Вычисляем среднее значение квадратов разностей
    mean_squared_diffs = np.mean(squared_diffs)

    # Извлекаем квадратный корень из среднего значения квадратов разностей
    std_deviation = np.sqrt(mean_squared_diffs)

    print("Стандартное отклонение:", std_deviation)


def l1():
    from sklearn.decomposition import KernelPCA
    from sklearn.datasets import make_circles
    import matplotlib.pyplot as plt

    # Создаем нелинейные данные
    X, y = make_circles(n_samples=100, random_state=42)
    print(X.shape)
    print(y)
    # Создаем объект ядерного PCA
    kpca = KernelPCA(n_components=2, kernel='rbf')  # Используем радиальное базисное ядро (RBF kernel)

    # Применяем ядерный PCA к данным
    X_kpca = kpca.fit_transform(X)
    print(X_kpca)

    # Визуализируем результаты
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Kernel PCA')
    plt.show()


if __name__ == '__main__':
    # cov_matrix()
    # sco()
    l1()