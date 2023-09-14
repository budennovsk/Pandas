import pandas as pd


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

def gg():
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Создаем массив данных размером (10, 100)
    data = np.random.rand(10, 100)

    # Создаем экземпляр класса TSNE с указанием perplexity
    perplexity = min(30, data.shape[0] - 1)  # Примерное правило выбора perplexity

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)

    # Обучаем модель t-SNE на данных
    X_tsne = tsne.fit_transform(data)

    # Выводим результаты
    print(X_tsne)

def matlibsabon():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Создание случайных данных
    np.random.seed(0)
    data = np.random.randn(10, 100)  # 10 векторов данных с 100 элементами каждый

    # Создание графика
    sns.set()  # Настройка стиля seaborn по умолчанию
    plt.figure(figsize=(8, 6))  # Создание фигуры графика

    # Построение линий для каждого вектора данных
    for i in range(len(data)):
        plt.plot(data[i], label=f'Вектор {i + 1}')

    plt.xlabel('Индекс')  # Название оси x
    plt.ylabel('Значение')  # Название оси y
    plt.title('Визуализация данных для 10 векторов')  # Заголовок графика
    plt.legend()  # Отображение легенды

    plt.show()  # Отображение графика

def FastICA():

    from sklearn.decomposition import FastICA
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Создание случайных данных
    np.random.seed(0)
    X = np.random.rand(100, 5)  # Матрица данных размером 100x5
    y = np.random.rand(100, 1)

    # Создание экземпляра FastICA
    ica = FastICA(n_components=2)

    # Применение FastICA к данным
    transformed_data = ica.fit_transform(X)

    # Транспонирование векторного пространства
    transposed_data = ica.inverse_transform(transformed_data)

    # Вывод результатов
    print("Исходные данные:")
    print(pd.DataFrame(X))

    print("Транспонированные данные:")
    print(pd.DataFrame(transposed_data))

    tsvd2D_df = pd.DataFrame(data=transformed_data, columns=['x', 'y'])
    tsvd2D_df['cluster'] = y

    sns.scatterplot(x='x', y='y', hue='cluster', data=tsvd2D_df)
    plt.title("TruncatedSVD")
    plt.show()


if __name__ == '__main__':
    # cov_matrix()
    # sco()
    # l1()
    # gg()
    # matlibsabon()
    FastICA()