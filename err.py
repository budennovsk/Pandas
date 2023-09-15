import numpy
import pandas as pd
from scipy.optimize import dual_annealing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def optimize():
    def objective(x):
        return -(x[0] ** 2 + x[1] ** 2)

    x0 = [2, 2]
    bounds = [(-2, 2), (-2, 2)]

    result1 = dual_annealing(objective, bounds, x0=x0, seed=1237)
    result2 = dual_annealing(objective, bounds, x0=x0, seed=42)

    print("Минимум функции с seed=1237:", result1.fun)
    print("Оптимальные значения переменных с seed=1237:", result1.x)

    print("Минимум функции с seed=42:", -result2.fun)
    print("Оптимальные значения переменных с seed=42:", result2.x)


import numpy as np
from scipy.linalg import svd


def pca_scipy():
    # Example dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Perform PCA
    U, s, Vt = svd(X)
    print('U', U)
    print('s', s)
    print('Vt', Vt)

    # Calculate the explained variance ratio
    explained_variance_ratio = (s ** 2) / np.sum(s ** 2)

    # Print the principal components and explained variance ratio
    print("Principal Components:")
    print(Vt)
    print("Explained Variance Ratio:")
    print(explained_variance_ratio)


def pca_sklearn():
    from sklearn.decomposition import PCA
    import numpy as np

    # Пример набора данных
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Создание объекта PCA с числом компонентов = 2
    pca = PCA(n_components=1, random_state=42)
    print(type(pca))
    print(pca)
    v = pca.fit(X)

    # Применение PCA к набору данных
    X_pca = pca.fit_transform(X)
    ccc = pca.inverse_transform(X_pca)

    # Вывод преобразованных данных
    print("Преобразованные данные:")
    print(X_pca)
    print("Преобразованные данные reverse:")
    print(ccc)
    print('Возвращает дисперсию, объясняемую каждой главной компонентой.')
    print(pca.explained_variance_ratio_)
    print(' Возвращает относительную объясненную дисперсию каждой главной компоненты.')
    print(pca.explained_variance_ratio_)
    print(' Возвращает сингулярные значения, связанные с главными компонентами.')
    print(pca.singular_values_)
    print('Возвращает главные компоненты, которые представляют собой комбинацию исходных признаков.')
    print(pca.components_)
    print('Количество исходных признаков:')
    print(pca.n_features_)
    # Получение оценки ковариационной матрицы
    covariance_matrix = pca.get_covariance()
    # Вычисление среднеквадратической ошибки
    print("Оценка ковариационной матрицы:")
    print(covariance_matrix)


def score_pca():
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np

    # Создание DataFrame с данными
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    print(df)
    # Создание объекта PCA
    pca = PCA(n_components=3)

    # Обучение модели PCA на данных
    vc = pca.fit_transform(df)
    print(pca.components_)
    # Получение имен признаков
    feature_names_out = pca.get_feature_names_out(input_features=['feature1', 'feature2', 'feature3'])
    print(pca.get_params(deep=True))
    print('d!!!!!!!!')
    pca.get_feature_names_out(input_features=['feature1', 'feature2', 'feature3'])
    pca.set_output(transform='pandas')
    fff = pca.transform(df)
    print(fff, 'kkk')
    # Вывод имен признаков
    print("Имена признаков:")
    print(feature_names_out)
    print(vc)


def fds():
    import pandas as pd

    # Создание DataFrame
    data = {'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['apple', 'banana', 'cherry', 'date', 'elderberry']}
    df = pd.DataFrame(data)

    # Вывод сводной статистической информации только для числовых столбцов
    print(df.describe(include=[int, float]))

    # Вывод сводной статистической информации без столбцов типа 'object'
    print(df.describe(exclude='object'))

    # Вывод сводной статистической информации с дополнительными процентилями
    print(df.describe(percentiles=[0.1, 0.9]))


def pf():
    import pandas as pd
    import numpy as np

    # Создание массива NumPy с 100 строками
    arr = np.random.rand(100, 1)
    print(arr)
    # Преобразование массива в DataFrame
    df = pd.DataFrame(arr.reshape(1, 100), columns=[f'Column {i + 1}' for i in range(100)])

    print(df)


def npt():
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Сгенерировать данные для примера
    # X = np.random.rand(100, 4)  # Матрица данных размером 100x3

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Центрирование данных
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    print('центрированные данные:')
    print(X_centered)
    # Вычисление ковариационной матрицы
    cov_matrix = np.cov(X_centered.T)
    print('Ковариационная матрица:')
    print(cov_matrix)

    # Вычисление собственных значений и собственных векторов
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    print('eigenvalues:')
    print(eigenvalues)
    print('eigenvectors:')
    print(eigenvectors)

    # Сортировка собственных значений в убывающем порядке
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    print('Cобственные значения:')
    print(sorted_eigenvectors)

    # Выбор компонент исходя из объясненной дисперсии
    explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Выбор количества компонент, объясняющих более 90% дисперсии
    n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    # Проецирование данных на выбранные компоненты
    X_projected = np.dot(X_centered, sorted_eigenvectors[:, :n_components])

    # Вывод результатов
    print("Исходные данные:")
    print(X)
    print("\nПроецированные данные:")
    print(X_projected)
    print("\nВыбранные компоненты:")
    print(sorted_eigenvectors[:, :n_components])
    print('kkk')
    print(X_projected)
    # Обратное преобразование
    X_reconstructed = np.dot(X_projected[0], sorted_eigenvectors[:, :n_components].T) + X_mean

    # Вывод результатов
    print("\nВосстановленные данные:")
    print(X_reconstructed)


def cov_matrix():
    import numpy as np

    # Набор данных: количество часов учебы и оценки по математике
    hours_studied = np.array([4, 6, 3, 5, 2, 7])
    math_scores = np.array([80, 85, 70, 75, 60, 90])

    # Вычисление ковариационной матрицы
    cov_matrix = np.cov(hours_studied, math_scores)

    # Вывод ковариационной матрицы
    print(cov_matrix)


def res_R():
    import numpy as np

    # Исходная матрица
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # Вычисление ковариационной матрицы
    cov_matrix = np.cov(A)

    # Вывод ковариационной матрицы
    print(cov_matrix)
    # Вычисление SVD
    U, S, V = np.linalg.svd(A)

    # Вывод результатов
    print("Матрица U:")
    print(U)
    print("Сингулярные значения S:")
    print(S)
    print("Транспонированная матрица V:")
    print(V)


def en():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Загрузка набора данных Iris
    data = load_iris()
    X = data.data
    y = data.target

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Прогнозирование классов для тестового набора данных
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print("Точность модели: ", accuracy)

    mse = mean_squared_error(y_pred, y_test)
    print(f'Среднеквадратичная ошибка результата между входными данными и обратными данными: {mse}')


def gf():
    # Аппроксимация данных ядерным сглаживанием

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0, 10, 0.1)  # отсчеты для исходного сигнала
    x_est = np.arange(0, 10, 0.01)  # отсчеты, где производится восстановление функции
    N = len(x)
    y_sin = np.sin(x)
    y = y_sin + np.random.normal(0, 0.5, N)

    # аппроксимация ядерным сглаживанием
    h = 1.0  # при окне меньше 0.1 для финитных ядер будут ошибки

    K = lambda r: np.exp(-2 * r * r)  # гауссовское ядро
    # K = lambda r: np.abs(1 - r) * bool(r <= 1)        # треугольное ядро
    # K = lambda r: bool(r <= 1)        # прямоугольное ядро

    ro = lambda xx, xi: np.abs(xx - xi)  # метрика
    w = lambda xx, xi: K(ro(xx, xi) / h)  # веса

    plt.figure(figsize=(7, 7))
    plot_number = 0

    for h in [0.1, 0.3, 1, 10]:
        y_est = []
        for xx in x_est:
            ww = np.array([w(xx, xi) for xi in x])
            yy = np.dot(ww, y) / sum(ww)  # формула Надарая-Ватсона
            y_est.append(yy)

        plot_number += 1
        plt.subplot(2, 2, plot_number)

        plt.scatter(x, y, color='black', s=10)
        plt.plot(x, y_sin, color='blue')
        plt.plot(x_est, y_est, color='red')
        plt.title(f"Гауссовское ядро с h = {h}")
        plt.grid()

    plt.show()


def g6():
    from catboost import CatBoostRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Генерация случайных данных для примера
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = CatBoostRegressor(iterations=1000, depth=3, learning_rate=0.15)
    model.fit(X_train, y_train)
    d = model.predict(X_test)

    # Оценка производительности модели на тестовых данных с использованием метода score()
    score = model.score(X_test, y_test)

    # Вывод значения метрики
    print(f"Значение метрики: {score}")

    mse = mean_squared_error(y_test, d)
    print(f'Среднеквадратичная ошибка результата между y_true и y_predict: {mse}')


def qa():
    from catboost import Pool, cv

    cv_data = [["France", 1924, 44],
               ["USA", 1932, 37],
               ["Switzerland", 1928, 25],
               ["Norway", 1952, 30],
               ["Japan", 1972, 35],
               ["Mexico", 1968, 112]]

    labels = [1, 1, 0, 0, 0, 1]

    cat_features = [0]

    cv_dataset = Pool(data=cv_data,
                      label=labels,
                      cat_features=cat_features)

    params = {"iterations": 100,
              "depth": 2,
              "loss_function": "Logloss",
              "verbose": False}

    scores = cv(cv_dataset,
                params,
                fold_count=2,
                plot="True")


def das():
    from sklearn.decomposition import PCA
    import numpy as np

    # Создание примера набора данных
    X1 = np.array([[802.408453, 881.720452, 985.027474, 1009.583972, 1490.590736, 1418.246371, 1400.779943, 2090.043939,
                    2073.109809, 2099.947995, 0.75244, 4.233884, 0.564948, 0.531403, 1.586234, 7.916954, 16.3143,
                    -6.5567, -1.31823, 2.763446, 6.979393, 16.78894, -0.011786, 10.55865, -0.334712, 25.670812,
                    1.092996,
                    124.662071, 7.928688, 8.54063, -0.13163, -0.06626, -0.564832, 0.19986, -1.581049, 2.142479, 13.6388,
                    -1.39587, -0.53886, -0.837903, 31.084691, 24.389145, 34.560635, 119.431984, 86.207378, 94.612593,
                    17.690793, 186.165464, 107.592844, 41.857742, 10.054779, 18.212009, 6.914855, 18.062345, 5.019213,
                    5.470457, 14.06261, -7.811555, 2.342685, 5.856844, 978.8, 975.1, 975.6, 975.9, 968.4, 509.4, 381.1,
                    583.5, 645.1, 324.2, 0.00129, 0.1113, -0.100763, 0.073148, 0.105472, 7.80352, -5.884845, 22.751856,
                    4.956022, 2.807563, 23.5, 6.8, 12.0, 7.1, 6.5, 5.1, 4.2, 1.8, 1.5, 2.4, 0.57006, -1.233333, 0.75974,
                    2.23809, -6.42381, 6.0, 39.166667, 6.766667, 0.0, 0.1]])
    X2 = np.array([[0] * 100])
    X3 = np.array([[0] * 100])
    X4 = np.array([[0] * 100])
    X = np.concatenate((X1, X2, X3, X4))
    print(X.shape)

    # Создание экземпляра объекта PCA с whiten=True
    pca = PCA(n_components=2, whiten=True)

    # Применение модели PCA к данным
    pca.fit(X)

    # Получение преобразованных данных
    X_transformed = pca.transform(X)
    ff = pca.inverse_transform(X_transformed)
    print(pca.score(X), 'ff')
    print(mean_squared_error(X, ff))
    print(X_transformed, 'dd')

    print(pca.explained_variance_ratio_)
    '''[[ 4025.12421195]
 [-1341.70807065]
 [-1341.70807065]
 [-1341.70807065]]'''
    '''[1.0000000e+00 9.9298067e-32 1.0808314e-62]'''


def asa():
    import numpy as np

    def pca(X, num_components):
        # Центрирование данных
        X_centered = X - np.mean(X, axis=0)

        # Расчет ковариационной матрицы
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Расчет собственных значений и векторов
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Сортировка собственных значений в убывающем порядке
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Выбор указанного количества главных компонент
        selected_eigenvectors = sorted_eigenvectors[:, :num_components]

        # Проекция данных на главные компоненты
        X_transformed = np.dot(X_centered, selected_eigenvectors)

        return X_transformed, selected_eigenvectors

    # Создание примера набора данных
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]).reshape(3, 3)

    # Вызов функции PCA
    X_transformed, selected_eigenvectors = pca(X, num_components=2)

    print("Преобразованные данные:")
    print(X_transformed)

    print("Выбранные главные компоненты:")
    print(selected_eigenvectors)


def ff():
    import numpy as np

    # Задание набора данных
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    data1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mean_1 = np.mean(data1, axis=0)
    mean_1[0] = 7
    print(mean_1)
    data_2 = np.array([2, 5, 8])
    mean_2 = np.mean(data_2, axis=0)

    # Центрирование данных
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    centered_data_1 = data_2 - mean_2
    print('centered_data\n', centered_data)
    print('centered_data_1\n', centered_data_1)

    # Вычисление ковариационной матрицы
    covariance_matrix = np.cov(centered_data, rowvar=False)
    print('covariance_matrix', covariance_matrix)

    # Вычисление собственных векторов и собственных значений
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print('eigenvalues\n', eigenvalues)
    print('____')
    print('eigenvectors]\n', eigenvectors)
    # Сортировка собственных векторов по убыванию собственных значений
    sorted_indices = np.argsort(eigenvalues)[::-1]
    print('sorted_indices\n', sorted_indices)
    eigenvalues_sorted = eigenvalues[sorted_indices]
    print('eigenvalues_sorted\n', eigenvalues_sorted)
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    print('eigenvectors_sorted\n', eigenvectors_sorted)

    # Выбор нужного количества главных компонент
    k = 3
    selected_eigenvectors = eigenvectors_sorted[:, :k]
    print('selected_eigenvectors\n', selected_eigenvectors)

    # Проекция данных на главные компоненты
    X_pca = np.dot(centered_data, selected_eigenvectors)
    X_pca_1 = np.dot(centered_data_1, selected_eigenvectors)
    print('X_pca\n', X_pca)

    print('X_pca_1\n', X_pca_1)

    # Обратное преобразование к исходным данным
    reconstructed_data = np.dot(X_pca, selected_eigenvectors.T) + mean

    print('reconstructed_data\n', reconstructed_data)

    # Обратное преобразование к исходным данным
    reconstructed_data_1 = np.dot(X_pca_1, eigenvectors.T) + mean_2

    print('reconstructed_data_1\n', reconstructed_data_1)


def lox():
    import numpy as np

    # Генерируем случайные данные (пример)
    np.random.seed(0)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 100 наблюдений с 3 признаками

    # Центрируем данные
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Вычисляем матрицу ковариации
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Вычисляем собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Сортируем собственные векторы по убыванию собственных значений
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Выбираем первые k главных компонент (например, первые две)
    k = 2
    top_k_eigenvectors = eigenvectors[:, :k]

    # Преобразуем данные в новое пространство главных компонент
    transformed_data = np.dot(centered_data, top_k_eigenvectors)
    print(transformed_data)
    # Восстанавливаем исходные данные из нового пространства главных компонент
    reconstructed_data = np.dot(transformed_data, top_k_eigenvectors.T) + mean
    print("__________")
    print(reconstructed_data)

    # reconstructed_data теперь содержит восстановленные исходные данные


def max_10():
    import pandas as pd
    import numpy as np

    # Создание исходного DataFrame
    data = np.random.randint(0, 100, size=(10, 100))
    df = pd.DataFrame(data)
    print(df)

    # Нахождение максимума каждого 10 столбца
    max_values = df.iloc[:, 0::10].max()

    # Применение среднего значения ко всем остальным столбцам
    other_columns = df.columns.difference(df.columns[0::10])
    mean_values = df[other_columns].mean()
    print(mean_values)

    # Замена значений в исходном DataFrame
    df.loc[:, 0::10] = max_values
    print(df)
    print('___')
    df[other_columns] = mean_values.values

    print(df)
    print(df.head(1))
    '''other_columns = df.columns.difference(df.columns[9::10])
mean_values = df[other_columns].mean()

# Применяем максимумы и средние значения к исходному DataFrame
df.iloc[:, 9::10] = max_values
df[other_columns] = mean_values'''


def tr(data_max, data_max_mean, data):
    import numpy as np

    # Ваша матрица данных
    # data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # data_max = np.array([2,5,8])
    mean_max = np.mean(data_max.to_numpy(), axis=0)

    # data_max_mean = data_max - mean_max

    # Вычислить среднее значение по столбцам
    mean = np.mean(data.to_numpy(), axis=0)

    # Отцентровать данные
    centered_data = data - mean

    # Вычислить ковариационную матрицу
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Вычислить собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Отсортировать собственные значения в убывающем порядке
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Количество главных компонент, которые вы хотите оставить
    num_components = 7

    # Выбрать только первые num_components главных компонент
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Проецировать данные на главные компоненты
    projected_data = np.dot(centered_data, selected_eigenvectors)

    # Восстановить исходные данные
    reconstructed_data = np.dot(projected_data, selected_eigenvectors.T) + mean
    print('sasi')

    # Вывести восстановленные данные

    print(pd.DataFrame(reconstructed_data))
    print(pd.DataFrame(data))
    print("__w__")

    print('___')

    res = data_max_mean.dot(selected_eigenvectors)

    reconstructed_data_max = np.dot(res, selected_eigenvectors.T) + mean_max
    print('data_max')
    print(pd.DataFrame(data_max))
    print('____S___')

    print(pd.DataFrame(np.real(reconstructed_data_max)), 'otvet')
    '''[ 1. -1.  0.] mean'''
    '''[7,5,6]'''
    '''[4.  4.5 6.5] otvet mean'''
    '''[10.  10.5 12.5] otvet'''
    '''[5.  3.5 6.5] otvet'''
    '''0  991.336039+0.000000j  785.247283+0.000000j  1119.463357+0.000000j  303.327609+0.000000j  1806.598104+0.000000j  2161.843749+0.000000j  1866.465453+0.000000j  2726.891065+0.000000j  2310.459409+0.000000j  2175.700146+0.000000j  888.068747+0.000000j  888.686272+0.000000j  849.19417+0.00000j  857.392954+0.000000j  802.929352+0.000000j  851.747562+0.000000j  920.87706+0.00000j  842.796146+0.000000j  909.065941+0.000000j  868.326855+0.000000j  799.356194+0.000000j  684.068734+0.000000j  834.571343+0.000000j  790.732522+0.000000j  929.079434+0.000000j  1007.807398+0.000000j  974.156462+0.000000j  880.082899+0.000000j  1009.999558+0.000000j  877.18804+0.00000j  885.398387+0.000000j  899.912609+0.000000j  860.656947+0.000000j  885.52335+0.00000j  857.820582+0.000000j  869.4799+0.0000j  863.506438+0.000000j  879.384425+0.000000j  871.513401+0.000000j  865.777013+0.000000j  666.337505+0.000000j  853.418607+0.000000j  1255.065333+0.000000j  1263.97856+0.00000j  780.73027+0.00000j  496.922902+0.000000j  724.919812+0.000000j  634.437687+0.000000j  812.004207+0.000000j  1055.414133+0.000000j  863.325941+0.000000j  891.178223+0.000000j  865.294593+0.000000j  874.564749+0.000000j  840.875471+0.000000j  846.668268+0.000000j  907.494457+0.000000j  686.515727+0.000000j  872.919029+0.000000j  900.991326+0.000000j  872.224553+0.000000j  869.82743+0.00000j  863.367741+0.000000j  879.024501+0.000000j  861.742585+0.000000j  1054.595618+0.000000j  1498.897475+0.000000j  909.627137+0.000000j  840.645711+0.000000j  787.620457+0.000000j  878.506893+0.000000j  878.10778+0.00000j  877.306537+0.000000j  878.310077+0.000000j  876.644348+0.000000j  922.205679+0.000000j  871.18235+0.00000j  969.906147+0.000000j  902.210162+0.000000j  896.729948+0.000000j  880.859067+0.000000j  875.430855+0.000000j  890.381599+0.000000j  872.590607+0.000000j  875.742807+0.000000j  880.214977+0.000000j  878.81934+0.00000j  880.049315+0.000000j  880.187049+0.000000j  880.152561+0.000000j  875.119787+0.000000j  878.256471+0.000000j  907.38844+0.00000j  852.514852+0.000000j  840.141299+0.000000j  865.524085+0.000000j  740.203342+0.000000j  857.643246+0.000000j  880.824667+0.000000j  880.425001+0.000000j otvet
'''
    '''0  991.336039+0.000000j  786.143069+0.000000j  1223.666165+0.000000j  432.086914+0.000000j  2416.364173+0.000000j  2699.265454+0.000000j  2386.42073+0.00000j  3936.110337+0.000000j  3502.744551+0.000000j  3394.823475+0.000000j  23.894565+0.000000j  12.095489+0.000000j -31.065549+0.000000j -22.900309+0.000000j -76.309081+0.000000j -21.160151+0.000000j  56.366703+0.000000j -44.585281+0.000000j  26.923045+0.000000j -9.734366+0.000000j -37.035367+0.000000j -179.966987+0.000000j -46.26511+0.00000j -79.533489+0.000000j  47.920055+0.000000j  152.653543+0.000000j  94.424791+0.000000j  123.920303+0.000000j  137.103579+0.000000j  4.904005+0.000000j  13.955442+0.000000j  19.021677+0.000000j -20.732551+0.000000j  4.898548+0.000000j -24.585133+0.000000j -9.202288+0.000000j -3.679348+0.000000j -2.836118+0.000000j -9.850134+0.000000j -15.885557+0.000000j  39.650679+0.000000j -3.016915+0.000000j  408.801301+0.000000j  502.585877+0.000000j -13.887018+0.000000j -289.289172+0.000000j -138.214062+0.000000j -60.221516+0.000000j  38.772384+0.000000j  216.447208+0.000000j  72.061019+0.000000j  28.565565+0.000000j -8.615219+0.000000j  11.802428+0.000000j -34.929983+0.000000j -28.685941+0.000000j  40.732408+0.000000j -202.120494+0.000000j -5.562953+0.000000j  26.023503+0.000000j  981.399886+0.000000j  964.102764+0.000000j  958.143075+0.000000j  974.099834+0.000000j  949.317918+0.000000j  683.170952+0.000000j  999.172809+0.000000j  612.30247+0.00000j  604.921044+0.000000j  230.995791+0.000000j -1.400342+0.000000j -2.605497+0.000000j -3.618893+0.000000j -2.441442+0.000000j -4.074847+0.000000j  49.184536+0.000000j -15.527162+0.000000j  111.833337+0.000000j  26.341517+0.000000j  18.712845+0.000000j  24.0344+0.0000j  1.406188+0.000000j  21.556932+0.000000j -1.13406+0.00000j  1.41814+0.00000j  4.49031+0.00000j  2.194673+0.000000j  1.024648+0.000000j  0.862383+0.000000j  1.727895+0.000000j  3.386029+0.000000j -3.801529+0.000000j  27.323513+0.000000j -26.07172+0.00000j -47.107177+0.000000j -9.300582+0.000000j -101.454658+0.000000j -16.414754+0.000000j  0.0+0.0j -0.299666+0.000000j otvet
'''


res = pd.read_excel('ML_manufacture_second_all_10_HYPER_Батончики_Супермаркет.xlsx')


def er():
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])

    random_values = np.random.uniform(1.01, 1.10, size=90)


    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values

    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)

    # Ваша матрица данных
    data = X.to_numpy()

    data1 = X_10.to_numpy()

    # Вычислить среднее значение по столбцам
    mean = np.mean(data, axis=0)

    # # Вычислить среднее значение и стандартное отклонение
    # mean = np.mean(data1[0], axis=0)
    # print('mean',pd.DataFrame(mean).T)
    # std = np.std(data1[0], axis=0)
    # print('std',pd.DataFrame(std).T)

    # # Шкалирование данных
    # scaled_data = (data - mean) / std
    # scaled_data= np.nan_to_num(scaled_data,nan=0)

    # # Шкалирование данных
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(data)
    # print(scaled_data)

    # Отцентровать данные
    centered_data = data - mean
    print(centered_data)
    centered_data_1 = data1 - mean
    print('ssssssssssssssss')
    print(centered_data_1)

    # Вычислить ковариационную матрицу
    covariance_matrix = np.cov(centered_data, rowvar=False, bias=True)
    print(covariance_matrix.shape)
    print(centered_data.shape)

    # Вычислить собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Отсортировать собственные значения в убывающем порядке
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Количество главных компонент, которые вы хотите оставить
    num_components = 10

    # Выбрать только первые num_components главных компонент
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    print('w1')
    # Проецировать данные на главные компоненты
    projected_data = np.dot(centered_data, selected_eigenvectors)
    print('PSA')
    print(pd.DataFrame(np.real(projected_data)))
    projected_data_1 = np.dot(centered_data_1, selected_eigenvectors)
    c = np.array(list(map(lambda x: x.real, projected_data_1[0]))).reshape(1, -1)
    print(pd.DataFrame(c))

    print('w2')
    # Восстановить исходные данные
    reconstructed_data = np.dot(projected_data, selected_eigenvectors.T) + mean
    reconstructed_data_1 = np.dot(projected_data_1, selected_eigenvectors.T) + mean

    # Вывести восстановленные данные
    print("входные данные для обучения модели")
    print(X)

    print("данные после обучения модели")
    print(pd.DataFrame(np.real(reconstructed_data)))
    print("___")
    print("входные данные максимумов")
    print(X_10)
    print('выходные данные максимумов')
    # print(X_1)

    print(pd.DataFrame(np.array(list(map(lambda x: x.real, reconstructed_data_1[0]))).reshape(1, -1)))

    mse = mean_squared_error(X, pd.DataFrame(np.real(reconstructed_data)))
    print(round(mse, 30))
    print(mean_absolute_error(X, pd.DataFrame(np.real(reconstructed_data))))
def pca_1():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import Lasso, Ridge

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
     # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.10, size=90)
    print(random_values)
    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)
    y = res['Percentage_Sales_rub'].to_numpy()

    X_10_max = pd.DataFrame(res[res.columns[6:]].max().values.reshape(1, -1))

    # Ваш массив данных
    data = X

    # Шкалирование данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data_1 = scaler.transform(X_10)
    # scaled_data_1 = np.repeat(X_10.to_numpy(), 5,axis=0)





    # Применение метода главных компонент
    pca = PCA(n_components=8, random_state=42, svd_solver='full', whiten=True)

    projected_data = pca.fit_transform(data)
    projected_data_1 = pca.transform(X_10)
    print('covariance')
    print(projected_data_1)


    # Возвращение к исходным значениям
    # reconstructed_data = scaler.inverse_transform(pca.inverse_transform(projected_data))
    # reconstructed_data_1 = scaler.inverse_transform(pca.inverse_transform(projected_data_1))

    reconstructed_data = pca.inverse_transform(projected_data)
    reconstructed_data_1 = pca.inverse_transform(projected_data_1)

    print(X)
    # Вывод восстановленных данных
    print(pd.DataFrame(reconstructed_data))
    print('___')

    print(X_10)
    # print(pd.DataFrame(scaled_data_1))
    print(pd.DataFrame(reconstructed_data_1))
    mse = mean_squared_error(X.values.tolist(), pd.DataFrame(reconstructed_data).values.tolist())
    print(round(mse,30))
    print(mean_absolute_error(X, pd.DataFrame(reconstructed_data)))
    print('___')
    print(mean_absolute_error(X_10, pd.DataFrame(reconstructed_data_1)))
    print('___')
    print(X_10.iloc[:, 0::10])
    print(pd.DataFrame(reconstructed_data_1[0][0::10].reshape(1,-1)))

    print('___')
    print(mean_absolute_error(X_10.iloc[:,0::10], pd.DataFrame(reconstructed_data_1[0][0::10].reshape(1,-1))))

    # # Создание графика
    # sns.set()  # Настройка стиля seaborn по умолчанию
    # plt.figure(figsize=(8, 6))  # Создание фигуры графика
    #
    # # Построение линий для каждого вектора данных
    # for i in range(len(projected_data)):
    #     plt.plot(projected_data[i], label=f'Вектор {i + 1}')
    #
    # plt.xlabel('Индекс')  # Название оси x
    # plt.ylabel('Значение')  # Название оси y
    # plt.title('Визуализация данных для 10 векторов')  # Заголовок графика
    # plt.legend()  # Отображение легенды
    #
    # plt.show()  # Отображение графика


    # # Визуализация данных в двумерном пространстве
    # plt.scatter(projected_data[:, 0], projected_data[:, 1], c=y)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title('t-SNE Visualization of Iris Dataset')
    # plt.show()

    # tsvd2D_df = pd.DataFrame(data=projected_data, columns=['x', 'y'])
    # tsvd2D_df['cluster'] = y
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=tsvd2D_df)
    # plt.title("TruncatedSVD")
    # plt.show()

def ica():
    import numpy as np
    from sklearn.decomposition import KernelPCA
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    import seaborn as sns


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.10, size=90)

    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)

    y = res['Percentage_Sales_rub'].to_numpy()

    # Ваша матрица данных
    data = X.to_numpy()

    data1 = X_10.to_numpy()



    # data = res.iloc[1:, 6:].T
    # print(data.shape)
    # data1 = X_10.T
    # print(data1.shape)
    # y = res.iloc[0, 6:].values.reshape(100, 1)
    # print(y.shape)

    # Создаем объект ядерного PCA
    kpca = KernelPCA(n_components=8, kernel='rbf',alpha=0.001, fit_inverse_transform=True, random_state=42)  # Используем радиальное базисное ядро (RBF kernel)

    # Применяем ядерный PCA к данным
    X_kpca = kpca.fit_transform(data)

    X_kpca1 = kpca.transform(data1)

    '''3.667213678658044, 12.276249004991566,  44.843923992707545  kernel='rbf',alpha=0.001 '''
    '''79.70483475026967 , 33.92121755253269 62.11173683433403'''
    '''25.596688764236628 , 13.139620554771419, 45.015154392326956'''
    # Выполняем обратное преобразование для восстановления исходных значений
    X_inverse = kpca.inverse_transform(X_kpca)
    X_inverse1 = kpca.inverse_transform(X_kpca1)

    print('ICA')
    print(pd.DataFrame(X_kpca))
    print('X_inverse')
    print(pd.DataFrame(data))
    print('____')
    print(pd.DataFrame(X_inverse))

    print('ICA')
    print(pd.DataFrame(X_kpca1))
    print('X_inverse')
    print(pd.DataFrame(data1))
    print('____')
    print(pd.DataFrame(X_inverse1))

    mse = mean_squared_error(pd.DataFrame(data), pd.DataFrame(X_inverse))
    print(round(mse, 30))
    print(mean_absolute_error(pd.DataFrame(data),pd.DataFrame(X_inverse)))
    print('___')
    print(mean_absolute_error(pd.DataFrame(data1),pd.DataFrame(X_inverse1)))
    print('___')
    print(pd.DataFrame(data1).iloc[:, 0::10])
    print(pd.DataFrame(X_inverse1[0][0::10].reshape(1, -1)))

    print('___')
    print(mean_absolute_error(pd.DataFrame(data1).iloc[:, 0::10], pd.DataFrame(X_inverse1[0][0::10].reshape(1, -1))))


    '''167.55654103704993 ,  109.42664794907515,  117.58876684444292'''
    # # Визуализация данных в двумерном пространстве
    # plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title('t-SNE Visualization of Iris Dataset')
    # plt.show()

    # # Создание графика
    # sns.set()  # Настройка стиля seaborn по умолчанию
    # plt.figure(figsize=(8, 6))  # Создание фигуры графика
    #
    # # Построение линий для каждого вектора данных
    # for i in range(len(X_kpca)):
    #     plt.plot(X_kpca[i], label=f'Вектор {i + 1}')
    #
    # plt.xlabel('Индекс')  # Название оси x
    # plt.ylabel('Значение')  # Название оси y
    # plt.title('Визуализация данных для 10 векторов')  # Заголовок графика
    # plt.legend()  # Отображение легенды
    #
    # plt.show()  # Отображение графика

    # tsvd2D_df = pd.DataFrame(data=X_kpca, columns=['x', 'y'])
    # tsvd2D_df['cluster'] = y
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=tsvd2D_df)
    # plt.title("TruncatedSVD")
    # plt.show()

    # Визуализация главных компонент
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Визуализация главных компонент')
    plt.show()

def tsne():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.manifold import TSNE
    import seaborn as sns

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.10, size=90)

    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)

    # # Ваша матрица данных
    # data = X.to_numpy()
    #
    # data1 = X_10.to_numpy()


    data = res.iloc[1:, 6:].T

    y = res.iloc[0, 6:].values.reshape(100, 1)

    # Загружаем набор данных Iris
    # data = np.random.rand(10, 100)
    # print(data.shape)
    # X = X.to_numpy()
    # print(X.shape)

    # y = res['Percentage_Sales_rub'].to_numpy()

    # Шкалирование данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)


    # Применяем алгоритм t-SNE для снижения размерности данных до 2D

    perplexity = min(30, data.shape[0] - 1)  # Примерное правило выбора perplexity
    print(perplexity)
    tsne = TSNE(n_components=2,method='exact', perplexity=perplexity, random_state=42)
    print('q1')
    print(X.shape)
    X_tsne = tsne.fit_transform(scaled_data)

    # print(X_tsne)
    # print('q2')
    # tsvd2D_df = pd.DataFrame(data=X_tsne, columns=['x', 'y'])
    # tsvd2D_df['cluster'] = y
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=tsvd2D_df)
    # plt.title("TruncatedSVD")
    # plt.show()


    # # Визуализация данных в двумерном пространстве
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.title('t-SNE Visualization of Iris Dataset')
    # plt.show()

    # Визуализация главных компонент
    plt.scatter(X_tsne [:, 0], X_tsne [:, 1], c=y)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Визуализация главных компонент')
    plt.show()

def FastICA():
    from sklearn.decomposition import FastICA
    import numpy as np
    import matplotlib.pyplot as plt


    import seaborn as sns

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    y = res['Percentage_Sales_rub'].to_numpy()

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.10, size=90)

    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)

    # # Ваша матрица данных
    # data = X.to_numpy()
    #
    #
    # data1 = X_10.to_numpy()

    data = res.iloc[1:, 6:].T

    y = res.iloc[0, 6:].values.reshape(100, 1)


    ica2D = FastICA(n_components=2, random_state=42)
    ica_data2D = ica2D.fit_transform(data)
    # ica_data2D_1 = ica2D.transform(data1)

    revers = ica2D.inverse_transform(ica_data2D)
    # revers_1 = ica2D.inverse_transform(ica_data2D_1)


    print(X)

    print(pd.DataFrame(revers))

    mse = mean_squared_error(pd.DataFrame(data), pd.DataFrame(revers))
    print(round(mse, 30))
    print(mean_absolute_error(pd.DataFrame(data), pd.DataFrame(revers)))
    print('___')

    print(X_10)

    # print(pd.DataFrame(revers_1))

    # mse = mean_squared_error(X_10, pd.DataFrame(revers_1))
    # print(round(mse, 30))
    # print(mean_absolute_error(X_10, pd.DataFrame(revers_1)))
    print('___')


    # print(pd.DataFrame(data1).iloc[:, 0::10])
    # print(pd.DataFrame(revers_1[0][0::10].reshape(1, -1)))

    print('___')
    # print(mean_absolute_error(pd.DataFrame(data1).iloc[:, 0::10], pd.DataFrame(revers_1[0][0::10].reshape(1, -1))))

    '''7.127513507707408 14.551747135557182 47.49917585158504'''
    #
    # ica2D_df = pd.DataFrame(data=ica_data2D, columns=['x', 'y'])
    #
    # ica2D_df['cluster'] = y
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=ica2D_df)
    # plt.title("ICA")
    # # plt.show()
    #
    # # Создание графика
    # sns.set()  # Настройка стиля seaborn по умолчанию
    # plt.figure(figsize=(8, 6))  # Создание фигуры графика
    #
    # # Построение линий для каждого вектора данных
    # for i in range(len(ica_data2D)):
    #     plt.plot(ica_data2D[i], label=f'Вектор {i + 1}')
    #
    # plt.xlabel('Индекс')  # Название оси x
    # plt.ylabel('Значение')  # Название оси y
    # plt.title('Визуализация данных для 10 векторов')  # Заголовок графика
    # plt.legend()  # Отображение легенды
    #
    # plt.show()  # Отображение графика

    # Визуализация главных компонент
    plt.scatter(ica_data2D[:, 0], ica_data2D[:, 1], c=y)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Визуализация главных компонент')
    plt.show()

def MDS():


    from sklearn.manifold import MDS

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    y = res['Percentage_Sales_rub'].to_numpy()

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].min()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.10, size=90)

    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)
    #
    # # Ваша матрица данных
    # data = X.to_numpy()
    #
    # data1 = X_10.to_numpy()

    data = res.iloc[1:, 6:].T

    y = res.iloc[0, 6:].values.reshape(100, 1)

    # Шкалирование данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)


    mds2D = MDS(n_components=2, random_state=42)

    mds_data2D = mds2D.fit_transform(scaled_data)

    #
    # mds2D_df = pd.DataFrame(data=mds_data2D, columns=['x', 'y'])
    #
    # mds2D_df['cluster'] = y
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=mds2D_df)
    # plt.title("MDS")
    # plt.show()

    # Визуализация главных компонент
    plt.scatter(mds_data2D[:, 0],  mds_data2D[:, 1], c=y)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Визуализация главных компонент')
    plt.show()



def UPA():
    from umap.umap_ import UMAP
    from sklearn.preprocessing import StandardScaler
    import numba


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    X = res[res.columns[6:]]
    X_a = res.iloc[1, 6:]
    X_1 = pd.DataFrame(X_a.values.reshape(1, -1))

    y = res['Percentage_Sales_rub'].to_numpy()

    X_10 = res[res.columns[6:]].copy()
    X_max_10 = X_10.iloc[:, 0::10].max()
    other_columns = X_10.columns.difference(X_10.columns[0::10])
    # Генерация случайных чисел в диапазоне от 1 до 10
    random_values = np.random.uniform(1.01, 1.10, size=90)

    mean_values = X_10[other_columns].mean() * random_values

    X_10.loc[:, 0::10] = X_max_10.values
    X_10[other_columns] = mean_values.values
    X_10 = X_10.head(1)

    # # Ваша матрица данных
    # data = X.to_numpy()
    #
    # data1 = X_10.to_numpy()

    data = res.iloc[1:, 6:].T

    y = res.iloc[0, 6:].values.reshape(100, 1)

    # Шкалирование данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)


    umap2D = UMAP(n_components=2,random_state=42)
    umap_data2D = umap2D.fit_transform(scaled_data)
    # umap_data2D_1 = umap2D.transform(data1)


    revers = umap2D.inverse_transform(umap_data2D)
    # revers_1 = umap2D.inverse_transform(umap_data2D_1)

    print(X)

    print(pd.DataFrame(revers))

    mse = mean_squared_error(pd.DataFrame(data), pd.DataFrame(revers))
    print(round(mse, 30))
    print(mean_absolute_error(pd.DataFrame(data), pd.DataFrame(revers)))
    print('___')

    print(X_10)

    # print(pd.DataFrame(revers_1))

    # mse = mean_squared_error(X_10, pd.DataFrame(revers_1))
    # print(round(mse, 30))
    # print(mean_absolute_error(X_10, pd.DataFrame(revers_1)))
    # print('___')

    # print(pd.DataFrame(data1).iloc[:, 0::10])
    # print(pd.DataFrame(revers_1[0][0::10].reshape(1, -1)))

    print('___')
    # print(mean_absolute_error(pd.DataFrame(data1).iloc[:, 0::10], pd.DataFrame(revers_1[0][0::10].reshape(1, -1))))

    # umap2D_df = pd.DataFrame(data=umap_data2D, columns=['x', 'y'])
    #
    # umap2D_df['cluster'] = y
    #
    # sns.scatterplot(x='x', y='y', hue='cluster', data=umap2D_df)
    # plt.title("UMAP")
    # plt.show()

    # Визуализация главных компонент
    plt.scatter(umap_data2D[:, 0], umap_data2D[:, 1], c=y)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Визуализация главных компонент')
    plt.show()

def isomap():
    import numpy as np
    from sklearn.manifold import Isomap
    import matplotlib.pyplot as plt

    data = res.iloc[1:, 6:].T

    y = res.iloc[0, 6:].values.reshape(100, 1)

    # Шкалирование данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)


    # Создадим экземпляр класса Isomap с 2 компонентами
    isomap = Isomap(n_components=2)

    # Применим алгоритм Isomap к данным
    embedding = isomap.fit_transform(scaled_data)

    # Выведем результаты
    print(embedding)

    # Визуализация главных компонент
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y)
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title('Визуализация главных компонент')
    plt.show()

def lle():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import LocallyLinearEmbedding

    data = res.iloc[1:, 6:].T

    y = res.iloc[0, 6:].values.reshape(100, 1)

    # Создадим экземпляр класса LLE с 2 компонентами
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)

    # Применим метод LLE к данным
    transformed_data = lle.fit_transform(data)

    # Визуализация результата
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=y)
    plt.xlabel('Первая компонента')
    plt.ylabel('Вторая компонента')
    plt.title('Визуализация методом LLE')
    plt.show()


if __name__ == '__main__':
    # optimize()
    # pca_scipy()
    # pca_sklearn()
    # score_pca()
    # fds()
    # pf()
    # npt()
    # cov_matrix()
    # res_R()
    # en()
    # gf()
    # g6()
    # qa()
    # das()
    # asa()
    # ff()
    # lox()
    # max_10()
    # tr()
    # er()
    # pca_1()
    ica()
    # tsne()
    # FastICA()
    # MDS()
    # UPA()
    # isomap()
    # lle()