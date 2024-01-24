# # from pulp import LpMaximize, LpProblem, LpStatus, LpVariable
# #
# # # Создаем модель
# # model = LpProblem(name="small-problem", sense=LpMaximize)
# #
# # # Инициализируем переменные
# # x = LpVariable(name="x", lowBound=0) # Количество Продукта 1
# # y = LpVariable(name="y", lowBound=2) # Количество Продукта 2
# #
# # # Добавляем ограничения
# # model += (2 * x + 3 * y <= 20), "Машина 1"
# #
# # model += (4 * x + 1 * y <= 10, "Машина 2")
# #
# # # Добавляем целевую функцию
# # model += 3 * x + 5 * y
# #
# #
# # # Решаем задачу оптимизации
# # status = model.solve()
# # print('_____')
# #
# #
# # # Выводим результаты
# # print(f"status: {model.status}, {LpStatus[model.status]}")
# # print(f"Прибыль: {model.objective.value()}")
# # print(f"Продукт 1: {x.value()}")
# # print(f"Продукт 2: {y.value()}")
# # print('_____')
# # print(model)
#
# # from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
# #
# # # Создаем модель
# # model = LpProblem(name="mixed-integer-problem", sense=LpMaximize)
# #
# # # Инициализируем переменные
# # x = LpVariable(name="x", lowBound=0, cat="Integer")  # x - целочисленная переменная
# # y = LpVariable(name="y", lowBound=0)  # y - непрерывная переменная
# #
# # # Добавляем ограничения в модель
# # model += (x + 2 * y <= 20, "constraint_1")
# # model += (3 * x - y >= 0, "constraint_2")
# #
# # # Добавляем целевую функцию
# # model += lpSum([2*x, y])
# #
# # # Решаем задачу
# # status = model.solve()
# #
# # # Выводим результат
# # print(f"status: {model.status}, {LpStatus[model.status]}")
# # print(f"objective: {model.objective.value()}")
# # for var in model.variables():
# #     print(f"{var.name}: {var.value()}")
# # print('________')
# # print(model)
# #
# # from sklearn import tree
# # import pandas as pd
# #
# # # Данные: возраст машины и решение (1 - купить новую, 0 - оставить старую)
# # data = pd.DataFrame({
# #   'Age': [1, 2, 3, 4, 5, 6],
# #   'Decision': [0, 0, 0, 1, 1, 1]
# # })
# #
# # # Обучение дерева решений
# # clf = tree.DecisionTreeClassifier()
# # clf = clf.fit(data[['Age']], data['Decision'])
# #
# # # Предсказание решения для машины возрастом 3.5 года
# # print(clf.predict([[101]]))
#
#
# # from sklearn import tree
# # from pyscipopt import Model
# # import pandas as pd
# #
# # # Данные: возраст машины и решение (1 - купить новую, 0 - оставить старую)
# # data = pd.DataFrame({
# #   'Age': [1, 2, 3, 4, 5, 6],
# #   'Decision': [0, 0, 0, 1, 1, 1]
# # })
# #
# # # Обучение дерева решений
# # clf = tree.DecisionTreeClassifier()
# # clf = clf.fit(data[['Age']], data['Decision'])
# #
# # # Создаем модель
# # model = Model("Car Decision")
# #
# # # Определяем переменную - возраст машины
# # age = model.addVar("Age", vtype="I", lb=1, ub=20)  # ограничение: 1 <= age <= 20
# #
# # # Определяем целевую функцию - стоимость выбора машины, предсказанную деревом решений
# # cost = clf.predict([[age]])[0]  # 0 - оставить старую, 1 - купить новую
# #
# # model.setObjective(cost, "minimize")
# #
# # # Решаем модель
# # model.optimize()
# #
# # # Выводим оптимальное решение
# # print("Optimal age:", model.getVal(age))
# # print("Decision cost:", model.getObjVal())
#
# from sklearn import tree
# from pyscipopt import Model
# import pandas as pd
# import numpy as np
#
# # Данные: возраст машины и решение (1 - купить новую, 0 - оставить старую)
# data = pd.DataFrame({
#   'Age': [1, 2, 3, 4, 5, 6],
#   'Decision': [0, 0, 0, 1, 1, 1]
# })
#
# # Обучение дерева решений
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(data[['Age']], data['Decision'])
#
# # Создаем модель
# model = Model("Car Decision")
#
# # Определяем переменную - возраст машины
# age = model.addVar("Age", vtype="I", lb=1, ub=20)  # ограничение: 1 <= age <= 20
# print(type(age))
# # Получаем "правила" от дерева решений
# ages = np.arange(1, 21)  # возраста от 1 до 20
# decisions = clf.predict(ages.reshape(-1, 1))  # предсказания для каждого возраста
#
# # Добавляем "правила" в модель в виде ограничений
# for a, d in zip(ages, decisions):
#     if d == 0:  # если нужно купить новую машину
#
#         model.addCons(age >= a)  # возраст машины должен быть не меньше a
#
# # Определяем целевую функцию (в данном случае просто минимизируем возраст)
# model.setObjective(age, "minimize")
#
# # Решаем модель
# model.optimize()
#
# # Выводим оптимальное решение
# print("Optimal age:", model.getVal(age))


import numpy as np
import pandas as pd
import statsmodels.api as sm

# # Генерируем случайные данные
# np.random.seed(0)
# N = 100 # количество наблюдений
# K = 3 # количество признаков
# X = np.random.rand(N, K)
# # Генерируем метки классов
# y = np.random.choice([0, 1, 2], N)
#
# # Преобразуем в pandas DataFrame
# df = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(K)])
# df['target'] = y
#
# # Создаем матрицу признаков и целевую переменную
# X = df[['feature1', 'feature2', 'feature3']]
# y = df['target']
# print(df)
#
# # Добавляем константу для перехвата
# X = sm.add_constant(X)
# print(X)
#
# # Создаем MNL модель
# mnl_model = sm.MNLogit(y, X)
#
# print(mnl_model)
# # Обучаем модель
# result = mnl_model.fit()
#
# # Выводим результаты
# print(result.summary())
#
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs
#
# # создаем набор данных
# data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#
# # создаем объект KMeans
# kmeans = KMeans(n_clusters=4)
#
# # обучаем модель
# kmeans.fit(data)
#
# # предсказываем кластеры для наших данных
# predicted_clusters = kmeans.predict(data


# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.datasets import load_digits
#
# # Загрузка набора данных
# digits = load_digits()
# data = digits.data
# labels = digits.target
#
# # Применение t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# data_tsne = tsne.fit_transform(data)
#
# # Визуализация результатов
# plt.figure(figsize=(13,10))
# scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
#
# # Добавление легенды
# legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
# plt.add_artist(legend1)

# plt.show()


# from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder
#
# # Предположим, что df - это ваш исходный DataFrame
#
# # Фильтруем данные
# filtered_df = df[['Store','Date','Fuel_Price','Dept','Type']]
#
#
# # Кодируем категориальные переменные
# le = LabelEncoder()
# filtered_df['Type'] = le.fit_transform(filtered_df['Type'])
#
# filtered_df['Date'] = (filtered_df['Date'] - filtered_df['Date'].min()).dt.days
#
# # Выбираем колонки для кластеризации
# X = filtered_df[['Fuel_Price', 'Date']]
#
#
# # Создаем модель KMeans
# kmeans = KMeans(n_clusters=3)
#
# # Обучаем модель
# kmeans.fit(X)
#
# # Получаем метки кластеров
# labels = kmeans.labels_
#
# # Добавляем метки кластеров в DataFrame
# filtered_df.loc[:,'Cluster'] = labels
#
# # # Выводим DataFrame
# # print(filtered_df)
#
#
# plt.figure(figsize=(20,10))
#
# # Строим scatter plot, используя Fuel_Price для оси X, Type для оси Y и метки кластеров для цвета
# plt.scatter(filtered_df['Date'], filtered_df['Fuel_Price'], c=filtered_df['Cluster'])
#
# # Добавляем названия осей и заголовок
# plt.xlabel('Fuel_Price')
# plt.ylabel('Type')
# plt.title('KMeans Clustering')
#
# # Отображаем график
# plt.show()
# import pandas as pd
#
# # предположим, что df - это DataFrame с вашими данными о продажах
# # где 'StoreID' - идентификатор магазина, 'DepartmentID' - номер отдела, 'ProductName' - название товара, 'SaleDate' - дата продажи, 'Price' - цена товара, 'Quantity' - количество проданных единиц
#
# # вычисляем Recency и Frequency
# snapshot_date = df['SaleDate'].max() + pd.Timedelta(days=1)  # берем дату "снимка"
# df_rfm = df.groupby(['StoreID', 'DepartmentID', 'ProductName']).agg({
#     'SaleDate': lambda x: (snapshot_date - x.max()).days,  # Recency: количество дней с последней продажи
#     'Quantity': 'sum',  # Frequency: общее количество продаж
#     'Price': 'sum'  # Monetary: общая стоимость продаж (цена * количество)
# })
#
# # переименовываем столбцы
# df_rfm.rename(columns={'SaleDate': 'Recency',
#                        'Quantity': 'Frequency',
#                        'Price': 'MonetaryValue'}, inplace=True)
#
# # разделяем на квинтили
# df_rfm['RecencyQ'] = pd.qcut(df_rfm['Recency'], 5, labels=range(5, 0, -1))  # для Recency меньше - лучше
# df_rfm['FrequencyQ'] = pd.qcut(df_rfm['Frequency'], 5, labels=range(1, 6))  # для Frequency больше - лучше
# df_rfm['MonetaryQ'] = pd.qcut(df_rfm['MonetaryValue'], 5, labels=range(1, 6))  # для Monetary больше - лучше
# import numpy as np
# from seglearn.transform import FeatureRep, SegmentX
# from seglearn.feature_functions import mean, var, std
#
# # Создаем простой временной ряд
# time = np.arange(0, 1000)
# measurement = np.sin(time * 2 * np.pi / 24) + np.random.normal(size=1000)
# y = np.array([0 if i < 500 else 1 for i in range(1000)])
#
# # Извлекаем признаки с использованием Seglearn
# segmenter = SegmentX(width=100, overlap=0.5)
# X, y, _ = segmenter.fit_transform([measurement], [y])
#
# feature_extractor = FeatureRep(features={'mean': mean, 'var': var, 'std': std})
# X = feature_extractor.transform(X)
#
# print(X)
# counts = df['SALE'].dropna().value_counts()
# print(counts)
# import pandas as pd
# df = pd.DataFrame({'week':[1,2,3,4,5,6,7,8,10]})
#
#
# df['fff'] = df['week'].apply(lambda x: 1 if x in [1, 4, 6] else 0)
# # print(df)
# old_list = [7, 11, 15, 16]
# new_list = list(range(1, len(old_list) + 1))
# from skforecast.ForecasterAutoreg import ForecasterAutoreg
# from xgboost import XGBRegressor
#
# # Создаем список желаемых лагов
# my_lags = [1, 6, 12]
#
# forecaster_gb = ForecasterAutoreg(
#     regressor=XGBRegressor(random_state=963),
#     lags=my_lags
# )
# import matplotlib.pyplot as plt
#
# # Генерация случайной даты в указанном интервале
# random_date = random.choice(pd.date_range(start='2020-01-01', end='2023-01-01'))
#
# # Умножение значения в случайно выбранной точке на 1.2
# data.loc[random_date] *= 1.2
#
# # Построение графика временного ряда
# plt.figure(figsize=(10, 6))
# plt.plot(data.index, data.values, label='Time series')
#
# # Добавление точки на график
# plt.plot(random_date, data.loc[random_date], 'ro')  # 'ro' означает красные круглые маркеры
#
# # Добавление легенды
# plt.legend()
#
# # Отображение графика
# plt.show()
params_4 = {
    'depth': 10,
    'learning_rate':  0.22624323441899713,
    'iterations': 300
}


forecaster_cb_best_par_exog_import = ForecasterAutoreg(
                 regressor = CatBoostRegressor(random_state=963,silent=True,**params_4),
                 lags      = [1, 2, 3, 27,52],
                 differentiation = 1
             )
forecaster_cb_best_par_exog_import.fit(
        y = calendar_features['Values'],
        exog = calendar_features.iloc[:,1:]
)
feature_importances = forecaster_cb_best_par_exog_import.get_feature_importances()
print(feature_importances.sort_values(by='importance', ascending=False))