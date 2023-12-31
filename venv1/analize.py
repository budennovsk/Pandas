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
import pandas as pd

# предположим, что df - это DataFrame с вашими данными о продажах
# где 'StoreID' - идентификатор магазина, 'DepartmentID' - номер отдела, 'ProductName' - название товара, 'SaleDate' - дата продажи, 'Price' - цена товара, 'Quantity' - количество проданных единиц

# вычисляем Recency и Frequency
snapshot_date = df['SaleDate'].max() + pd.Timedelta(days=1)  # берем дату "снимка"
df_rfm = df.groupby(['StoreID', 'DepartmentID', 'ProductName']).agg({
    'SaleDate': lambda x: (snapshot_date - x.max()).days,  # Recency: количество дней с последней продажи
    'Quantity': 'sum',  # Frequency: общее количество продаж
    'Price': 'sum'  # Monetary: общая стоимость продаж (цена * количество)
})

# переименовываем столбцы
df_rfm.rename(columns={'SaleDate': 'Recency',
                       'Quantity': 'Frequency',
                       'Price': 'MonetaryValue'}, inplace=True)

# разделяем на квинтили
df_rfm['RecencyQ'] = pd.qcut(df_rfm['Recency'], 5, labels=range(5, 0, -1))  # для Recency меньше - лучше
df_rfm['FrequencyQ'] = pd.qcut(df_rfm['Frequency'], 5, labels=range(1, 6))  # для Frequency больше - лучше
df_rfm['MonetaryQ'] = pd.qcut(df_rfm['MonetaryValue'], 5, labels=range(1, 6))  # для Monetary больше - лучше