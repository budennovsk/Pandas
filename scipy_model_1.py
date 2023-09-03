from scipy.optimize import minimize
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

print('ses')
# Генерация случайных данных для примера
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# Создание экземпляра модели CatBoostRegressor


# Обучение модели на тренировочных данных

print('gggg')

# Определение функции потерь, которую вы хотите минимизировать
def loss_function(parameters):
    print(parameters, 'ser')
    # Установка параметров модели на основе переданных значений
    model = CatBoostRegressor(iterations=int(parameters[0]), learning_rate=parameters[1], depth=int(parameters[2]))

    print('llll')
    # Обучение модели на тренировочных данных
    model.fit(X_train, y_train, verbose=False)

    # Вычисление и возврат значения функции потерь
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    print('fff')
    print(y_pred)
    print(loss)
    return loss

print('q1')
# Начальные значения параметров модели
initial_parameters = [100, 0.1, 5]
print('q2')
# Запуск минимизации функции потерь
result = minimize(loss_function, initial_parameters, method='Nelder-Mead')
print('q3')
# Оптимальные значения параметров
optimal_parameters = result.x
print("Optimal Parameters:", optimal_parameters)
print(result)
