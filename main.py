import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.arima.model import ARIMA
from pydlm import dlm, trend, seasonality
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from datetime import datetime
import matplotlib.dates as mdates
def dateparse(x): return datetime.strptime(x, '%d.%m.%Y')


matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Prophet модель


class Model:
    def __init__(self, time_series, period, fourier):
        self.model_obj = Prophet(daily_seasonality=False,
                                 weekly_seasonality=False,
                                 yearly_seasonality=False)
        self.model_obj.add_seasonality(
            name="monthly", period=period, fourier_order=fourier)
        self.model_obj.fit(time_series)

    def forecast(self, count_elements):
        future = self.model_obj.make_future_dataframe(
            periods=count_elements, freq="D", include_history=False)
        predict = self.model_obj.predict(future)
        return predict

    def model(self):
        future = self.model_obj.make_future_dataframe(
            periods=0, freq="D", include_history=True)
        predict = self.model_obj.predict(future)
        return predict


def Arima(train, test, order):
    m_fit = ARIMA(train, order=order).fit()
    forecast = m_fit.get_forecast(steps=len(test))
    mean = forecast.predicted_mean
    rmse = np.sqrt(mean_squared_error(train, m_fit.fittedvalues)) ** 0.5
    return m_fit.fittedvalues, mean, rmse, m_fit.aic


def bsts(train, test, season):
    m_fit = dlm(train) + trend(degree=1, discount=0.9,
                               name='linear_trend') + seasonality(len(test)+season, discount=0.9)
    m_fit.fit()
    model_values = m_fit.getMean(filterType='backwardSmoother')
    forecast = m_fit.predictN(date=m_fit.n-1, N=len(test))[0]
    rmse = np.sqrt(mean_squared_error(train, model_values)) ** 0.5
    return model_values, forecast, rmse

# График x от y


def plot(x, y, name):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('date')
    ax.set_ylabel('y')
    ax.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.plot(x, y)
    plt.gcf().autofmt_xdate()
    plt.savefig("{}.png".format(name))
    plt.close()

# Несколько графиков сразу


def plot_all(x, y, legend, name, step_date=10):
    fig, ax = plt.subplots()
    ax.set_xlabel('date')
    ax.set_ylabel('y')
    ax.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=step_date))
    for i in range(len(x)):
        plt.plot(x[i], y[i], label=legend[i])
    plt.gcf().autofmt_xdate()
    plt.legend(loc="upper left")
    plt.title(name)
    plt.savefig("{}.png".format(name))
    plt.close()


def find_rmse(test, pred):
    return np.sqrt(mean_squared_error(test, pred))**0.5

# Нахождение наилучшей модели


def find_model(train, test, num, part):
    data = train
    train_x = np.array(data)
    train_y = np.arange(0, len(data))

    # Линейная модель
    model = LinearRegression().fit(train_y.reshape(-1, 1), train_x)
    trend_l = model.predict(train_y.reshape(-1, 1))
    rmse_l = find_rmse(data, trend_l)

    # Квадратичная регрессия
    coefficients_quad = np.poly1d(np.polyfit(train_y, train_x, 2))
    trend_quad = coefficients_quad[2] * train_y ** 2 + \
        coefficients_quad[1] * train_y + coefficients_quad[0]
    rmse_quad = find_rmse(data, trend_quad)

    # Экспоненциальная регрессия
    coefficients_exp = np.polyfit(train_y, np.log(train_x), 1)
    a_exp = np.exp(coefficients_exp[1])
    b_exp = coefficients_exp[0]
    trend_exp = a_exp * np.exp(b_exp * train_y)
    rmse_exp = find_rmse(data, trend_exp)

    # Полином Чебышева
    degree_chebyshev = 3  # Задаем степень полинома Чебышева
    coefficients_chebyshev = np.polynomial.chebyshev.chebfit(
        train_y, train_x, degree_chebyshev)
    # Считаем по МНК
    trend_cheb = np.polynomial.chebyshev.chebval(
        train_y, coefficients_chebyshev)
    rmse_cheb = find_rmse(data, trend_cheb)

    # Вывод
    rmse_values = [rmse_l, rmse_quad, rmse_exp, rmse_cheb]
    trend = [trend_l, trend_quad, trend_exp, trend_cheb]
    models = ['Линейная', 'Квадратичная',
              'Экспоненциальная', 'Полином Чебышева']
    plt.plot(np.arange(len(data)), data, label='original',
             linestyle="--", color="black")
    for i in range(4):
        cur_model = models[i]
        cur_trend = trend[i]
        print(f'{cur_model} {num} - {part} trend')
        print(rmse_values[i])
        plt.plot(np.arange(len(data)), cur_trend, label=f'{cur_model}')
    plt.xlabel('index')
    plt.ylabel('y')
    title = f'Тренды {num} - {part} trend'
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f'{title}.png')
    plt.close()

    best_model = models[np.argmin(rmse_values)]
    best_trend = trend[np.argmin(rmse_values)]
    return best_model, best_trend, min(rmse_values)

# Сезонность


def find_season(data, num, part):
    train_x = np.array(data)
    train_y = np.arange(0, len(data))
    # Выполнение дискретного преобразования Фурье
    fft_result = np.fft.rfft(train_x)
    # Получение амплитуды и частоты
    amplitude = np.abs(fft_result)
    frequency = np.fft.rfftfreq(len(data), 1)
    # Отбрасываем отрицательные частоты и их амплитуды
    positive_freq = frequency >= 0
    frequency = frequency[positive_freq]
    amplitude = amplitude[positive_freq]*2
    # Визуализация результатов
    plt.figure()
    plt.plot(frequency[1:], amplitude[1:])
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    days = len(data)
    plt.title(f'Fourier {num} - seasonality {part}')
    plt.grid()
    plt.savefig(f'Fourier {num} - seasonality {part}.png')
    plt.close()


def find_resid(data, num, part):
    # Тест Шапиро-Уилка на нормальность
    shapiro_test = stats.shapiro(data)
    print("\nДлина мерного интервала: ", len(data))
    print("Статистика теста Шапиро-Уилка:", shapiro_test.statistic)
    print("p-значение теста Шапиро-Уилка:", shapiro_test.pvalue)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q plot {num} - {part} normality')
    plt.xlabel('theoretical')
    plt.ylabel('data')
    plt.grid()
    plt.savefig(f'Q-Q plot {num} - {part} normality.png')
    plt.close()


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Считываем данные
    data1 = pd.read_csv("data1.csv", names=["y"])
    data1["ds"] = pd.date_range(start="2024-01-01", periods=len(data1))

    data2 = pd.read_csv("data2.csv", parse_dates=["ds"])
    # Для каждого набора данных
    for data in [data1, data2]:
        num = 1 if data is data1 else 2
        print(f'=== data {num}')
        print(len(data))
        # Определяем часть, отведённую для обучения
        train_part = 9/10
        # Для различных мерных интервалов (25%, 50%, 75%, 90%)
        for part in [1 / 4, 1 / 2, 3 / 4, train_part]:
            # Определяем границы обучающей выборки
            end = int(train_part * len(data))
            start = int(end - part * len(data))
            print(f'== train {part}')
            print(f'= {start} - {end - 1}')
            # Разбиваем данные на обучающую выборку и тесты
            train = data[start:end]['y'].to_numpy()
            train_date = data[start:end]['ds'].to_numpy()
            test = data[end:]['y'].to_numpy()
            test_date = data[end:]['ds'].to_numpy()
            # Определяем параметры
            best_rmse = float("inf")
            best_params = None
            # model, trend, rmse = find_model(train, test, num, part)
            find_season(train, num, part)
            # find_resid(train, num, part)
            # ARIMA
            d = num - 1
            for p in range(6):
                for q in range(6):
                    _, _, rmse, aic = Arima(train, test, (p, d, q))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = (p, d, q)
            print(f'Params: {best_params}')

            m, f, rmse, _ = Arima(train, test, best_params)
            print(f'rmse: {rmse}')
            model_data = np.concatenate([m, f])
            model_dates = np.concatenate([train_date, test_date])
            xs = [data['ds'], train_date, test_date,
                  model_dates[best_params[1]:]]
            ys = [data['y'], train, test, model_data[best_params[1]:]]
            legend = ["original", "train", "test", "model"]
            plot_all(xs, ys, legend, f'ARIMA{num} {
                     best_params} - {part} train', step_date=30)

            xs = [train_date, test_date]
            ys = [train - m, test - f]
            legend = ["train", "test"]
            plot_all(xs, ys, legend, f'ARIMA{num} {
                     best_params} - {part} remainders', step_date=30)

            # BSTS
            for season in range(30):
                _, _, rmse = bsts(train, test, season)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = season
            m, f, rmse = bsts(train, test, best_params)
            print(best_params, rmse)
            model_data = np.concatenate([m, f])
            model_dates = np.concatenate([train_date, test_date])
            xs = [data['ds'], train_date, test_date, model_dates]
            ys = [data['y'], train, test, model_data]
            legend = ["original", "train", "test", "model"]
            plot_all(xs, ys, legend, f'BSTS{num} {
                     best_params} - {part} train', step_date=30)

            xs = [train_date, test_date]
            ys = [train - m, test - f]
            legend = ["train", "test"]
            plot_all(xs, ys, legend, f'BSTS{num} {
                     best_params} - {part} remainders', step_date=30)

            # Prophet
            for period in range(10, 40):
                m = Model(data.iloc[start:end], period, 2)
                rmse = np.sqrt(mean_squared_error(
                    train, m.model()['yhat'])) ** 0.5
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = period
            best_fourier = None
            best_rmse = float("inf")
            for fourier in range(1, 4):
                m = Model(data.iloc[start:end], best_params, fourier)
                rmse = np.sqrt(mean_squared_error(
                    train, m.model()['yhat'])) ** 0.5
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_fourier = fourier
            print(best_params, best_fourier, best_rmse)
            m = Model(data.iloc[start:end], best_params, best_fourier)
            m_val = m.model()['yhat'].to_numpy()
            forecast = m.forecast(len(test))
            merged = pd.concat([m.model(), forecast],
                               ignore_index=True, sort=False)
            m_for = forecast['yhat'].to_numpy()
            xs = [data['ds'], train_date, test_date, merged['ds']]
            ys = [data['y'], train, test, merged['yhat']]
            legend = ["original", "train", "test", "model"]
            plot_all(xs, ys, legend, f'Prophet{num} {best_params} {
                     best_fourier} - {part} train', step_date=30)

            xs = [train_date, test_date]
            ys = [train - m_val, test - m_for]
            legend = ["train", "test"]
            plot_all(xs, ys, legend, f'Prophet{num} {best_params} {
                     best_fourier} - {part} remainders', step_date=30)
