import pandas as pd
import numpy as np
import ccxt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Funções para calcular os indicadores técnicos
def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    short_ema = data['close'].ewm(span=12, adjust=False).mean()
    long_ema = data['close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_stochastic(data, k_period=14, d_period=3):
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    percent_k = 100 * (data['close'] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d

# Função para adicionar todos os indicadores técnicos ao DataFrame
def add_technical_indicators(data):
    data['ema_short'] = calculate_ema(data, 9)
    data['ema_long'] = calculate_ema(data, 21)
    data['rsi'] = calculate_rsi(data, 14)
    data['macd'], data['macd_signal'] = calculate_macd(data)
    data['stochastic_k'], data['stochastic_d'] = calculate_stochastic(data)
    return data

# Função para carregar os dados
def get_data(symbol, timeframe='1h', limit=100):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data = add_technical_indicators(data)
    return data

# Carregar os dados de 2 anos
symbol = 'BTC/USDT'
data = get_data(symbol, timeframe='1h', limit=2000)

# Preparar os dados para treinamento
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  # 1 se preço subiu no próximo período
data = data.dropna()

# Definir as features e o target
features = ['close', 'ema_short', 'ema_long', 'rsi', 'macd', 'stochastic_k', 'stochastic_d']
X = data[features]
y = data['target']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceamento de classes com SMOTE
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Ajuste de Hiperparâmetros usando GridSearchCV com RandomForest
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

# Melhor modelo após o ajuste
best_rf_model = grid_search.best_estimator_

# Avaliar a acurácia no conjunto de teste
y_pred_rf = best_rf_model.predict(X_test)
print(f'Acurácia do Random Forest: {accuracy_score(y_test, y_pred_rf)}')

# Ensemble de Modelos (Random Forest + XGBoost)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, random_state=42)
xgb_model.fit(X_train_res, y_train_res)

# Prevendo com os dois modelos e combinando os resultados (votação majoritária)
rf_pred = best_rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

final_pred = np.round((rf_pred + xgb_pred) / 2)
ensemble_accuracy = accuracy_score(y_test, final_pred)
print(f'Acurácia do Ensemble: {ensemble_accuracy}')

# Implementação do modelo LSTM para análise de séries temporais (Deep Learning)
X_train_lstm = X_train_res.values.reshape((X_train_res.shape[0], X_train_res.shape[1], 1))
X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(X_train_lstm, y_train_res, epochs=10, batch_size=32)

# Previsões LSTM
lstm_predictions = lstm_model.predict(X_test_lstm)

# Avaliar o modelo LSTM
mae = mean_absolute_error(y_test, lstm_predictions)
print(f'Mean Absolute Error (LSTM): {mae}')

# Visualizar as previsões vs reais (para LSTM)
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Real')
plt.plot(lstm_predictions, label='Previsão LSTM', linestyle='dashed')
plt.legend()
plt.title("Comparação entre Real e Previsões do LSTM")
plt.show()
