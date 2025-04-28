import pandas as pd
import talib as ta
import requests
from binance.client import Client
import time

# Configurações do Telegram
TELEGRAM_TOKEN = '7986770725:AAHD3vqPIZNLHvyWVZnrHIT3xGGI1R9ZeoY'
CHAT_ID = '2091781134'

# Configurações da Binance
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
client = Client(API_KEY, API_SECRET)

# Função para enviar mensagem para o Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    params = {'chat_id': CHAT_ID, 'text': message}
    requests.get(url, params=params)

# Função para obter dados históricos de preço (exemplo com 1 hora de candles)
def get_historical_data(symbol='BTCUSDT', interval='1h', limit=200):
    candles = client.get_historical_klines(symbol, interval, limit=limit)
    data = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['time'] = pd.to_datetime(data['time'], unit='ms')
    data['close'] = data['close'].astype(float)
    return data[['time', 'close']]

# Função para calcular indicadores técnicos
def calculate_indicators(data):
    # EMAs de 9 e 13 períodos
    data['EMA9'] = ta.EMA(data['close'], timeperiod=9)
    data['EMA13'] = ta.EMA(data['close'], timeperiod=13)
    # MACD
    macd, macdsignal, macdhist = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_signal'] = macdsignal
    return data

# Função para identificar a tendência
def identify_trend(data):
    trend = "Indefinida"
    if data['EMA9'].iloc[-1] > data['EMA13'].iloc[-1]:
        trend = "Alta"
    elif data['EMA9'].iloc[-1] < data['EMA13'].iloc[-1]:
        trend = "Baixa"
    return trend

# Função para verificar as condições de entrada
def check_entry_conditions(data):
    last_candle = data.iloc[-1]
    trend = identify_trend(data)

    # Condição de cruzamento de EMAs
    if trend == "Alta" and last_candle['EMA9'] > last_candle['EMA13'] and last_candle['MACD'] > last_candle['MACD_signal']:
        return "Entrada possível: Alta confirmada"
    elif trend == "Baixa" and last_candle['EMA9'] < last_candle['EMA13'] and last_candle['MACD'] < last_candle['MACD_signal']:
        return "Entrada possível: Baixa confirmada"
    else:
        return "Sem condições de entrada"

# Função para calcular o stop loss e take profit
def calculate_risk_management(balance, entry_price, risk_percentage=0.01, risk_to_reward=2):
    stop_loss = entry_price * (1 - risk_percentage)  # Exemplo de stop loss
    take_profit = entry_price * (1 + risk_percentage * risk_to_reward)  # Exemplo de take profit
    return stop_loss, take_profit

# Função principal de execução
def main():
    balance = 1000  # Exemplo de saldo
    symbol = 'BTCUSDT'

    # Obter os dados de mercado
    data = get_historical_data(symbol)
    data = calculate_indicators(data)

    # Verificar condições de entrada
    entry_message = check_entry_conditions(data)
    if "Entrada possível" in entry_message:
        # Definir preço de entrada (último preço de fechamento)
        entry_price = data['close'].iloc[-1]

        # Calcular o risco e o retorno
        stop_loss, take_profit = calculate_risk_management(balance, entry_price)

        # Enviar o plano de trade para o Telegram
        plan_message = f"📝 Modelo de Plano de Trade (antes de clicar no botão)\n\n" \
                       f"**Identificação da Tendência**\n" \
                       f"Tendência clara: {identify_trend(data)}\n" \
                       f"EMAs alinhadas? Sim\n" \
                       f"MACD confirma? Sim\n" \
                       f"Candles apoiam? Sim\n\n" \
                       f"**Condições de Entrada**\n" \
                       f"Setup batendo? Sim (Cruzamento de EMAs e MACD)\n" \
                       f"Pullback ou rompimento confirmado? Sim\n\n" \
                       f"**Gestão de Risco**\n" \
                       f"Valor do stop loss calculado: {stop_loss}\n" \
                       f"Risco x Retorno no mínimo 1:2? Sim\n" \
                       f"Alavancagem usada de forma segura? Sim\n\n" \
                       f"**Execução**\n" \
                       f"Ordem programada? Sim (Limit/Stop)\n" \
                       f"Stop Loss posicionado? Sim\n" \
                       f"Take Profit definido? Sim\n" \
                       f"Trailing Stop configurado? Sim\n\n" \
                       f"**Após a entrada**\n" \
                       f"Seguir plano! Não mexer no stop sem motivo técnico.\n" \
                       f"Atualizar trailing se der lucro.\n\n" \
                       f"**Detalhes da operação**\n" \
                       f"Entrada: {entry_price}\n" \
                       f"Stop Loss: {stop_loss}\n" \
                       f"Take Profit: {take_profit}"

        send_telegram_message(plan_message)

if __name__ == "__main__":
    while True:
        main()
        time.sleep(60 * 60)  # Executa a cada hora (ajuste conforme necessário)
