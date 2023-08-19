#ANALISES PREDITIVAS
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

ticker = input("Digite o codigo da ação: ")
dados = yf.Ticker(ticker).history("2y")
print(dados.head())

dados["Close"].plot()


#tratamento de dados
dados.head()
#resetnado os indices
treinamento = dados.reset_index()
treinamento = treinamento [["Date", "Close"]]
treinamento["Date"] = treinamento["Date"].dt.tz_localize(None)
treinamento.columns = ['ds', 'y']
print(treinamento)

#TREINANDO O MODELO DE MACHINE LEARNING
#criar modelo de IA - ML
modelo = Prophet()

#treinar modelo
modelo.fit(treinamento)

#REALIZAR PREVISOES
periodo = modelo.make_future_dataframe(90)
print(periodo.tail())

previsoes = modelo.predict(periodo)

#gerar visualizacao grafica
grafico = plot_plotly(modelo, previsoes)
grafico.write_html("Grafico.html")
