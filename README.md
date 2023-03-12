# Project-Stock-price-Analysis
-The entire idea of Analysing stock prices to gain significant profits.
-Analysed useful columns.
-Found correlation between columns, columns Open,Close,High and Low are closely related with each other.
-Visualised data using: pair-plot, line-plot and checked the ups and downs in stock prices.
-Visualised data using Moving Average to check if prices are moving in an uptrend or a downtrend.
-Using Expanding Window method concluded that there was less loss and more profit.
-Downsampled data to maintain consistency in data.
-Applied Neural Networks and their RMSE values:
 1) Stacked LSTM: RMSE-11.09
 2) Bidirectional LSTM: RMSE-11.80
 3) GRU(Gated Recurrent Unit) LSTM: RMSE- 6.67
 4) BGRU(Bidirectional GRU) LSTM: RMSE-6.93
-Data Driven Forecasting Methods:
 1) Simple Exponential Method: RMSE:64.45
 2) Holt Method: RMSE- 71.87
 3) Holts Winter Exponential smoothing and additive seasonality and aditive trend: RMSE- 71.88
 4) Holts Winter Exponential smoothing and multiplicative seasonality and aditive trend: RMSE- 71.76
-Model Based Forecasting Methods:
1) Linear model: RMSE- 84.92 
2) Exponential Model: RMSE- 83.83
3) Quadratic Model: RMSE- 58.76
-We have taken past 6 months of data and predicted for 3 months of future prediction and we can see from graph that the future prediction is going low.
