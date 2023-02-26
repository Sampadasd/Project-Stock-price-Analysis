#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Libraries

# In[1]:


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pickle  #to load a saved model
import base64  #to open .gif files in streamlit app
from PIL import Image


# In[2]:


#HOME PAGE
@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages


# ## 1. For Home Page

# In[3]:


#DATASET
if app_mode=='Home':
    st.title('STOCK MARKET PRICE PREDICTION')
    st.markdown('The entire idea of predicting stock price is to gain significant profits.')
    st.image('image.jpeg',width=500)
    st.subheader('Dataset')
    st.markdown('The JSW STEEL dataset has information of 2 Years from 1 Nov 2017 to 1 Nov 2019.')
    data=pd.read_csv('JSW_Steel_Stock_Price.csv', index_col = 'Date', parse_dates = True)
    data_1 = data.sort_values(by = ['Date'])
    st.write(data_1)


# In[4]:


if app_mode=='Home':
    st.subheader('Visualization of our Dataset')
    fig = plt.figure(figsize = (13,9))
    data.OPEN.plot()
    data.HIGH.plot()
    data.LOW.plot()
    data.close.plot()

    plt.title('JSW Steel Stock Price')
    plt.ylabel('Price in Rupees')
    plt.legend(['Open','High','Low','Close'])
    plt.show()

    st.write(fig)
    
    st.subheader('Moving Average ( MA ) Trading Strategy')
    st.image('MA.png',width=800)
    st.markdown('Moving averages are commonly used in technical analysis of stocks to predict the past and future price trends.A 10-day moving average is a powerful tool to know if prices are moving in an uptrend or a downtrend.')


# ## 2. Data Understanding

# In[5]:


if app_mode=='Prediction':
    
    #Can be used wherever a "file-like" object is accepted:
    stock_details= pd.read_csv('JSW_Steel_Stock_Price.csv',index_col = 'Date', parse_dates = True)
    
    
    #Dropping unwanted columns
    stock_details = stock_details.drop(['series','PREV.CLOSE','ltp','vwap','52W H','52W L','VALUE','No of trades'], axis = 1)
    
    #Renaming the Columns
    stock_details.rename(columns = {'OPEN':'Open', 'HIGH':'High', 'LOW':'Low', 'close':'Close', 'VOLUME':'Volume'},inplace = True)
    ma_day = [10,20,30]
    for ma in ma_day:
        column_name = "MA for %s days" %(str(ma))
        stock_details[column_name] = pd.DataFrame.rolling(stock_details['Close'],ma).mean()
    df = stock_details.drop(['Volume','MA for 10 days','MA for 20 days','MA for 30 days',],axis = 1)
    stock_details_1 = df.sort_values(by = ['Date'])
    
    stock_details_resample = stock_details_1.resample('d').mean()
    
    stock_price = stock_details_resample.interpolate(method = "polynomial", order = 2)
    round(stock_price, 2)
    
    stock_price.reset_index(inplace = True)


# ## 3. Model Training 

# In[6]:


if app_mode=='Prediction':
    
    #Model Training
    data = stock_price.filter(['Open'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .90 ))

    #Normalizing Data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    #Creating X_train and y_train Data Structures
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            if i<= 61:
                print(x_train)
                print(y_train)

    #Reshape the Data
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 


# ## Model 1

# In[7]:


if app_mode=='Prediction':
    option = st.sidebar.selectbox('Select the Model Method',('Select','Neural Network', 'Forecasting'))
    
    if option=='Neural Network':
        method_1 = st.sidebar.selectbox('Select the Model',('Select','Stacked LSTM','Bidirectional LSTM',
                                                           'GRU (Gated Recurrent Unit) LSTM',
                                                           'BGRU (Bidirectional GRU) LSTM','Future Forecasting'))
        st.header('Now Lets Predict the Price')
        st.sidebar.write('You selected:', method_1)
        
        #Model 1
        if method_1 == 'Stacked LSTM':
            if st.sidebar.button('Submit'):
                st.subheader('Predicting the Price Using Stacked LSTM For Last 72 Days')
                #Model Building
                from keras.models import Sequential
                from keras.layers import Dense, LSTM
                model_1 = Sequential()
                model_1.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
                model_1.add(LSTM(64, return_sequences = False))
                model_1.add(Dense(25))
                model_1.add(Dense(1))
                #Fitting Model
                model_1.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
                model_1.fit(x_train, y_train, batch_size = 1, epochs = 1)
                #Create Test Dataset
                test_data = scaled_data[training_data_len - 60: , :]
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
                #Predicting Values
                predictions_1 = model_1.predict(x_test)
                predictions_1 = scaler.inverse_transform(predictions_1)
                #RMSE Value
                rmse_1 = np.sqrt(np.mean(((predictions_1 - y_test) ** 2)))
                print('RMSE value of Stacked LSTM :', rmse_1)
                #Ploting
                train = data[:training_data_len]
                valid_1 = data[training_data_len:]
                valid_1['Predictions_1'] = predictions_1
                fig_1 = plt.figure(figsize = (16,6))
                plt.title('Stacked LSTM Plot')
                plt.xlabel('Date', fontsize = 18)
                plt.ylabel('Open Price in Rupees(₹)', fontsize = 18)
                plt.plot(train['Open'])
                plt.plot(valid_1[['Open', 'Predictions_1']])
                plt.legend(['Train', 'Valid_1', 'Predictions_1'], loc = 'upper right')
                plt.show()
                st.write(fig_1)

                a = valid_1
                st.write(a)
                st.write('RMSE value of Stacked LSTM :',rmse_1)

                st.success('Prediction Done', icon="✅")
            
            
                    
            
    else:
        method_2 = st.sidebar.selectbox('Select the Model',('Select','Data Driven Forecasting Methods'))
        st.sidebar.write('You selected:', method_2)
        
        #Forecasting
        if method_2 == 'Data Driven Forecasting Methods':
            if st.sidebar.button('Submit'):
            
                #1. Data Driven Forecasting Methods
                df = stock_price.iloc[:,1:2]
                from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
                from statsmodels.tsa.holtwinters import Holt 
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                Train = df.head(550)
                Test = df.tail(181)
                from statsmodels.tsa.seasonal import seasonal_decompose
                ts_decompose = seasonal_decompose(df.Open,period = 12)
                decompose = ts_decompose.plot()
                plt.show()
                st.header('Time series decomposition plot')
                st.write(decompose)
                from sklearn.metrics import mean_squared_error
                from math import sqrt
                import warnings
                warnings.filterwarnings('ignore')
                def RMSE(org, pred):
                    rmse = np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
                    return rmse

                #Simple Exponential Method
                simple_model = SimpleExpSmoothing(Train["Open"]).fit()
                pred_simple_model = simple_model.predict(start = Test.index[0],end = Test.index[-1])
                rmse_simple_model = RMSE(Test.Open, pred_simple_model)
                st.subheader('1. Simple Exponential Method')
                st.write('RMSE Value of Simple Exponential :',rmse_simple_model)
                #Holt Method
                holt_model = Holt(Train["Open"]).fit()
                pred_holt_model = holt_model.predict(start = Test.index[0],end = Test.index[-1])
                rmse_holt_model = RMSE(Test.Open, pred_holt_model)
                st.subheader('2. Holt Method')
                st.write('RMSE Value of Holt :',rmse_holt_model)
                #Holts winter exponential smoothing with additive seasonality and additive trend
                holt_model_add_add = ExponentialSmoothing(Train["Open"],seasonal = "add",trend = "add",seasonal_periods = 4).fit()
                pred_holt_add_add = holt_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
                rmse_holt_add_add_model = RMSE(Test.Open, pred_holt_add_add)
                st.subheader('3. Holts winter exponential smoothing with additive seasonality and additive trend')
                st.write('RMSE Value of Holts add and add :',rmse_holt_add_add_model)
                #Holts winter exponential smoothing with multiplicative seasonality and additive trend
                holt_model_multi_add = ExponentialSmoothing(Train["Open"],seasonal = "mul",trend = "add",seasonal_periods = 4).fit() 
                pred_holt_multi_add = holt_model_multi_add.predict(start = Test.index[0],end = Test.index[-1])
                rmse_holt_model_multi_add_model = RMSE(Test.Open, pred_holt_multi_add)
                st.subheader('4. Holts winter exponential smoothing with multiplicative seasonality and additive trend')
                st.write('RMSE Value of Holts Multi and add :',rmse_holt_model_multi_add_model)

                #2. Model based Forecasting Methods
                df = stock_price.iloc[:,1:2]
                df["t"] = np.arange(0,731)
                df["t_squared"] = df["t"]*df["t"]
                df["log_Open"] = np.log(df["Open"])
                df.head()
                Train = df.head(550)
                Test = df.tail(181)

                #Linear Model
                import statsmodels.formula.api as smf 
                linear_model = smf.ols('Open~t',data = Train).fit()
                pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
                rmse_linear_model = RMSE(Test['Open'], pred_linear)
                st.subheader('5. Linear Model')
                st.write('RMSE Value of Linear :',rmse_linear_model)
                #Exponential Model
                Exp_model = smf.ols('log_Open~t',data = Train).fit()
                pred_Exp = pd.Series(Exp_model.predict(pd.DataFrame(Test['t'])))
                rmse_Exp_model = RMSE(Test['Open'], np.exp(pred_Exp))
                st.subheader('6. Exponential Model')
                st.write('RMSE Value of Exponential :',rmse_Exp_model)
                #Quadratic Model
                Quad_model= smf.ols('Open~t+t_squared',data = Train).fit()
                pred_Quad = pd.Series(Quad_model.predict(Test[["t","t_squared"]]))
                rmse_Quad_model = RMSE(Test['Open'], pred_Quad)
                st.subheader('7. Quadratic Model')
                st.write('RMSE Value of Quadratic :',rmse_Quad_model)

                #ARIMA Model
                series = stock_price.iloc[:,1:2]
                split_point = len(series) - 12
                split_point = len(series) - 12
                dataset, validation = series[0:split_point], series[split_point:]
                print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
                dataset.to_csv('dataset.csv', header = False)
                validation.to_csv('validation.csv', header = False)
                from pandas import read_csv
                from sklearn.metrics import mean_squared_error
                from math import sqrt
                train = read_csv('dataset.csv', header = None, index_col = 0, parse_dates = True, squeeze = True)
                X = train.values
                X = X.astype('float32')
                train_size = int(len(X) * 0.75)
                train, test = X[0:train_size], X[train_size:]
                history = [x for x in train]
                predictions = list()
                for i in range(len(test)):
                    yhat = history[-1]
                    predictions.append(yhat)
                # observation
                    obs = test[i]
                    history.append(obs)
                    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
                rmse = sqrt(mean_squared_error(test, predictions))
                print('RMSE Value of ARIMA : %.3f' % rmse)
                rmse_Persistence_model = 5.117

                st.subheader('8. ARIMA Model')
                st.write('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
                st.write('RMSE Value of ARIMA :',rmse_Persistence_model)

                st.success('Prediction Done', icon="✅")


                #Conclusion
                st.header('Conclusion')
                list = [['Simple Exponential Method',rmse_simple_model],
                ['Holt Method',rmse_holt_model],['Holt exp smoothing add Method',rmse_holt_add_add_model],
                ['Holt exp smoothing multi Method',rmse_holt_model_multi_add_model],['Linear Model',rmse_linear_model],
                ['Exponential Model',rmse_Exp_model],['Quadratic Model',rmse_Quad_model],
                ['Persistence/ ARIMA Model', rmse_Persistence_model]]

                Result = pd.DataFrame(list, columns = ['Models', 'RMSE Values'])
                plot = plt.figure(figsize = (15,6))
                sns.barplot(data = Result,x = 'Models',y = 'RMSE Values')
                plt.title('RMSE Value vs Model Plot', fontsize=20)
                plt.xlabel('Models', fontsize=18)
                plt.ylabel('RMSE Values', fontsize=18)
                plt.xticks(rotation = 90)
                plt.show()

                st.subheader('RMSE Values of Forecasting Methods')
                st.dataframe(Result)

                st.subheader('Visualization Plot for RMSE Values Vs Forecasting Models')
                st.pyplot(plot)


# ## Model 2

# In[8]:


if app_mode=='Prediction':
    if option=='Neural Network':
        
        #Model 2
        if method_1 == 'Bidirectional LSTM':
            if st.sidebar.button('Submit'):
                st.subheader('Predicting the Price Using Bidirectional LSTM For Last 72 Days')
                #Model Building
                from keras.layers import Bidirectional
                from keras.models import Sequential
                from keras.layers import Dense, LSTM
                model_2 = Sequential()
                model_2.add(Bidirectional(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1))))
                model_2.add(Bidirectional(LSTM(units = 50)))
                model_2.add(Dense(units = 1))
                #Fitting Model
                model_2.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
                model_2.fit(x_train, y_train, batch_size = 1, epochs = 1)
                #Create Test Dataset
                test_data = scaled_data[training_data_len - 60: , :]
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
                #Predicting Values            
                predictions_2 = model_2.predict(x_test)
                predictions_2 = scaler.inverse_transform(predictions_2)
                #RMSE Value
                rmse_2 = np.sqrt(np.mean(((predictions_2 - y_test) ** 2)))
                print('RMSE value of Bidirectional LSTM :', rmse_2)
                #Ploting
                train = data[:training_data_len]
                valid_2 = data[training_data_len:]
                valid_2['Predictions_2'] = predictions_2
                fig_2 = plt.figure(figsize = (16,6))
                plt.title('Bidirectional LSTM Plot')
                plt.xlabel('Date', fontsize = 18)
                plt.ylabel('Open Price in Rupees(₹)', fontsize=18)
                plt.plot(train['Open'])
                plt.plot(valid_2[['Open', 'Predictions_2']])
                plt.legend(['Train', 'Valid_2', 'Predictions_2'], loc = 'upper right')
                plt.show()
                st.write(fig_2)

                b = valid_2
                st.write(b)            
                st.write('RMSE value of Bidirectional LSTM :',rmse_2)   

                st.success('Prediction Done', icon="✅")


# ## Model 3

# In[9]:


if app_mode=='Prediction':
    if option=='Neural Network':
        
        #Model 3
        if method_1 == 'GRU (Gated Recurrent Unit) LSTM':
            if st.sidebar.button('Submit'):
                st.subheader('Predicting the Price Using GRU (Gated Recurrent Unit) LSTM For Last 72 Days')
                #Model Building
                from keras.layers import GRU
                from keras.models import Sequential
                from keras.layers import Dense, LSTM
                model_3 = Sequential()
                model_3.add(GRU(units = 50,input_shape = (x_train.shape[1],1)))
                model_3.add(Dense(units = 1))
                #Fitting Model
                model_3.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
                model_3.fit(x_train, y_train, batch_size = 1, epochs = 1)
                #Create Test Dataset
                test_data = scaled_data[training_data_len - 60: , :]
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
                #Predicting Values
                predictions_3 = model_3.predict(x_test)
                predictions_3 = scaler.inverse_transform(predictions_3)
                #RMSE Value
                rmse_3 = np.sqrt(np.mean(((predictions_3 - y_test) ** 2)))
                print('RMSE value of GRU LSTM :', rmse_3)
                #Plotting
                train = data[:training_data_len]
                valid_3 = data[training_data_len:]
                valid_3['Predictions_3'] = predictions_3
                fig_3 = plt.figure(figsize = (16,6))
                plt.title('GRU LSTM Plot')
                plt.xlabel('Date', fontsize = 18)
                plt.ylabel('Open Price in Rupees(₹)', fontsize = 18)
                plt.plot(train['Open'])
                plt.plot(valid_3[['Open', 'Predictions_3']])
                plt.legend(['Train', 'Valid_3', 'Predictions_3'], loc = 'upper right')
                plt.show()  
                st.write(fig_3)

                c = valid_3
                st.write(c)
                st.write('RMSE value of GRU LSTM :',rmse_3)

                st.success('Prediction Done', icon="✅")


# ## Model 4

# In[10]:


if app_mode=='Prediction':
    if option=='Neural Network':
        
        #Model 4
        if method_1 == 'BGRU (Bidirectional GRU) LSTM':
            if st.sidebar.button('Submit'):
                st.subheader('Predicting the Price Using  BGRU (Bidirectional GRU) LSTM For Last 72 Days')
                #Model Building
                from keras.layers import SpatialDropout1D
                from keras.models import Sequential
                from keras.layers import Dense, LSTM
                from tensorflow.keras.layers import Bidirectional
                from keras.layers import GRU
                model_4 = Sequential()
                model_4.add(Bidirectional(GRU(units = 50,input_shape = (x_train.shape[1],1))))
                model_4.add(Dense(units = 1))
                #Fitting Model
                model_4.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
                model_4.fit(x_train, y_train, batch_size = 1, epochs = 1)
                #Create Test Dataset
                test_data = scaled_data[training_data_len - 60: , :]
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
                #Predicting Values
                predictions_4 = model_4.predict(x_test)
                predictions_4 = scaler.inverse_transform(predictions_4)
                #RMSE Value
                rmse_4 = np.sqrt(np.mean(((predictions_4 - y_test) ** 2)))
                print('RMSE value of Bidirectional GRU LSTM :', rmse_4)
                #Plotting
                train = data[:training_data_len]
                valid_4 = data[training_data_len:]
                valid_4['Predictions_4'] = predictions_4
                fig_4 = plt.figure(figsize = (16,6))
                plt.title('Bidirectional GRU LSTM Plot')
                plt.xlabel('Date', fontsize = 18)
                plt.ylabel('Open Price in Rupees(₹)', fontsize = 18)
                plt.plot(train['Open'])
                plt.plot(valid_4[['Open', 'Predictions_4']])
                plt.legend(['Train', 'Valid_4', 'Predictions_4'], loc = 'upper right')
                plt.show()
                st.write(fig_4)

                d = valid_4
                st.write(d)
                st.write('RMSE value of Bidirectional GRU LSTM :',rmse_4)

                st.success('Prediction Done', icon="✅")


# ## Model 5

# In[12]:


if app_mode=='Prediction':
    if option=='Neural Network':
        
        #Model 5
        if method_1 == 'Future Forecasting':
            if st.sidebar.button('Submit'):
                st.subheader('Forecasting Future Prediction For 90 Days Using LSTM')
                df = stock_details_resample.interpolate(method = "polynomial", order = 2)
                round(df, 2)
                y = df['Open'].fillna(method='ffill')
                y = y.values.reshape(-1,1)
                scaler = MinMaxScaler(feature_range=(0,1))
                scaler=scaler.fit(y)
                y=scaler.transform(y)
                n_lookback =180
                n_forecast =90
                X =[]
                Y =[]

                for i in range(n_lookback, len(y) - n_forecast + 1):
                    X.append(y[i - n_lookback: i])
                    Y.append(y[i: i + n_forecast])

                X = np.array(X)
                Y = np.array(Y)
                from keras.models import Sequential
                from keras.layers import Dense, LSTM
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
                model.add(LSTM(units=50))
                model.add(Dense(n_forecast)) 
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(X, Y, epochs=10, batch_size=32, verbose=0)
                X_ = y[- n_lookback:]  
                X_ = X_.reshape(1, n_lookback, 1)
                Y_ = model.predict(X_).reshape(-1, 1)
                Y_ = scaler.inverse_transform(Y_)
                df_past = df[['Open']].reset_index()
                df_past.rename(columns={'index': 'Date', 'Open': 'Actual'}, inplace=True)
                df_past['Date'] = pd.to_datetime(df_past['Date'])
                df_past['Forecast'] = np.nan
                df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
                df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
                df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1) + pd.Timedelta(days=1), 
                                                  periods=n_forecast)
                df_future['Forecast'] = Y_.flatten()
                df_future['Actual'] = np.nan
                results = df_past.append(df_future).set_index('Date')
                gra = results.plot(title='JSW Steel 3 Months Future Prediction')
                plt.ylabel('Open Price in Rupees(₹)')
                plt.show()

                future = df_future['Forecast']

                fig_41 = plt.figure(figsize = (16,6))
                plt.title('Bidirectional GRU LSTM Plot')
                plt.xlabel('Date', fontsize = 18)
                plt.ylabel('Open Price in Rupees(₹)')
                plt.plot(results)
                plt.show()
                st.write(fig_41)
                st.write(future)
                
                st.success('Prediction Done', icon="✅")

