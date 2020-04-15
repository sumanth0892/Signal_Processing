#Packages to plot data 
import matplotlib.pyplot as plt 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd 
import os

#Date wrangling 
from datetime import datetime,timedelta 

#The deep learning class 
from deep_model import DeepModelTS

#Read the configuration file 
import yaml 

#Read the hyperparameters 
with open('conf.yml','r') as file:
	conf = yaml.load(file,Loader = yaml.FullLoader)

#Read the data 
df = pd.read_csv('DAYTON_hourly.csv')

#Erase duplicates 
df = df.groupby('Datetime',as_index = False)['DAYTON_MW'].mean()
print(df.info())
df.sort_values('Datetime', inplace=True)

#Initialize the class 
deepLearner = DeepModelTS(data = df,
	Y_var = 'DAYTON_MW',
	lag = conf.get('lag'),
	LSTM_layer_depth = conf.get('LSTM_layer_depth'),
	epochs = conf.get('epochs'),
	train_test_split = conf.get('train_test_split'))

#Fit the model 
model = deepLearner.LSTModel()

#Make predictions on the validation set 
yhat = deepLearner.predict()

if len(yhat)>0:
	#Construct the forecast dataframe 
	fc = df.tail(len(yhat)).copy()
	fc.reset_index(inplace = True)
	fc['forecast'] = yhat 

	#Plot the forecasts 
	plt.figure(figsize = (12,8))
	for dtype in ['DAYTON_MW','forecast']:
		plt.plot('Datetime',
			dtype,
			data = fc,
			label = dtype,
			alpha = 0.8)
	plt.legend()
	plt.grid()
	plt.show()

#Forecast n steps ahead
deepPredictor = DeepModelTS(
    data=df, 
    Y_var='DAYTON_MW',
    lag=24,
    LSTM_layer_depth=64,
    epochs=10,
    train_test_split=0 
)

deepPredictor.LSTModel()
n_ahead = 168
yhat = deepPredictor.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

#Forecast dataframe 
fc = df.tail(400).copy()
fc['type'] = 'original'

lastDate = max(fc['Datetime'])
hatFrame = pd.DataFrame({
    'Datetime': [last_date + timedelta(hours=x + 1) for x in range(n_ahead)], 
    'DAYTON_MW': yhat,
    'type': 'forecast'})
fc = fc.append(hatFrame)
fc.reset_index(inplace = True,drop = True)

#Plot the forecasts 
plt.figure(figsize = (12,8))
for colType in ['original','forecast']:
	plt.plot(
		'Datetime',
		'DAYTON_MW',
		data = fc[fc['type'] == colType],
		label = colType)

plt.legend()
plt.grid()
plt.show()



