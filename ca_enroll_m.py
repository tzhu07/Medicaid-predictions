import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.offline as py
from matplotlib import pyplot as plt

# Read in the data
df = pd.read_csv('D:/rdata/ca_enroll_monthly.csv')

# Convert ds to datetime
df['ds'] = pd.to_datetime(df['ds'])

df['y'] = np.log(df['y'])
# Split the data into train and test sets
train = df[df['ds'] < '2018-01-01']
test = df[df['ds'] >= '2018-01-01']

import plotly.express as px
fig = px.line(train, x="ds", y="y")
fig.show()

# Fit the model on training data
m = Prophet(yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_prior_scale=0.)
m.fit(train)

# Predictions on training set
future_train = train[['ds']].copy()
forecast_train = m.predict(future_train)

# Predictions on test set
future_test = test[['ds']].copy()
forecast_test = m.predict(future_test)

# Calculate and print metrics for training set
train_mae = mean_absolute_error(train['y'], forecast_train['yhat'])
train_r2 = r2_score(train['y'], forecast_train['yhat'])
train_rmse = np.sqrt(mean_squared_error(train['y'], forecast_train['yhat']))
train_nrmse = train_rmse / (train['y'].max() - train['y'].min())
print(f'Train MAE: {train_mae}')
print(f'Train R2: {train_r2}')
print(f'Train RMSE: {train_rmse}')
print(f'Train NRMSE: {train_nrmse}')

# Calculate and print metrics for test set
test_mae = mean_absolute_error(test['y'], forecast_test['yhat'])
test_r2 = r2_score(test['y'], forecast_test['yhat'])
test_rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
test_nrmse = test_rmse / (test['y'].max() - test['y'].min())
print(f'Test MAE: {test_mae}')
print(f'Test R2: {test_r2}')
print(f'Test RMSE: {test_rmse}')
print(f'Test NRMSE: {test_nrmse}')

# Check for overfitting
if test_mae > train_mae and test_r2 < train_r2:
    print('Model may be overfitting, as training error is significantly lower than test error.')

# Fit the model on the whole data
m_full = Prophet(yearly_seasonality=False,
                 weekly_seasonality=False,
                 daily_seasonality=False,
                 seasonality_prior_scale=0.1
                )
m_full.fit(df)

# Predictions for the next 10 years
future = m_full.make_future_dataframe(periods=120, freq='M')
forecast = m_full.predict(future)

# Plot the forecast
m_full.plot(forecast)
plt.show()

# Plot the forecast components
m_full.plot_components(forecast)
plt.show()

# Create interactive plots
plot_plotly(m_full, forecast)
plot_components_plotly(m_full, forecast)

fig1 = plot_plotly(m_full, forecast)  # this returns a Plotly Figure
py.plot(fig1, filename='forecast.html')

fig2 = plot_components_plotly(m_full, forecast)  # this returns a Plotly Figure
py.plot(fig2, filename='components.html')

# Calculate metrics for the full data
full_forecast = m_full.predict(df[['ds']])

full_mae = mean_absolute_error(df['y'], full_forecast['yhat'])
full_r2 = r2_score(df['y'], full_forecast['yhat'])
full_rmse = np.sqrt(mean_squared_error(df['y'], full_forecast['yhat']))
full_nrmse = full_rmse / (df['y'].max() - df['y'].min())

print(f'Full data MAE: {full_mae}')
print(f'Full data R2: {full_r2}')
print(f'Full data RMSE: {full_rmse}')
print(f'Full data NRMSE: {full_nrmse}')


# Print the forecast
# log
test.loc[:, 'y'] = np.exp(test['y'])
forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.offline as py
from matplotlib import pyplot as plt

# Read in the data
df = pd.read_csv('D:/rdata/test1.csv')

# Convert ds to datetime
df['ds'] = pd.to_datetime(df['ds'])

df['y'] = np.log(df['y'])
# Split the data into train and test sets
train = df[df['ds'] < '2018-01-01']
test = df[df['ds'] >= '2018-01-01']

# Fit the model on training data
m = Prophet(yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_prior_scale=0.)
m.fit(train)

# Predictions on training set
future_train = train[['ds']].copy()
forecast_train = m.predict(future_train)

# Predictions on test set
future_test = test[['ds']].copy()
forecast_test = m.predict(future_test)

# Calculate and print metrics for training set
train_mae = mean_absolute_error(train['y'], forecast_train['yhat'])
train_r2 = r2_score(train['y'], forecast_train['yhat'])
train_rmse = np.sqrt(mean_squared_error(train['y'], forecast_train['yhat']))
train_nrmse = train_rmse / (train['y'].max() - train['y'].min())
print(f'Train MAE: {train_mae}')
print(f'Train R2: {train_r2}')
print(f'Train RMSE: {train_rmse}')
print(f'Train NRMSE: {train_nrmse}')

# Calculate and print metrics for test set
test_mae = mean_absolute_error(test['y'], forecast_test['yhat'])
test_r2 = r2_score(test['y'], forecast_test['yhat'])
test_rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
test_nrmse = test_rmse / (test['y'].max() - test['y'].min())
print(f'Test MAE: {test_mae}')
print(f'Test R2: {test_r2}')
print(f'Test RMSE: {test_rmse}')
print(f'Test NRMSE: {test_nrmse}')

# Check for overfitting
if test_mae > train_mae and test_r2 < train_r2:
    print('Model may be overfitting, as training error is significantly lower than test error.')

# Fit the model on the whole data
m_full = Prophet(yearly_seasonality=False,
                 weekly_seasonality=False,
                 daily_seasonality=False,
                 seasonality_prior_scale=0.1
                )
m_full.fit(df)

# Predictions for the next 10 years
future = m_full.make_future_dataframe(periods=24, freq='M')
forecast = m_full.predict(future)

# Plot the forecast
m_full.plot(forecast)
plt.show()

# Plot the forecast components
m_full.plot_components(forecast)
plt.show()

# Create interactive plots
plot_plotly(m_full, forecast)
plot_components_plotly(m_full, forecast)

fig1 = plot_plotly(m_full, forecast)  # this returns a Plotly Figure
py.plot(fig1, filename='forecast.html')

fig2 = plot_components_plotly(m_full, forecast)  # this returns a Plotly Figure
py.plot(fig2, filename='components.html')

# Calculate metrics for the full data
full_forecast = m_full.predict(df[['ds']])

full_mae = mean_absolute_error(df['y'], full_forecast['yhat'])
full_r2 = r2_score(df['y'], full_forecast['yhat'])
full_rmse = np.sqrt(mean_squared_error(df['y'], full_forecast['yhat']))
full_nrmse = full_rmse / (df['y'].max() - df['y'].min())

print(f'Full data MAE: {full_mae}')
print(f'Full data R2: {full_r2}')
print(f'Full data RMSE: {full_rmse}')
print(f'Full data NRMSE: {full_nrmse}')


# Print the forecast
# log
test.loc[:, 'y'] = np.exp(test['y'])
forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

