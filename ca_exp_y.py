# us expenditure yearly
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.offline as py
from sklearn.metrics import mean_squared_error
import numpy as np
from prophet.plot import plot_plotly, plot_components_plotly


df = pd.read_csv('D:/rdata/ca_exp_yearly.csv', skiprows=1, names=['ds', 'y'])

# Create a Prophet model
model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
model.fit(df)

# Next 10 years' prediction
future = model.make_future_dataframe(periods=10, freq='Y')
forecast = model.predict(future)

# Print the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
model.plot(forecast)
plt.show()

plot_plotly(model, forecast)
plot_components_plotly(model, forecast)

fig1 = plot_plotly(model, forecast)  # this returns a Plotly Figure
py.plot(fig1, filename='forecast.html')

fig2 = plot_components_plotly(model, forecast)  # this returns a Plotly Figure
py.plot(fig2, filename='components.html')

# Calculate RMSE
mse = mean_squared_error(df['y'], forecast['yhat'].head(len(df)))
rmse = np.sqrt(mse)

# Calculate NRMSE
range_y = df['y'].max() - df['y'].min()
nrmse = rmse / range_y

print(f'RMSE: {rmse}')
print(f'NRMSE: {nrmse}')