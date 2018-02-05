# Import necessary libraries
from pandas import read_csv
#from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
# AIC
from statsmodels.tsa.arima_model import ARMAResults
 
# Import the dataset 
series = read_csv('YearAndTotalDistanceTravelled.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
# fit model
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
# print the summary table
print(model_fit.summary())
# calculate & print aic
print(model_fit.aic)
# Print AIC table
# print(ARMAResults.summary(model_fit))
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
#residual
residuals.plot(kind='kde')
pyplot.show()

print(residuals.describe())
# AIC
for xx in range(0, 5):
    model5 = ARIMA(series, order=(xx,1,0))
    model_fit5 = model5.fit(disp=0)
    print(model_fit5.aic)

for xx in range(0, 5):
    model5 = ARIMA(series, order=(xx,2,0))
    model_fit5 = model5.fit(disp=0)
    print(model_fit5.aic)
