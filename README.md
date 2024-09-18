### DEVELOPED BY: AJAY ASWIN M
### REGISTER NO: 212222240005
### DATE:


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
3. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
4. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
5. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

# PROGRAM:
```python
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load your Petrol Price data
petrol_price_data = pd.read_csv('petrol.csv')

# Convert 'Date' column to datetime and set it as the index
petrol_price_data['Date'] = pd.to_datetime(petrol_price_data['Date'])
petrol_price_data.set_index('Date', inplace=True)

# Resample the 'Chennai' column by day to get daily price
daily_price = petrol_price_data['Chennai'].resample('D').sum()

# Simulating an ARMA(1,1) Process
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
plt.show()

# Simulating an ARMA(2,2) Process
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=1000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
plt.show()



```


# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:


![Screenshot 2024-09-18 093647](https://github.com/user-attachments/assets/33cd9979-98ed-4048-bb7c-9dfcebfa6558)

## Partial Autocorrelation


![Screenshot 2024-09-18 093743](https://github.com/user-attachments/assets/5b5e3646-e3d9-4c6a-b1cd-cd9cda1c4fd6)


## Autocorrelation

![Screenshot 2024-09-18 093719](https://github.com/user-attachments/assets/bdcc50be-ba5f-44bb-a6e9-c3e72401d60a)



## SIMULATED ARMA(2,2) PROCESS:

![Screenshot 2024-09-18 093758](https://github.com/user-attachments/assets/d7357365-7a1d-4368-8a7f-8267a309a5b7)


## Partial Autocorrelation

![Screenshot 2024-09-18 093828](https://github.com/user-attachments/assets/b0bb0bc1-0749-4f38-b61e-f1861b141b43)


## Autocorrelation

![Screenshot 2024-09-18 093813](https://github.com/user-attachments/assets/6f738a32-200a-4184-893a-a156dc86a541)


# RESULT:
Thus, a python program is created to fit ARMA Model for Time Series successfully.
