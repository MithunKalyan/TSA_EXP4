# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("CarPrice_Assignment.csv")

# Select the numerical column
ts = data["price"]

# -----------------------------
# Plot Original Time Series
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(ts)
plt.title("Original Time Series (Price)")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# -----------------------------
# ACF Plot
# -----------------------------
plot_acf(ts, lags=35)
plt.title("Autocorrelation Function (ACF)")
plt.show()

# -----------------------------
# PACF Plot
# -----------------------------
plot_pacf(ts, lags=35, method='ywm')
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# ==========================================================
#                 ARMA(1,1) MODEL
# ==========================================================

print("="*60)
print("ARMA(1,1) MODEL")
print("="*60)

arma11 = ARIMA(ts, order=(1,0,1))
result11 = arma11.fit()

print(result11.summary())

# Predicted Values
pred11 = result11.predict()

# Plot Actual vs Predicted
plt.figure(figsize=(12,5))
plt.plot(ts, label="Actual")
plt.plot(pred11, color="red", label="Predicted")
plt.title("ARMA(1,1) Model")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot
plt.figure(figsize=(12,5))
plt.plot(result11.resid)
plt.title("Residuals - ARMA(1,1)")
plt.xlabel("Observation")
plt.ylabel("Residual")
plt.grid(True)
plt.show()

# ==========================================================
#                 ARMA(2,2) MODEL
# ==========================================================

print("="*60)
print("ARMA(2,2) MODEL")
print("="*60)

arma22 = ARIMA(ts, order=(2,0,2))
result22 = arma22.fit()

print(result22.summary())

# Predicted Values
pred22 = result22.predict()

# Plot Actual vs Predicted
plt.figure(figsize=(12,5))
plt.plot(ts, label="Actual")
plt.plot(pred22, color="green", label="Predicted")
plt.title("ARMA(2,2) Model")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Residual Plot
plt.figure(figsize=(12,5))
plt.plot(result22.resid)
plt.title("Residuals - ARMA(2,2)")
plt.xlabel("Observation")
plt.ylabel("Residual")
plt.grid(True)
plt.show()

# -----------------------------
# ACF and PACF of Residuals - ARMA(1,1)
# -----------------------------

plot_acf(result11.resid, lags=35)
plt.title("ACF of Residuals - ARMA(1,1)")
plt.show()

plot_pacf(result11.resid, lags=35, method='ywm')
plt.title("PACF of Residuals - ARMA(1,1)")
plt.show()

# -----------------------------
# ACF and PACF of Residuals - ARMA(2,2)
# -----------------------------

plot_acf(result22.resid, lags=35)
plt.title("ACF of Residuals - ARMA(2,2)")
plt.show()

plot_pacf(result22.resid, lags=35, method='ywm')
plt.title("PACF of Residuals - ARMA(2,2)")
plt.show()
```

### OUTPUT:

<img width="1013" height="450" alt="image" src="https://github.com/user-attachments/assets/5a7dbc4f-50d7-4d02-a8dd-e2bce61a3acf" />

<img width="1001" height="748" alt="image" src="https://github.com/user-attachments/assets/7e3aa86c-cdbd-4a4f-801e-9be54dec0757" />

<img width="1001" height="460" alt="image" src="https://github.com/user-attachments/assets/7713792d-bc73-4e91-80a8-e6637dec7265" />

<img width="1021" height="457" alt="image" src="https://github.com/user-attachments/assets/35f48603-a85a-4b10-af77-a8e7517eae99" />

<img width="1000" height="448" alt="image" src="https://github.com/user-attachments/assets/e5ab4918-f78b-45fe-b49d-59393d7473bf" />

<img width="1025" height="443" alt="image" src="https://github.com/user-attachments/assets/6cde71e7-ce18-47e4-9d6e-d693b92197b2" />

<img width="1013" height="755" alt="image" src="https://github.com/user-attachments/assets/23cb1617-31ef-4b40-af13-82b7f929d936" />

<img width="998" height="756" alt="image" src="https://github.com/user-attachments/assets/d969f46d-3f80-47e9-94a3-899d67816607" />

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
