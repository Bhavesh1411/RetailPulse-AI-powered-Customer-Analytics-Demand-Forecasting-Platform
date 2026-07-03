import os
import pandas as pd
import numpy as np

filepath = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'Demand_Forecasting/datasets/daily_sales_forecast_features.csv')
df = pd.read_csv(filepath)
print("Columns:", list(df.columns))
print("Shape:", df.shape)
print("Min Date:", df['date'].min())
print("Max Date:", df['date'].max())
print("Null count:\n", df.isnull().sum())
