import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


df = pd.read_csv('Advertising.csv')
print(df.head())



