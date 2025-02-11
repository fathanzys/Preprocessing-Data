import pandas as pd
import numpy as np
import matplotlib as pyplot

data = pd.read_csv (r'C:\Users\fathan\OneDrive\Desktop\PERKULIAHAN\Semester 5\AI & ML\tugas besar\Preprocessing-Data\test.csv')
print(data.shape)
print(data)
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.dtypes)
