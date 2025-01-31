import sys
import pandas as pd
import numpy as np

sys.path.append(r"C:\Users\fathan\OneDrive\Desktop\PERKULIAHAN\Semester 5\AI & ML\tugas besar\Preprocessing-Data\main")

import main

data = main.data

data['Financial Stress'] = data['Financial Stress'].fillna(data['Financial Stress'].mean())

print(data['Financial Stress'])
print(data.isnull().sum())
