import sys
import pandas as pd
import numpy as np

sys.path.append(r"C:\Users\fathan\OneDrive\Desktop\PERKULIAHAN\Semester 5\AI & ML\tugas besar\Preprocessing-Data\main")

import main
print(main.data.drop('id', axis=1, inplace=True))
print(main.data.drop('City', axis=1, inplace=True))
print(main.data.drop('Degree', axis=1, inplace=True))
print(main.data.head())

y = main.data.pop ('Depression')
print(main.data.shape)
print(y.head())