import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'C:\Users\fathan\OneDrive\Desktop\PERKULIAHAN\Semester 5\AI & ML\tugas besar\Preprocessing-Data\data\Student Depression Dataset.csv')

y = data['Depression']  # Gantilah dengan nama kolom target yang sesuai
x = data.drop(columns=['Depression'])  # Gantilah dengan nama kolom target yang sesuai

x_train_oversample, x_test_oversample, y_train_oversample, y_test_oversample = train_test_split(x, y, test_size=0.10, random_state=1)

print("Ukuran x_train :", x_train_oversample.shape)
print("Ukuran y_train :", y_train_oversample.shape)
print("Ukuran x_test :", x_test_oversample.shape)
print("Ukuran y_test :", y_test_oversample.shape)