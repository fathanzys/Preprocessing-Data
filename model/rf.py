import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# Tambahkan path ke folder yang berisi main.py
sys.path.append(r"C:\Users\fathan\OneDrive\Desktop\PERKULIAHAN\Semester 5\AI & ML\tugas besar\Preprocessing-Data")

# Import main.py jika ada fungsi preprocessing yang dibutuhkan
import main  

# Buat dataset contoh (jika belum ada dataset)
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split dataset menjadi train dan test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menggunakan SMOTE untuk oversampling
smote = SMOTE(random_state=42)
x_train_oversample, y_train_oversample = smote.fit_resample(x_train, y_train)
x_test_oversample, y_test_oversample = smote.fit_resample(x_test, y_test)

# Inisialisasi model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model dengan data training yang sudah dioversample
model.fit(x_train_oversample, y_train_oversample)

# Prediksi pada data uji
prediksi = model.predict(x_test_oversample)

# Konversi hasil prediksi ke DataFrame
prediksi_df = pd.DataFrame(prediksi, columns=['hasil prediksi'])
print(prediksi_df)

# Konversi y_test_oversample ke DataFrame
data9 = pd.DataFrame(y_test_oversample)
print(data9)

# Hitung Confusion Matrix
conf_matrix = confusion_matrix(y_test_oversample, prediksi)
print("Confusion Matrix:\n", conf_matrix)

# Hitung Akurasi
accuracy = accuracy_score(y_test_oversample, prediksi)
print("Akurasi:", accuracy)
