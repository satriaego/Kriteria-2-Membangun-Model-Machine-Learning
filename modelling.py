import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

current_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(current_dir, 'mlflow.db')}")

# Atur nama eksperimen dulu
mlflow.set_experiment("Eksperimen_SML_Satria_Ego_Vania")

mlflow.autolog()

print("Memulai proses latihan model...")

data_path = 'preprocessing/Titanic_cleaned_latest.csv'
df = pd.read_csv(data_path)

df_numeric = df.select_dtypes(include=['int64', 'float64'])
X = df_numeric.drop(columns=['Survived'])
y = df_numeric['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Training_RandomForest_Lokal"):
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    

    mlflow.log_metric("testing_accuracy_score", akurasi)
    
    print(f"Model berhasil dilatih! Akurasi: {akurasi:.2%}")
    print("Tercatat otomatis oleh MLflow.")