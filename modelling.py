import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

current_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file:///{os.path.join(current_dir, 'mlruns')}")

mlflow.sklearn.autolog()

# Atur nama eksperimen
mlflow.set_experiment("Eksperimen_SML_Satria_Ego_Vania")

print("Memulai proses latihan model...")

with mlflow.start_run():

    data_path = 'preprocessing/Titanic_cleaned_latest.csv'
    df = pd.read_csv(data_path)
    
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    
    X = df_numeric.drop(columns=['Survived'])
    y = df_numeric['Survived']
    
    # Bagi data menjadi Train (80%) dan Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi Model
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    mlflow.log_metric("testing_accuracy_score", akurasi)
    
    print(class_name := f"Model berhasil dilatih! Akurasi: {akurasi:.2%}")
    print("Tercatat otomatis oleh MLflow.")