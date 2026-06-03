import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, log_loss, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import dagshub

TARGET_TRACKING = "LOCAL"

if len(sys.argv) > 1 and sys.argv[1] == "--dagshub":
    TARGET_TRACKING = "DAGSHUB"

print(f"📡 Mengonfigurasi MLflow Tracking ke mode: [{TARGET_TRACKING}]")

if TARGET_TRACKING == "DAGSHUB":
    dagshub.init(
        repo_owner='satriaego',
        repo_name='Kriteria-2-Membangun-Model-Machine-Learning',
        mlflow=True
    )
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(current_dir, 'mlflow.db')}")

mlflow.set_experiment("Eksperimen_SML_Satria_Ego_Vania_Tuning")

data_path = 'preprocessing/Titanic_cleaned_latest.csv'
df = pd.read_csv(data_path)

df_numeric = df.select_dtypes(include=['int64', 'float64'])
X = df_numeric.drop(columns=['Survived'])
y = df_numeric['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dataset_mlflow: PandasDataset = mlflow.data.from_pandas(
    df=df_numeric, 
    targets="Survived", 
    name="Titanic_Train_Cleaned"
)

kombinasi_tuning = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10}
]

print("\nHyperparameter Tuning")

for i, params in enumerate(kombinasi_tuning):
    run_name = f"Manual_Tuning_Iterasi_{i+1}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\n🚀 Menjalankan {run_name} dengan params: {params}")
        mlflow.log_input(dataset_mlflow, context="training")
        
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"], 
            max_depth=params["max_depth"], 
            random_state=42
        )
        
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        train_prec = precision_score(y_train, y_train_pred, average='macro')
        train_rec = recall_score(y_train, y_train_pred, average='macro')
        train_loss = log_loss(y_train, y_train_prob)
        train_roc = roc_auc_score(y_train, y_train_prob[:, 1]) 
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        mlflow.log_metric("training_accuracy_score", train_acc)
        mlflow.log_metric("training_f1_score", train_f1)
        mlflow.log_metric("training_log_loss", train_loss)
        mlflow.log_metric("training_precision_score", train_prec)
        mlflow.log_metric("training_recall_score", train_rec)
        mlflow.log_metric("training_roc_auc", train_roc)
        mlflow.log_metric("training_score", train_acc)
        mlflow.log_metric("testing_accuracy_score", test_acc)
        # -----------------------------------------------------------

        mlflow.sklearn.log_model(model, artifact_path="model")

        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Survived', 'Survived'], 
                    yticklabels=['Not Survived', 'Survived'])
        plt.title(f"Confusion Matrix - Iterasi {i+1}")
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        
        nama_gambar = f"confusion_matrix_iterasi_{i+1}.png"
        plt.tight_layout()
        plt.savefig(nama_gambar)
        plt.close()
        
        mlflow.log_artifact(nama_gambar, artifact_path="plots_evaluasi")
        
        if (params["n_estimators"] == 100 and params["max_depth"] == 5) or \
           (params["n_estimators"] == 200 and params["max_depth"] == 10):
            
            import joblib
            
            nama_file_model = f"compiled_model_iterasi_{i+1}.pkl"
            
            joblib.dump(model, nama_file_model)
            
            mlflow.log_artifact(nama_file_model, artifact_path="saved_compiled_models")
            
            if os.path.exists(nama_file_model): 
                os.remove(nama_file_model)
                
            print(f"   ↳ 📦 Artefak Tambahan: {nama_file_model} berhasil diunggah ke DagsHub!")
        
        print(f"   Score -> Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

print("\nHyperparameter tuning selesai!")