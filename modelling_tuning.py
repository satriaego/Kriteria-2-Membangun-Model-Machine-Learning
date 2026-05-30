import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file:///{os.path.join(current_dir, 'mlruns')}")

# Set nama eksperimen
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

print("Hyperparameter Tuning")

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
        
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        print(f"   Score -> Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

print("\nHyperparameter tuning selesai")