import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt
import joblib
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

df = pd.read_csv("customer.csv")

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"No": 0.0, "Yes": 1.0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

binary_columns = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
multi_columns = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_columns),
        ('bin', OrdinalEncoder(), binary_columns),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), multi_columns)
    ],
    remainder='passthrough'
)

mlflow.set_experiment("customer_churn_detection")
depths = [2, 4, 6, 8, 10]
best_accuracy = 0
best_model = None

for depth in depths:
    with mlflow.start_run():
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(max_depth=depth, learning_rate=0.05, n_estimators=100, random_state=42))
        ])
        
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Depth:", depth, "Accuracy:", accuracy)
        mlflow.log_param("depth", depth)
        mlflow.log_metric("accuracy", accuracy)
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("confusion_matrix.png")
        plt.close()
        
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.sklearn.log_model(
            model_pipeline,
            name="XGBoost_model",
            input_example=X_train.head(1)
        )
        print(f"Model training with depth {depth} completed!")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_pipeline

joblib.dump(
    {
        "model": best_model,
        "feature_names": X.columns.tolist()
    },
    "customer_churn_model.joblib"
)
print(f"\nBest model (depth={best_model.named_steps['classifier'].max_depth}) saved")
print(f"Accuracy: {best_accuracy:.7f}")
