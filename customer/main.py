import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"No": 0.0, "Yes": 1.0})

le = LabelEncoder()
binary_columns = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_columns:
    df[col] = le.fit_transform(df[col])

multi_columns = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
df = pd.get_dummies(df, columns=multi_columns, drop_first=True)

X = df.drop("Churn", axis=1).astype(float)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("customer_churn_detection")
depths = [2, 4, 6, 8, 10]
best_accuracy = 0
best_model = None

for depth in depths:
    with mlflow.start_run():
        model = XGBClassifier(max_depth=depth, learning_rate=0.05, n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
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
            model,
            name="XGBoost_model",
            input_example=X_train,
            skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
        )
        print(f"Model training with depth {depth} completed!")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

joblib.dump(
    {
        "model": best_model,
        "feature_names": X.columns.tolist()
    },
    "customer_churn_model.joblib"
)
print(f"\nBest model (depth={best_model.max_depth}) saved")
print(f"Accuracy: {best_accuracy:.7f}")
