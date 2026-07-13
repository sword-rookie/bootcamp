# 🚀 Customer Churn Prediction using XGBoost, MLflow, FastAPI & AWS

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn based on their account and service information.

The project demonstrates the complete Machine Learning lifecycle, from data preprocessing and model training to cloud deployment using Amazon Web Services.

---

# 📌 Features

- Customer Churn Prediction using XGBoost
- Data Preprocessing and Feature Engineering
- Experiment Tracking using MLflow
- Model Serialization using Joblib
- Model Storage using Amazon S3
- REST API using FastAPI
- Interactive Frontend Dashboard
- Cloud Deployment using AWS Elastic Beanstalk

---

# 🏗 Project Workflow

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
XGBoost Model Training
   │
   ▼
MLflow Experiment Tracking
   │
   ▼
Best Model Selection
   │
   ▼
Save Model (.joblib)
   │
   ▼
Upload Model to Amazon S3
   │
   ▼
FastAPI Backend
   │
   ▼
Elastic Beanstalk Deployment
   │
   ▼
Customer Churn Dashboard
```

---

# 📂 Project Structure

```text
bootcamp(megha)
│
├── customer
│   ├── customer.csv
│   ├── main.py
│   ├── requirements.txt
│   └── mlruns/
│
└── customer-run
    ├── app.py
    ├── Procfile
    ├── requirements.txt
    ├── templates/
    │      └── index.html
    └── static/
           ├── style.css
           └── script.js
```

---

# 🚀 Getting Started

## Prerequisites

- Python 3.11+
- Git
- AWS Account (for deployment)
- AWS CLI (optional)

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd <repository-name>
```

---

# 📊 Phase 1 — Model Training

The `customer` directory contains the complete model training pipeline. By using a separate virtual environment, we avoid installing web framework packages when we are only training models.

## Step 1: Environment Setup

Navigate to the training directory:

```bash
cd customer
```

Create a Virtual Environment:

**Linux / macOS**
```bash
python3 -m venv venv
```

**Windows**
```bash
python -m venv venv
```

Activate the Environment:

**Linux / macOS**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 2: Run Training

Run the training script:

```bash
python main.py
```

The script performs the following operations:

- Loads the dataset
- Cleans missing values
- Encodes categorical features
- Performs train-test split
- Trains multiple XGBoost models
- Evaluates model performance
- Logs each experiment to MLflow
- Saves the best model

The trained model is stored as:

```text
customer_churn_model.joblib
```

---

## Step 3: View Experiments

Launch MLflow to view your experiment results:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui
```

Open your browser to:

```text
http://127.0.0.1:5000
```

MLflow allows you to compare:

- Parameters
- Accuracy
- Confusion Matrix
- Trained Models

---

# 🌐 Phase 2 — Local Web Application

The `customer-run` directory contains the deployed application. We will use a fresh virtual environment specifically for deployment.

## Step 1: Environment Setup

Open a new terminal (or navigate from the root) to the deployment folder:

```bash
cd customer-run
```

Create a Virtual Environment:

**Linux / macOS**
```bash
python3 -m venv venv
```

**Windows**
```bash
python -m venv venv
```

Activate the Environment:

**Linux / macOS**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 2: Run the FastAPI Application

Start the web server using Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open your browser to:

```text
http://127.0.0.1:8000
```

The dashboard allows users to:

- Enter customer details
- Predict customer churn
- View churn probability in real time

---

# ☁️ Phase 3 — Cloud Deployment

After testing locally, the application is deployed to AWS.

---

## Step 1 — Upload Model to Amazon S3

After training completes, upload

```text
customer_churn_model.joblib
```

to an Amazon S3 bucket.

Example:

```text
Amazon S3

model-bootcamp
    └── customer_churn_model.joblib
```

The FastAPI application downloads this model automatically during startup using **boto3**.

---

## Step 2 — Prepare the Deployment Package

The deployment package should contain:

```text
app.py
Procfile
requirements.txt

templates/
static/
```

Do **not** zip the parent folder.

---

## Step 3 — Procfile

Elastic Beanstalk starts the application using:

```text
web: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Step 4 — Create an Elastic Beanstalk Environment

1. Open AWS Console
2. Navigate to Elastic Beanstalk
3. Create a new Application
4. Select the Python Platform
5. Create an Environment
6. Wait for the environment to become available

---

## Step 5 — Deploy

1. Compress the application files into a ZIP archive.
2. Open your Elastic Beanstalk environment.
3. Select **Upload and Deploy**.
4. Choose the ZIP file.
5. Deploy the application.

Elastic Beanstalk automatically:

- Creates an EC2 instance
- Installs dependencies
- Configures Nginx
- Starts Uvicorn
- Hosts the FastAPI application

---

## Step 6 — Access the Application

After deployment, Elastic Beanstalk generates a public URL.

Example:

```text
http://customer-model-env.elasticbeanstalk.com
```

Open the URL to access the Customer Churn Prediction dashboard.

---

# ☁️ Cloud Architecture

```text
                 User
                  │
                  ▼
         Web Browser
                  │
                  ▼
 AWS Elastic Beanstalk
        (FastAPI)
                  │
                  ▼
Downloads Model from
      Amazon S3
                  │
                  ▼
 customer_churn_model.joblib
                  │
                  ▼
      XGBoost Model
                  │
                  ▼
      Prediction Result
                  │
                  ▼
       User Dashboard
```

---

# 🔌 REST API

## Home

```text
GET /
```

Displays the frontend dashboard.

---

## Health Check

```text
GET /health
```

Response:

```json
{
    "status":"healthy"
}
```

---

## Prediction

```text
POST /predict
```

Returns:

```json
{
    "prediction":"Yes",
    "probability":0.8742
}
```

---

# 🛠 Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost

### Experiment Tracking

- MLflow

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Cloud

- Amazon S3
- AWS Elastic Beanstalk

### Deployment

- Joblib
- Procfile

---

# 👨‍💻 Author

**Ayush C N**

Computer Science Engineering (Data Science)

Alva's Institute of Engineering and Technology

---

# 📄 License

This project is intended for educational and learning purposes.
