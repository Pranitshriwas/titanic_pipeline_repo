# Cell 1: Preprocessing and saving transformed dataimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os
import pandas as pd 

# Load preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Use already one-hot encoded columns
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Handle missing values (if any remain)
X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Fare'] = X['Fare'].fillna(X['Fare'].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training and logging with MLflow
mlflow.set_experiment("Titanic_LogReg")


with mlflow.start_run():
   model = LogisticRegression(max_iter=500)
   model.fit(X_train, y_train)


   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)


   mlflow.log_param("model_type", "LogisticRegression")
   mlflow.log_param("max_iter", 500)
   mlflow.log_metric("accuracy", accuracy)
   mlflow.sklearn.log_model(model, "model")


   print("Accuracy:", accuracy)
   print(classification_report(y_test, y_pred))


   os.makedirs('model', exist_ok=True)
   joblib.dump(model, 'model/logreg_model.pkl')
